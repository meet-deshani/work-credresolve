"""Streamlit app for analyzing conversion KPI correlations."""
from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = {"lead_code", "conversion_flag"}


def validate_dataframe(df: pd.DataFrame) -> List[str]:
    """Run basic sanity checks on the uploaded data."""

    errors: List[str] = []

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        errors.append(
            "Missing required columns: " + ", ".join(sorted(missing))
        )
        # No need to continue further validation if required columns are missing.
        return errors

    invalid_flags = ~df["conversion_flag"].isin([0, 1])
    if invalid_flags.any():
        errors.append("`conversion_flag` must contain only 0 or 1.")

    duplicate_leads = df["lead_code"].duplicated().sum()
    if duplicate_leads:
        errors.append(
            f"Found {duplicate_leads} duplicated lead_code values."
        )

    return errors


def render_data_health(df: pd.DataFrame) -> None:
    """Display basic data health metrics after validation."""
    total_rows = len(df)
    total_conversions = int(df["conversion_flag"].sum())
    conversion_rate = df["conversion_flag"].mean()

    st.subheader("Data Summary")
    lead_col, conv_col, rate_col = st.columns(3)
    lead_col.metric("Leads", f"{total_rows:,}")
    conv_col.metric("Conversions", f"{total_conversions:,}")
    rate_col.metric("Conversion rate", f"{conversion_rate:.2%}")

    st.caption(
        "The conversion rate is calculated as conversions divided by total leads."
    )

    st.markdown("### Missing value overview")
    st.dataframe(
        df.isna().sum().rename("missing_count").to_frame(),
        use_container_width=True,
    )


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def numeric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation between numeric fields and the conversion flag."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference(
        ["conversion_flag"]
    )
    if numeric_cols.empty:
        return pd.DataFrame(columns=["parameter", "correlation"])

    corr_series = df[numeric_cols].corrwith(df["conversion_flag"])
    corr_df = (
        corr_series.dropna()
        .rename_axis("parameter")
        .reset_index(name="correlation")
        .sort_values(by="correlation", key=lambda x: x.abs(), ascending=False)
    )
    return corr_df


def parameter_summary(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Summarise conversion performance for a single parameter."""
    grouped = (
        df.groupby(parameter)["conversion_flag"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "leads", "mean": "conversion_rate", "sum": "conversions"})
    )
    return grouped.reset_index()


def combination_summary(df: pd.DataFrame, parameters: Iterable[str]) -> pd.DataFrame:
    """Summarise conversion performance for a parameter combination."""
    grouped = (
        df.groupby(list(parameters))["conversion_flag"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "leads", "mean": "conversion_rate", "sum": "conversions"})
        .reset_index()
    )
    return grouped


def compute_focus_recommendations(
    df: pd.DataFrame,
    available_columns: Iterable[str],
    *,
    min_leads: int = 30,
) -> pd.DataFrame:
    """Identify parameter attributes that over- or under-perform versus the overall rate."""

    overall_rate = df["conversion_flag"].mean()
    focus_rows = []

    for parameter in available_columns:
        series = df[parameter]

        if series.nunique(dropna=False) <= 1:
            continue

        if pd.api.types.is_numeric_dtype(series):
            unique_non_na = series.dropna().nunique()
            if unique_non_na <= 1:
                continue
            try:
                buckets = pd.qcut(
                    series,
                    q=min(4, unique_non_na),
                    duplicates="drop",
                )
            except ValueError:
                continue

            summary = (
                df.assign(_bucket=buckets)
                .groupby("_bucket")["conversion_flag"]
                .agg(["count", "mean", "sum"])
                .rename(
                    columns={
                        "count": "leads",
                        "mean": "conversion_rate",
                        "sum": "conversions",
                    }
                )
                .reset_index()
                .rename(columns={"_bucket": "attribute"})
            )
            summary["attribute"] = summary["attribute"].astype(str)
        else:
            summary = (
                df.groupby(parameter)["conversion_flag"]
                .agg(["count", "mean", "sum"])
                .rename(
                    columns={
                        "count": "leads",
                        "mean": "conversion_rate",
                        "sum": "conversions",
                    }
                )
                .reset_index()
                .rename(columns={parameter: "attribute"})
            )

        summary = summary[summary["leads"] >= min_leads]
        if summary.empty:
            continue

        summary["parameter"] = parameter
        summary["lift_vs_overall"] = summary["conversion_rate"] - overall_rate
        summary["focus_direction"] = summary["lift_vs_overall"].apply(
            lambda lift: "Outperforming" if lift > 0 else "Underperforming"
        )

        focus_rows.append(summary)

    if not focus_rows:
        return pd.DataFrame(
            columns=
            [
                "parameter",
                "attribute",
                "leads",
                "conversions",
                "conversion_rate",
                "lift_vs_overall",
                "focus_direction",
            ]
        )

    focus_df = pd.concat(focus_rows, ignore_index=True)
    return focus_df[
        [
            "parameter",
            "attribute",
            "leads",
            "conversions",
            "conversion_rate",
            "lift_vs_overall",
            "focus_direction",
        ]
    ]


def compute_combination_focus(
    df: pd.DataFrame,
    available_columns: Iterable[str],
    *,
    min_leads: int = 30,
    max_combination_size: int = 3,
    max_combinations: int = 250,
) -> Tuple[pd.DataFrame, bool, int]:
    """Summarize which parameter combinations outperform or underperform overall."""

    overall_rate = df["conversion_flag"].mean()
    combinations_evaluated = 0
    truncated = False
    focus_rows: List[pd.DataFrame] = []

    available = list(available_columns)
    if len(available) < 2:
        return pd.DataFrame(), truncated, combinations_evaluated

    for size in range(2, min(max_combination_size, len(available)) + 1):
        for combo in itertools.combinations(available, size):
            combinations_evaluated += 1
            if combinations_evaluated > max_combinations:
                truncated = True
                break

            summary = combination_summary(df, combo)
            summary = summary[summary["leads"] >= min_leads]
            if summary.empty:
                continue

            combo_label = " + ".join(combo)
            attribute = (
                summary[list(combo)]
                .astype(str)
                .agg(" | ".join, axis=1)
            )

            combo_focus = summary.assign(
                parameter_combination=combo_label,
                attribute=attribute,
                combination_size=size,
                lift_vs_overall=summary["conversion_rate"] - overall_rate,
            )

            combo_focus["focus_direction"] = combo_focus["lift_vs_overall"].apply(
                lambda lift: "Outperforming" if lift > 0 else "Underperforming"
            )

            focus_rows.append(
                combo_focus[
                    [
                        "parameter_combination",
                        "attribute",
                        "combination_size",
                        "leads",
                        "conversions",
                        "conversion_rate",
                        "lift_vs_overall",
                        "focus_direction",
                    ]
                ]
            )
        if truncated:
            break

    if not focus_rows:
        return pd.DataFrame(), truncated, combinations_evaluated

    focus_df = pd.concat(focus_rows, ignore_index=True)
    return focus_df, truncated, combinations_evaluated


def render_single_parameter_view(df: pd.DataFrame, available_columns: List[str]) -> None:
    st.header("Single Parameter Performance")

    parameter = st.selectbox(
        "Select parameter",
        options=available_columns,
        key="single_parameter_choice",
    )

    if not parameter:
        return

    summary = parameter_summary(df, parameter).sort_values(
        by="conversion_rate", ascending=False
    )
    st.dataframe(summary, use_container_width=True)
    st.caption(
        "Conversion rate equals conversions divided by leads for the selected parameter."
    )

    st.markdown("---")
    st.subheader("Numeric Parameter Correlation")
    corr_df = numeric_correlation(df)
    if corr_df.empty:
        st.write("No numeric parameters available for correlation analysis.")
    else:
        st.dataframe(corr_df, use_container_width=True)


def render_combination_view(df: pd.DataFrame, available_columns: List[str]) -> None:
    st.header("Parameter Combination Performance")

    st.write(
        "Select two or more parameters to understand how their combinations influence conversions."
    )

    parameters = st.multiselect(
        "Select parameters for combination",
        options=available_columns,
        key="combination_parameters",
    )

    if len(parameters) < 2:
        st.info("Select at least two parameters to analyze combinations.")
        return

    max_combination_size = min(len(parameters), 4)
    for combo_size in range(2, max_combination_size + 1):
        st.markdown(f"### {combo_size}-way combinations")
        for combo in itertools.combinations(parameters, combo_size):
            st.markdown("**" + ", ".join(combo) + "**")
            combo_summary = combination_summary(df, combo).sort_values(
                by="conversion_rate", ascending=False
            )
            st.dataframe(combo_summary, use_container_width=True)


def render_focus_recommendations(df: pd.DataFrame, available_columns: List[str]) -> None:
    st.header("Focus Recommendations")
    st.write(
        "Identify the parameter attributes that most improve or hurt conversion compared to the overall rate."
    )

    min_leads = st.slider(
        "Minimum leads per attribute",
        min_value=1,
        max_value=500,
        value=30,
        help="Ignore parameter attributes that have fewer leads than this threshold.",
    )

    top_n = st.slider(
        "Show top insights per group",
        min_value=1,
        max_value=10,
        value=3,
        help=(
            "Limit the number of high- and low-performing attributes or combinations "
            "displayed for each group."
        ),
    )

    focus_df = compute_focus_recommendations(
        df,
        available_columns,
        min_leads=min_leads,
    )

    if focus_df.empty:
        st.info("No parameter attributes met the minimum lead threshold for analysis.")
        return

    overall_rate = df["conversion_flag"].mean()
    st.metric("Overall conversion rate", f"{overall_rate:.2%}")

    outperform = (
        focus_df[focus_df["lift_vs_overall"] > 0]
        .sort_values(["lift_vs_overall", "conversion_rate"], ascending=False)
        .groupby("parameter", group_keys=False)
        .head(top_n)
    )

    underperform = (
        focus_df[focus_df["lift_vs_overall"] <= 0]
        .sort_values(["lift_vs_overall", "conversion_rate"], ascending=[True, True])
        .groupby("parameter", group_keys=False)
        .head(top_n)
    )

    if not outperform.empty:
        st.markdown("### High-performing attributes")
        st.dataframe(
            outperform,
            use_container_width=True,
        )
    else:
        st.info("No attributes are currently outperforming the overall conversion rate.")

    if not underperform.empty:
        st.markdown("### Low-performing attributes")
        st.dataframe(
            underperform,
            use_container_width=True,
        )
    else:
        st.info("No attributes are currently underperforming the overall conversion rate.")

    st.markdown("---")
    st.subheader("Parameter combination insights")

    if len(available_columns) < 2:
        st.info(
            "Add at least two eligible parameters to evaluate combination-focused insights."
        )
        return

    max_combo_size = st.slider(
        "Maximum combination size",
        min_value=2,
        max_value=min(4, len(available_columns)),
        value=min(3, len(available_columns)),
        help="Broaden or narrow the search for multi-parameter opportunities.",
    )

    combo_focus_df, truncated, combos_evaluated = compute_combination_focus(
        df,
        available_columns,
        min_leads=min_leads,
        max_combination_size=max_combo_size,
    )

    if combos_evaluated == 0:
        st.info("Add at least two eligible parameters to evaluate combinations.")
        return

    if combo_focus_df.empty:
        st.info(
            "No parameter combinations met the minimum lead threshold for analysis."
        )
        return

    if truncated:
        st.warning(
            "Displayed results are truncated to keep analysis responsive. "
            "Reduce the maximum combination size or filter columns for a complete view."
        )

    outperform_combos = (
        combo_focus_df[combo_focus_df["lift_vs_overall"] > 0]
        .sort_values(["combination_size", "lift_vs_overall", "conversion_rate"], ascending=[True, False, False])
        .groupby("combination_size", group_keys=False)
        .head(top_n)
    )

    underperform_combos = (
        combo_focus_df[combo_focus_df["lift_vs_overall"] <= 0]
        .sort_values(["combination_size", "lift_vs_overall", "conversion_rate"], ascending=[True, True, True])
        .groupby("combination_size", group_keys=False)
        .head(top_n)
    )

    if not outperform_combos.empty:
        st.markdown("### High-performing combinations")
        st.dataframe(outperform_combos, use_container_width=True)
    else:
        st.info("No parameter combinations are outperforming the overall conversion rate.")

    if not underperform_combos.empty:
        st.markdown("### Combinations needing attention")
        st.dataframe(underperform_combos, use_container_width=True)
    else:
        st.info("No parameter combinations are underperforming the overall conversion rate.")


def reset_validation_state(current_file) -> None:
    """Clear cached validation results when a new file is uploaded."""
    file_info = (
        getattr(current_file, "name", None),
        getattr(current_file, "size", None),
        getattr(current_file, "type", None),
    )

    if st.session_state.get("uploaded_file_info") != file_info:
        st.session_state["validated_df"] = None
        st.session_state["validation_errors"] = []
        st.session_state["uploaded_file_info"] = file_info


def render_upload_and_validation() -> None:
    st.header("Data Upload & Validation")
    st.write(
        "Upload a CSV file that includes the required `lead_code` and `conversion_flag` columns. "
        "The `conversion_flag` should be 0 for non-converted leads and 1 for converted leads."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type="csv",
        key="uploaded_file",
    )

    reset_validation_state(uploaded_file)

    if st.button("Validate Data", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a CSV file before validating.")
        else:
            try:
                df = load_csv(uploaded_file)
            except Exception as exc:  # pragma: no cover - defensive programming
                st.session_state["validated_df"] = None
                st.session_state["validation_errors"] = [f"Unable to read CSV: {exc}"]
            else:
                errors = validate_dataframe(df)
                if errors:
                    st.session_state["validated_df"] = None
                    st.session_state["validation_errors"] = errors
                else:
                    st.session_state["validated_df"] = df
                    st.session_state["validation_errors"] = []

    validation_errors = st.session_state.get("validation_errors", [])
    validated_df = st.session_state.get("validated_df")

    if validation_errors:
        st.error("Validation failed:")
        for err in validation_errors:
            st.write(f"- {err}")
        return

    if validated_df is None:
        st.info("Upload a CSV file and click **Validate Data** to run sanity checks.")
        return

    st.success("Validation successful! Explore the summaries below.")
    render_data_health(validated_df)

    st.markdown("### Data preview")
    st.dataframe(validated_df.head(50), use_container_width=True)


def get_validated_dataframe() -> pd.DataFrame | None:
    df = st.session_state.get("validated_df")
    if df is None:
        st.info(
            "Please upload and validate your CSV in the **Data Upload & Validation** section first."
        )
    return df


def main() -> None:
    st.set_page_config(
        page_title="Conversion KPI Analyzer",
        layout="wide",
    )
    st.title("Conversion KPI Analyzer")
    st.write(
        "Upload your lead conversion data to explore how different parameters influence the conversion flag."
    )

    if "validated_df" not in st.session_state:
        st.session_state["validated_df"] = None
        st.session_state["validation_errors"] = []
        st.session_state["uploaded_file_info"] = None

    with st.sidebar:
        st.header("Navigation")
        menu_choice = st.radio(
            "Go to",
            options=[
                "Data Upload & Validation",
                "Single Parameter Explorer",
                "Combination Explorer",
                "Focus Recommendations",
            ],
        )
        st.caption("Use this navigation to move between analysis steps.")

    if menu_choice == "Data Upload & Validation":
        render_upload_and_validation()
        return

    df = get_validated_dataframe()
    if df is None:
        return

    available_columns = [
        col for col in df.columns if col not in REQUIRED_COLUMNS
    ]
    if not available_columns:
        st.warning("No additional parameters found besides the required fields.")
        return

    if menu_choice == "Single Parameter Explorer":
        render_single_parameter_view(df, available_columns)
    elif menu_choice == "Combination Explorer":
        render_combination_view(df, available_columns)
    elif menu_choice == "Focus Recommendations":
        render_focus_recommendations(df, available_columns)


if __name__ == "__main__":
    main()
