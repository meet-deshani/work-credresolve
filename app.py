"""Streamlit app for analyzing conversion KPI correlations."""
from __future__ import annotations

import itertools
from typing import Iterable, List

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
    st.subheader("Data Summary")
    st.write(
        {
            "Total Rows": len(df),
            "Total Columns": len(df.columns),
            "Missing Values": int(df.isna().sum().sum()),
        }
    )

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


def display_parameter_analysis(df: pd.DataFrame, available_columns: List[str]) -> None:
    st.subheader("Parameter Performance")

    analysis_mode = st.radio(
        "Analysis Mode",
        options=["Single Parameter", "Parameter Combination"],
        horizontal=True,
    )

    if analysis_mode == "Single Parameter":
        parameter = st.selectbox("Select parameter", options=available_columns)
        if parameter:
            st.dataframe(parameter_summary(df, parameter), use_container_width=True)
    else:
        parameters = st.multiselect(
            "Select parameters for combination",
            options=available_columns,
        )

        if len(parameters) >= 2:
            max_combination_size = min(len(parameters), 4)
            st.markdown("### Combination Summaries")
            for combo_size in range(2, max_combination_size + 1):
                st.markdown(f"#### {combo_size}-way combinations")
                for combo in itertools.combinations(parameters, combo_size):
                    st.markdown("**" + ", ".join(combo) + "**")
                    st.dataframe(
                        combination_summary(df, combo),
                        use_container_width=True,
                    )
        else:
            st.info("Select at least two parameters to analyze combinations.")

    st.markdown("---")
    st.subheader("Numeric Parameter Correlation")
    corr_df = numeric_correlation(df)
    if corr_df.empty:
        st.write("No numeric parameters available for correlation analysis.")
    else:
        st.dataframe(corr_df, use_container_width=True)


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
        st.header("Analysis Controls")
        menu_choice = st.selectbox(
            "Select view",
            options=["Conversion KPI"],
        )

        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        validate = st.button("Validate File", type="primary")

    reset_validation_state(uploaded_file)

    if menu_choice == "Conversion KPI":
        if uploaded_file is None:
            st.info("Upload a CSV file containing your lead data to begin.")
            return

        if validate:
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
            st.info("Click 'Validate File' after uploading to run sanity checks.")
            return

        st.success("Validation successful!")
        render_data_health(validated_df)

        available_columns = [
            col
            for col in validated_df.columns
            if col not in REQUIRED_COLUMNS
        ]

        if not available_columns:
            st.warning(
                "No additional parameters found besides the required fields."
            )
            return

        display_parameter_analysis(validated_df, available_columns)


if __name__ == "__main__":
    main()
