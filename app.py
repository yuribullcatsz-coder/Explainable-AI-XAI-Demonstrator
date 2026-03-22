"""
Explainable AI (XAI) Demonstrator — Streamlit entrypoint.

Run: streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from xai_demo.bias import BiasReport, compute_group_metrics, predictions_for_frame
from xai_demo.explain import lime_explanation_figure, partial_dependence_figure, shap_summary_plot
from xai_demo.pipeline import TrainResult, list_columns, train_model


def _load_sample_iris() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(
        {0: "setosa", 1: "versicolor", 2: "virginica"}
    )
    return df


def _default_categoricals(df: pd.DataFrame, target: str) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c]):
            out.append(str(c))
    return out


def _metrics_block(result: TrainResult, X_test: pd.DataFrame, y_test: Any) -> None:
    pred = result.model.predict(X_test)
    if result.task == "classification":
        acc = accuracy_score(y_test, pred)
        st.metric("Test accuracy", f"{acc:.3f}")
    else:
        st.metric("Test R²", f"{r2_score(y_test, pred):.3f}")
        st.metric("Test MAE", f"{mean_absolute_error(y_test, pred):.4f}")


def main() -> None:
    st.set_page_config(
        page_title="XAI Demonstrator",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Explainable AI (XAI) Demonstrator")
    st.caption(
        "Upload a CSV or use the sample dataset, train a Random Forest, then explore "
        "SHAP, partial dependence, LIME, and simple group fairness metrics."
    )

    with st.sidebar:
        st.header("1. Data")
        source = st.radio(
            "Data source",
            ["Upload CSV", "Sample: Iris"],
            horizontal=False,
        )
        uploaded: pd.DataFrame | None = None
        if source == "Upload CSV":
            f = st.file_uploader("CSV file", type=["csv"])
            if f is not None:
                try:
                    uploaded = pd.read_csv(f)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read CSV: {e}")
        else:
            uploaded = _load_sample_iris()
            st.success("Using built-in Iris sample (classification).")

        st.header("2. Columns")
        target_col = st.text_input(
            "Target column name",
            value="species" if source == "Sample: Iris" else "",
            help="The column the model should predict.",
        )
        categorical_cols = st.multiselect(
            "Categorical feature columns",
            options=list_columns(uploaded) if uploaded is not None else [],
            default=_default_categoricals(uploaded, target_col)
            if uploaded is not None and target_col in uploaded.columns
            else [],
            help="These columns are one-hot encoded. Others are treated as numeric.",
        )

        train_clicked = st.button("Train model", type="primary", use_container_width=True)

    if uploaded is None or not target_col:
        st.info("Choose a data source and enter the **target column** name in the sidebar.")
        st.markdown(
            "### Quick start\n"
            "1. Pick **Sample: Iris** (or upload a CSV).\n"
            "2. Set **Target** to `species` for the sample.\n"
            "3. Click **Train model**.\n"
            "4. Open the tabs for SHAP, PDP, LIME, and bias checks.\n"
        )
        return

    if target_col not in uploaded.columns:
        st.error(f"Target column {target_col!r} not found. Columns: {list(uploaded.columns)}")
        return

    if train_clicked:
        try:
            with st.spinner("Training…"):
                res, X_train, X_test, y_train, y_test = train_model(
                    uploaded,
                    target_col,
                    categorical_cols=categorical_cols or None,
                )
            st.session_state["bundle"] = {
                "result": res,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "df": uploaded,
                "target_col": target_col,
            }
            st.success(f"Trained **{res.task}** model on {len(X_train)} train / {len(X_test)} test rows.")
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("bundle", None)
            st.error(str(e))
            return

    bundle = st.session_state.get("bundle")
    if not bundle:
        st.warning('Click **Train model** in the sidebar when you are ready.')
        return

    result: TrainResult = bundle["result"]
    X_train: pd.DataFrame = bundle["X_train"]
    X_test: pd.DataFrame = bundle["X_test"]
    y_test: Any = bundle["y_test"]
    df: pd.DataFrame = bundle["df"]
    target_col = bundle["target_col"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Task", result.task)
    with c2:
        st.metric("Features", len(result.feature_names))
    with c3:
        _metrics_block(result, X_test, y_test)

    tab_shap, tab_pdp, tab_lime, tab_bias, tab_data = st.tabs(
        ["SHAP", "Partial dependence", "LIME", "Bias check", "Data preview"]
    )

    with tab_shap:
        st.markdown(
            "**SHAP** (TreeExplainer) shows how each *encoded* feature pushes predictions. "
            "One-hot columns appear as `column=value`."
        )
        max_s = st.slider("Max rows for SHAP sample", 50, 800, 300, 50)
        if st.button("Generate SHAP summary", key="btn_shap"):
            with st.spinner("Computing SHAP…"):
                img, err = shap_summary_plot(
                    result.model,
                    X_test,
                    max_samples=max_s,
                )
            if err:
                st.error(err)
            elif img:
                st.image(img, caption="SHAP summary (test sample)")

    with tab_pdp:
        st.markdown(
            "Shows average model response vs one **raw** column (sklearn partial dependence)."
        )
        feats = list(X_train.columns)
        if not feats:
            st.warning("No features available.")
        else:
            f_pdp = st.selectbox("Feature for PDP", feats, key="pdp_feat")
            if st.button("Plot partial dependence", key="btn_pdp"):
                with st.spinner("Plotting…"):
                    img, err = partial_dependence_figure(
                        result.model, X_train, f_pdp
                    )
                if err:
                    st.error(err)
                elif img:
                    st.image(img, caption=f"Partial dependence: {f_pdp}")

    with tab_lime:
        st.markdown(
            "**LIME** explains one instance using a local linear surrogate. "
            "Categoricals are label-encoded internally for perturbations."
        )
        n_inst = len(X_test)
        idx = st.number_input(
            "Test row index (0-based)",
            min_value=0,
            max_value=max(0, n_inst - 1),
            value=0,
            step=1,
        )
        if st.button("Explain instance with LIME", key="btn_lime"):
            row = X_test.iloc[[idx]]
            with st.spinner("Running LIME…"):
                img, err = lime_explanation_figure(
                    result.model,
                    X_train,
                    row,
                    result.task,
                )
            if err:
                st.error(err)
            elif img:
                pred = result.model.predict(row)[0]
                st.caption(f"Model prediction for this row: **{pred}**")
                st.image(img, caption="LIME explanation")

    with tab_bias:
        st.markdown(
            "**Bias check** (binary groups only): compares predicted positive rates between "
            "two groups on the **test set**. For classification, “positive” is the lexicographically "
            "last class unless you override below. This is a simple sanity check, not legal advice."
        )
        if result.task != "classification":
            st.info("Bias tab is intended for **classification**. Train a classifier to use it.")
        else:
            sens_choices = [c for c in df.columns if c != target_col]
            sens = st.selectbox("Sensitive attribute column", sens_choices, key="bias_sens")
            classes = sorted(y_test.astype(str).unique().tolist())
            pos = st.selectbox(
                "Positive class (for rate calculation)",
                classes,
                index=len(classes) - 1,
            )
            if st.button("Compute group metrics", key="btn_bias"):
                s_test = df.loc[X_test.index, sens].reset_index(drop=True)
                preds = predictions_for_frame(result.model, X_test)
                rep: BiasReport = compute_group_metrics(
                    preds, s_test, positive_label=str(pos)
                )
                if rep.message:
                    st.warning(rep.message)
                else:
                    st.write(
                        f"Group rates (positive = **{pos}**), then statistical parity "
                        "difference and disparate impact ratio."
                    )
                    gcols = sorted(s_test.astype(str).unique().tolist())
                    st.table(
                        pd.DataFrame(
                            {
                                "Group": gcols[:2],
                                "N": [rep.n_group_a, rep.n_group_b],
                                "P(predicted positive)": [
                                    rep.positive_rate_group_a,
                                    rep.positive_rate_group_b,
                                ],
                            }
                        )
                    )
                    st.metric("Statistical parity difference", f"{rep.statistical_parity_difference:.4f}")
                    di = rep.disparate_impact_ratio
                    st.metric(
                        "Disparate impact ratio (group B / group A)",
                        "—" if di is None else f"{di:.4f}",
                    )

    with tab_data:
        st.subheader("Raw data (first 200 rows)")
        st.dataframe(df.head(200), use_container_width=True)
        buf = io.StringIO()
        df.head(500).to_csv(buf, index=False)
        st.download_button(
            "Download preview CSV",
            data=buf.getvalue().encode(),
            file_name="data_preview.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
