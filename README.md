# Explainable AI (XAI) Demonstrator

A small [Streamlit](https://streamlit.io/) app for teaching and prototyping **model explanations**: you load a table (CSV or a built-in sample), train a **Random Forest** (classification or regression), then inspect **SHAP**, **partial dependence**, **LIME**, and simple **group fairness** metrics.

The previous repository revision only contained a placeholder `app.py`. This version adds a working UI, a testable `xai_demo` package, automated tests, and CI.

## What you get

| Area | Details |
|------|---------|
| **Model** | Scikit-learn `Pipeline`: imputation → scaling (numeric) / one-hot (categorical) → Random Forest |
| **SHAP** | `TreeExplainer` on the **preprocessed** matrix; multiclass problems use the mean of \|SHAP\| across classes for one summary plot |
| **Partial dependence** | Sklearn `PartialDependenceDisplay` on **raw** column names (works with the full pipeline) |
| **LIME** | Local explanations on **raw** rows; non-numeric columns are label-encoded for perturbations, then decoded before calling the pipeline |
| **Bias check** | For **binary** sensitive groups on the test set: positive prediction rate per group, statistical parity difference, disparate impact ratio (no AIF360 dependency) |

## Requirements

- **Python 3.10+** (CI uses 3.11)
- OS: Windows, macOS, or Linux

## Installation

```bash
git clone https://github.com/yuribullcatsz-coder/Explainable-AI-XAI-Demonstrator.git
cd Explainable-AI-XAI-Demonstrator
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Developer install (tests)

```bash
pip install -r requirements-dev.txt
```

## Run the app

```bash
streamlit run app.py
```

The UI opens in your browser. Recommended first run:

1. Sidebar → **Data source**: **Sample: Iris**
2. **Target column**: `species`
3. Leave **Categorical feature columns** empty (all numeric except target string — the sample marks species as strings; you can also treat none as categorical if your CSV uses numeric labels)
4. Click **Train model**
5. Open **SHAP**, **Partial dependence**, **LIME**, and **Bias check** tabs

**Note:** For Iris, `species` is string — the pipeline encodes the target via the forest’s internal handling of string `y`. Categorical **features** should be selected in the sidebar if they are non-numeric (e.g. `color`, `region`).

## Using your own CSV

1. **Upload CSV** in the sidebar.
2. Set **Target column** to the exact column name to predict.
3. Under **Categorical feature columns**, select every non-numeric column that should be one-hot encoded (excluding the target).
4. **Train model**.

**Tips**

- Rows with **missing target** values are dropped before training.
- **Classification** is inferred if the target looks categorical or has ≤10 distinct numeric values; otherwise **regression** is used.
- **SHAP** can be slow on large data; use the slider to cap the sample size.
- **LIME** needs finite numeric values after internal encoding; impute or drop rows with missing **feature** values if LIME errors.
- **Bias** tab expects a **sensitive** column with **exactly two** groups (after dropping NaNs) and a **classification** model.

## Project layout

```
Explainable-AI-XAI-Demonstrator/
  app.py                 # Streamlit UI
  xai_demo/
    pipeline.py          # Train/test split, preprocessing, Random Forest
    explain.py           # SHAP, PDP, LIME (matplotlib figures → PNG bytes)
    bias.py              # Group fairness metrics
  tests/                 # Pytest suite
  requirements.txt
  requirements-dev.txt
  .github/workflows/ci.yml
```

## Tests

Fast tests (default in CI):

```bash
pytest -q -m "not slow"
```

Include optional SHAP/LIME smoke tests:

```bash
pytest -q
```

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest -q -m "not slow"` on Ubuntu and Windows for Python 3.11.

## Limitations (by design)

- One **tabular** dataset at a time; no deep learning or image/text models.
- **Fairness** metrics are educational sanity checks, not compliance or auditing tooling.
- **Multiclass SHAP** uses an aggregate (mean \|SHAP\| across classes) for a single summary plot.

## License

Use and modify as needed for teaching and demos; add a license file if you redistribute publicly.
