# Explainable AI (XAI) Demonstrator

This Streamlit application provides an interactive interface to explore and visualize key Explainable AI (XAI) techniques. It allows users to upload a dataset, train a model, and then visualize model explanations using SHAP values, Partial Dependence Plots (PDP), and LIME-based counterfactuals. It also includes a basic check for model bias with respect to a sensitive attribute.

## Features

- **SHAP Values**: Visualize the impact of each feature on individual predictions and overall model behavior.
- **Partial Dependence Plots (PDP)**: Understand how a single feature influences the model's average prediction.
- **Counterfactual Explanations**: Explore "what-if" scenarios using LIME to see how changes in input features could affect the prediction.
- **Bias Check**: Analyze the model's predictions for potential bias with respect to a selected sensitive attribute using the AIF360 library.

## Prerequisites

- Python 3.8 or higher

## Installation

1.  Clone this repository or download the `app.py` and `README.md` files.
2.  It is recommended to create a new virtual environment:
    ```bash
    python -m venv xai_env
    source xai_env/bin/activate # On Windows: xai_env\Scripts\activate
    ```
3.  Install the required Python packages:
    ```bash
    pip install streamlit scikit-learn shap pdpbox lime aif360
    ```
    *Note: If you are on Windows and encounter issues with `aif360`, you might need to install its dependencies separately or use a pre-compiled wheel.*

## How to Run

1.  Navigate to the directory containing the `app.py` file.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your default web browser.

## How to Use

1.  **Upload Data**: Use the sidebar to upload a CSV file containing your dataset.
2.  **Configure**:
    - Enter the name of the column you wish to predict (the target).
    - Select any categorical columns that need to be encoded.
3.  **Train Model**: The app will automatically train a simple Random Forest model on your data.
4.  **Explore Explanations**:
    - **SHAP**: The summary plot will show feature importance based on SHAP values.
    - **PDP**: Select a feature from the dropdown to generate and view its partial dependence plot.
    - **Counterfactuals**: Select a specific data instance from the test set to see a LIME explanation for its prediction.
    - **Bias**: Select a sensitive attribute (e.g., gender, race) from the categorical columns to check for statistical parity and disparate impact.
