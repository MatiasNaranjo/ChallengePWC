import pandas as pd
import shap


def analyze_feature_importance_shap(
    model, X_test, df_processed, target_column="Salary", non_feature_column="id"
):
    """
    Analyzes feature importance using SHAP and visualizes the results.

    Parameters:
    - model: Trained TensorFlow model.
    - X_test: ndarray, feature matrix used for testing.
    - df_processed: DataFrame, preprocessed dataset containing feature names.
    - target_column: str, name of the target column to exclude (default: "Salary").
    - non_feature_column: str, name of the columns to exclude (default: "id").

    Returns:
    - shap_values: SHAP values calculated for the test set.
    """
    # Extract feature names from the processed DataFrame
    feature_names = df_processed.drop(
        columns=[target_column, non_feature_column]
    ).columns.tolist()

    # Create a DataFrame for X_test with proper column names
    df_X_test = pd.DataFrame(X_test, columns=feature_names)

    # Initialize the SHAP explainer
    explainer = shap.Explainer(model.predict, X_test)

    # Calculate SHAP values
    shap_values = explainer(df_X_test)

    # Plot SHAP visualizations
    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)

    return shap_values
