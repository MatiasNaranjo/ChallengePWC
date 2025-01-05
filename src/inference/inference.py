import pandas as pd

from src.models.models import evaluate_model_performance


def results_multiple_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluates multiple models and returns a DataFrame with the evaluation metrics.

    Parameters:
    - models: List of tuples, where each tuple contains a model and its name (e.g., [(model1, "Model 1"), ...]).
    - X_train: Features for the training dataset.
    - X_test: Features for the testing dataset.
    - y_train: Target values for the training dataset.
    - y_test: Target values for the testing dataset.

    Returns:
    - DataFrame: DataFrame containing evaluation metrics for each model.
    """
    # List to store evaluation results
    results = []

    # Loop through each model and evaluate
    for model, model_name in models:
        model_result = evaluate_model_performance(
            model, X_train, X_test, y_train, y_test, model_name=model_name
        )
        results.append(model_result)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df
