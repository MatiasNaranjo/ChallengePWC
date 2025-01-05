from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model_performance(
    model, X_train, X_test, y_train, y_test, model_name="model"
):
    """
    Evaluates a given model's performance on training and testing datasets and returns metrics.

    Parameters:
    - model: Trained model to evaluate.
    - X_train: Features for the training dataset.
    - X_test: Features for the testing dataset.
    - y_train: Target values for the training dataset.
    - y_test: Target values for the testing dataset.
    - model_name (str): Name of the model (default is "model").

    Returns:
    - dict: Dictionary containing model name, train MSE, test MSE, train MAE, and test MAE.
    """
    # Predict on training and testing data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Return the results as a dictionary
    return {
        "model": model_name,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_mae": train_mae,
        "test_mae": test_mae,
    }
