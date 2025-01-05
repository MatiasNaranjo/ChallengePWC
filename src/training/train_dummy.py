# Define the DummyRegressor (baseline model)
from sklearn.dummy import DummyRegressor


def train_dummy_model(X_train, y_train, strategy="mean"):
    """
    Train a DummyRegressor model with the specified strategy and evaluate its performance.

    Parameters:
    - X_train: ndarray, training features.
    - X_test: ndarray, testing features.
    - y_train: ndarray, training target values.
    - y_test: ndarray, testing target values.
    - strategy: str, the strategy to use for the dummy regressor (default is "mean").

    Returns:
    - dict: A dictionary containing the MSE and MAE for both train and test sets.
    """
    dummy_model = DummyRegressor(strategy=strategy)

    # Train the dummy model
    dummy_model.fit(X_train, y_train)

    return dummy_model
