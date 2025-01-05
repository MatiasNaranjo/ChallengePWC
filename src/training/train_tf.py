import numpy as np
import pandas as pd
import tensorflow as tf


def train_model_tf(
    X_train,
    X_test,
    y_train,
    y_test,
    model_save_path,
    epochs=300,
    batch_size=32,
    patience=10,
):
    """
    Train a TensorFlow model with provided training and testing data, and save the trained model.

    Parameters:
    - X_train: ndarray, training features.
    - X_test: ndarray, testing features.
    - y_train: ndarray, training target values.
    - y_test: ndarray, testing target values.
    - model_save_path: str, path where the trained model will be saved.
    - epochs: int, number of training epochs (default: 300).
    - batch_size: int, batch size for training (default: 32).
    - patience: int, number of epochs to wait for early stopping (default: 10).

    Returns:
    - history: History object containing the training history.
    """
    # Define parameters
    file_path = "../data/processed/processed.csv"
    target_column = "Salary"
    drop_columns = ["id"]

    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    # Define the model
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),  # Define the input
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),  # Output layer for regression
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    from tensorflow.keras.callbacks import EarlyStopping

    # Add EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor (validation loss in this case)
        patience=patience,  # Number of epochs to wait for improvement
        restore_best_weights=True,  # Restore the best model weights after stopping
        verbose=1,
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1,
    )

    # Save the model
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    return model
