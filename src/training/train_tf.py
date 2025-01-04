import numpy as np
import pandas as pd
import tensorflow as tf

from utils.data_processing import load_preprocess_train_data

# Define parameters
file_path = "../data/processed/processed.csv"
target_column = "Salary"
drop_columns = ["id"]

# Call the function
X_train, X_test, y_train, y_test = load_preprocess_train_data(
    file_path=file_path, target_column=target_column, drop_columns=drop_columns
)

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
    patience=10,  # Number of epochs to wait for improvement
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
model.save("../models/final/tf_model.keras")
