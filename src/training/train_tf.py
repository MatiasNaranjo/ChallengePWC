import numpy as np
import pandas as pd
import tensorflow as tf

df_model = pd.read_csv("../data/processed/processed.csv")

# Split features (X) and target (y)
X = df_model.drop(columns=["Salary", "id"])  # Select all columns except Salary and id
y = df_model["Salary"].values

# Normalize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
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
