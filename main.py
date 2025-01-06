import os

print(os.getcwd())

# Change the working directory to the specified path
os.chdir(r"d:\matna\Documents\Escritorio\ChallengePWC")

from src.data.explore_data import analize_data
from src.data.preprocess_data import preprocess_multiple_datasets
from src.inference.explain_features_shap import analyze_feature_importance_shap
from src.inference.inference import results_multiple_models
from src.training.train_dummy import train_dummy_model
from src.training.train_tf import train_model_tf
from src.utils.data_processing import load_preprocess_train_data

# Preprocess data
# Define the paths to the raw datasets (people and salary data)
dataset_paths = ["data/raw/people.csv", "data/raw/salary.csv"]
# Define the output path for the processed dataset
processed_csv = "data/processed/processed.csv"

# Preprocess the raw datasets:
# - Merge them on the "id" column
# - Perform text analysis on job titles (filtering words by length and frequency)
df_processed = preprocess_multiple_datasets(
    dataset_paths, processed_csv, merge_on="id", word_min_length=2, word_min_count=15
)

# Split data
# Specify the target column for predictions ("Salary") and columns to drop (like "id")
target_column = "Salary"
drop_columns = ["id"]

# Load and preprocess the data:
# - Split it into training and testing sets
# - Normalize features
X_train, X_test, y_train, y_test = load_preprocess_train_data(
    processed_csv,
    target_column,
    drop_columns=drop_columns,
    test_size=0.2,  # 20% of the data will be used for testing
    random_state=42,  # Ensure reproducibility
)

# Train models
# Train a Neural Network model using TensorFlow
# - The model is saved to the specified path after training
# - Early stopping is used to avoid overfitting
model_save_path = "models/final/model_tf.keras"
model_tf = train_model_tf(
    X_train,
    X_test,
    y_train,
    y_test,
    model_save_path,
    epochs=300,  # Maximum number of epochs
    batch_size=32,  # Number of samples per gradient update
    patience=10,  # Stop training if no improvement after 10 epochs
)

# Train a Dummy model
# - This serves as a baseline by predicting the mean of the target variable
model_dummy = train_dummy_model(X_train, y_train, strategy="mean")

models = [(model_dummy, "dummy"), (model_tf, "tf")]

# Results
df_result = results_multiple_models(models, X_train, X_test, y_train, y_test)

# Analize features importance
shap_values = analyze_feature_importance_shap(
    model_tf, X_test, df_processed, target_column="Salary", non_feature_column="id"
)
