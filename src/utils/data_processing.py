import pandas as pd


def merging_csv(dataset_paths, merge_on="id"):
    # Read the first dataset from the list of paths into a DataFrame
    df_combined = pd.read_csv(dataset_paths[0])

    # Loop through the rest of the dataset paths
    for path in dataset_paths[1:]:
        # Read the next dataset to merge
        df_to_merge = pd.read_csv(path)

        # Merge the current dataset with the previously merged DataFrame on the specified column
        df_combined = pd.merge(df_combined, df_to_merge, on=merge_on)

    # Return the combined DataFrame after all merges
    return df_combined


def load_preprocess_train_data(
    file_path, target_column, drop_columns=None, test_size=0.2, random_state=42
):
    """
    Load and preprocess the data for training and testing.

    Parameters:
    - file_path (str): Path to the CSV file.
    - target_column (str): Name of the target column.
    - drop_columns (list): List of column names to drop (besides the target).
    - test_size (float): Proportion of the data to use for testing.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train, X_test: Feature sets for training and testing.
    - y_train, y_test: Target values for training and testing.
    """
    df = pd.read_csv(file_path)

    if drop_columns:
        df = df.drop(columns=drop_columns)

    # Split features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    from sklearn.preprocessing import StandardScaler

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
