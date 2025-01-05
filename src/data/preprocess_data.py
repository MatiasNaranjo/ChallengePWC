import pandas as pd

from src.utils.data_processing import merging_csv


def preprocess_multiple_datasets(
    dataset_paths, output_csv, merge_on="id", word_min_length=2, word_min_count=15
):
    # Load the raw data from CSV files into DataFrames
    df_combined = merging_csv(dataset_paths, merge_on="id")

    # Remove rows with missing values
    df_processed = df_combined.dropna().copy()

    # Define the mapping for education levels
    education_mapping = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
    # Map the education levels in the "Education Level" column to numeric values
    df_processed["Education Level Numeric"] = df_processed["Education Level"].map(
        education_mapping
    )

    # Define the mapping for gender
    gender_mapping = {"Male": 0, "Female": 1}
    # Map the gender values in the "Gender" column to numeric values
    df_processed["Gender Numeric"] = df_processed["Gender"].map(gender_mapping)

    # Combine all job titles into a single string for analysis
    text = " ".join(df_processed["Job Title"])

    # Split the combined string into individual words
    words = text.split()

    from collections import Counter

    # Count the frequency of each word in the "Job Title" column
    word_count = Counter(words)

    # Filter the word counts to include only words with more than word_min_length letters
    # or uppercase words, and words that appear at least word_min_count
    filtered_dict = {
        word: count
        for word, count in word_count.items()
        if (len(word) > word_min_length or word.isupper()) and count >= word_min_count
    }

    # Convert the filtered words from the dictionary into a list
    filtered_words = list(filtered_dict.keys())

    # Add a new column for each filtered word
    # Each column will indicate whether the word appears in the "Job Title" column (1 if present, 0 otherwise)
    for word in filtered_words:
        df_processed[word] = (
            df_processed["Job Title"]
            .str.contains(
                rf"\b{word}\b", case=False, regex=True
            )  # Use regex to match whole words (case-insensitive)
            .astype(int)  # Convert the boolean result to an integer (1 or 0)
        )

    # Keep only the numerical columns in the DataFrame
    df_processed = df_processed.select_dtypes(include="number")

    # Save the preprocessed DataFrame to a new CSV file
    df_processed.to_csv(output_csv, index=False)

    return df_processed
