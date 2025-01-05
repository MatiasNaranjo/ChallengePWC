from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.data_processing import merging_csv


def analize_data(dataset_paths, top_n=25, line_value=15):

    # Combine the dataframes by merging
    df_combined = df_combined = merging_csv(dataset_paths, merge_on="id")

    # Print the dataset
    print(df_combined.head())
    print("Dataset length:", len(df_combined))
    print()

    # Remove rows with missing values
    df_processed = df_combined.dropna()

    # Print the number of NaN (missing) values for each column in the combined DataFrame
    print("Nan values in the different columns:")
    print(df_combined.isna().sum())
    print("Dataset length without NaN values:", len(df_processed))
    print()

    # Print the unique values in the "Education Level" column
    print("Education Levels:", df_processed["Education Level"].unique())

    # Print the unique values in the "Gender" column
    print("Genders:", df_processed["Gender"].unique())

    # Get the sorted list of unique job titles
    print("Number of Job titles:", len(sorted(df_processed["Job Title"].unique())))

    # Print the job titles
    print("Job titles:", sorted(df_processed["Job Title"].unique()))
    print()

    # Combine all job titles into a single string
    text = " ".join(df_processed["Job Title"])

    # Split the combined text into individual words
    allwords = text.split()

    # Count the frequency of each word using Counter
    word_count = Counter(allwords)

    # Convert the Counter to a dictionary to preserve the insertion order
    word_dict = dict(word_count)

    # Sort the dictionary by values in descending order
    sorted_word_dict = dict(
        sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
    )

    # Extract words and counts from the sorted dictionary
    list_words = list(sorted_word_dict.keys())[:top_n]
    list_counts = list(sorted_word_dict.values())[:top_n]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(list_words, list_counts, color="skyblue")

    # Add a horizontal line at a specified y-value
    plt.axhline(y=line_value, color="red", linestyle="--", label=f"y = {line_value}")

    plt.xlabel("Words in Job Titles")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
