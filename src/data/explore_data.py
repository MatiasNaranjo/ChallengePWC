import pandas as pd

# Load the raw data from CSV files into DataFrames
df_people = pd.read_csv("../data/raw/people.csv")
df_salary = pd.read_csv("../data/raw/salary.csv")

# Combine the dataframes by merging
df_combined = pd.merge(df_people, df_salary, on="id")

# Print the dataset
print(df_combined.head())
print("Dataset lenght:", len(df_combined))
print()

# Remove rows with missing values
df_processed = df_combined.dropna()

# Print the number of NaN (missing) values for each column in the combined DataFrame
print("Nan values in the different columns")
print(df_combined.isna().sum())
print("Dataset lenght without Nan values:", len(df_processed))
print()

# Print the unique values in the "Education Level" column
print("Education Levels:", df_processed["Education Level"].unique())

# Print the unique values in the "Gender" column
print("Genders:", df_processed["Gender"].unique())

# Get the sorted list of unique job titles
print("Number of Job titles:", len(sorted((df_processed["Job Title"].unique()))))

# Print the job titles
print("Job titles:", sorted((df_processed["Job Title"].unique())))
print()

# Combine all job titles into a single string
text = " ".join(df_processed["Job Title"])

# Split the combined text into individual words
words = text.split()

from collections import Counter

# Count the frequency of each word using Counter
word_count = Counter(words)
print("Words of Job Titles:", word_count)
print()
