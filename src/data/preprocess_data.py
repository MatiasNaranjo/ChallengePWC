import pandas as pd

# Load the raw data from CSV files into DataFrames
df_people = pd.read_csv("../../data/raw/people.csv")
df_salary = pd.read_csv("../../data/raw/salary.csv")

# Combine the dataframes by merging
df_combined = pd.merge(df_people, df_salary, on="id")

# Remove rows with missing values
df_processed = df_combined.dropna()

df_processed = df_processed.copy()

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
