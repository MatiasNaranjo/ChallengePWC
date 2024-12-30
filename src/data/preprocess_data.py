import pandas as pd

# Load the raw data from CSV files into DataFrames
df_people = pd.read_csv("../../data/raw/people.csv")
df_salary = pd.read_csv("../../data/raw/salary.csv")

# Combine the dataframes by merging
df_combined = pd.merge(df_people, df_salary, on="id")

# Remove rows with missing values
df_processed = df_combined.dropna()
