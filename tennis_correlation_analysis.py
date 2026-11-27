import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Step 1: Load Dataset into a Pandas DataFrame using whitespace delimiter
df = pd.read_csv("/content/tennis.csv", sep=r"\s+")
print("=== Original Dataset Info ===")
print(df.head(), "\n")

# Step 2: Identify and Count Missing Values
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print("--- Missing Values Count per Column ---")
print(missing_values)
print(f"Total Missing Values in Dataset: {total_missing}\n")

# Step 3: Data Cleaning (Removing Noise)
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Dropped {initial_rows - len(df)} duplicate row(s).")

# Step 4: Convert Categorical Columns to Numeric for Correlation Analysis
df_encoded = df.copy()
for col in df_encoded.columns:
    df_encoded[col] = df_encoded[col].astype("category").cat.codes

# Step 5: Correlation Analysis
corr_matrix = df_encoded.corr()
print("--- Correlation Matrix ---")
print(corr_matrix)

# Extract unique pairs to find most positive and most negative correlations
stacked_corr = corr_matrix.stack().reset_index()
stacked_corr.columns = ["Attr1", "Attr2", "Correlation"]
stacked_corr = stacked_corr[stacked_corr["Attr1"] != stacked_corr["Attr2"]]
stacked_corr["pair"] = stacked_corr.apply(
    lambda row: tuple(sorted([row["Attr1"], row["Attr2"]])), axis=1
)
unique_corr = stacked_corr.drop_duplicates(subset=["pair"]).drop(
    columns=["pair"]
)

if not unique_corr.empty:
    most_positive = unique_corr.loc[unique_corr["Correlation"].idxmax()]
    most_negative = unique_corr.loc[unique_corr["Correlation"].idxmin()]
    print("\n--- Correlation Insights ---")
    print(
        f"Most Positively Correlated: '{most_positive['Attr1']}' and"
        f" '{most_positive['Attr2']}' (r = {most_positive['Correlation']:.4f})"
    )
    print(
        f"Most Negatively Correlated: '{most_negative['Attr1']}' and"
        f" '{most_negative['Attr2']}' (r = {most_negative['Correlation']:.4f})"
    )
