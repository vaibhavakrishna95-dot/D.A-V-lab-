import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Load Iris dataset
iris = load_iris()
# Create a pandas DataFrame for easier data manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: iris.target_names[0],
 1: iris.target_names[1],
 2: iris.target_names[2]})
feature_cols = iris.feature_names
print("--- DATASET PREVIEW ---")
print(df.head(), "\n")
stats_summary = pd.DataFrame(index=feature_cols)
stats_summary['Mean'] = df[feature_cols].mean()
stats_summary['Median'] = df[feature_cols].median()
stats_summary['Mode'] = [df[col].mode()[0] for col in feature_cols]
stats_summary['Std Dev'] = df[feature_cols].std()
stats_summary['Std Error (SEM)'] = df[feature_cols].sem()
# 95% Confidence Interval calculation
confidence = 0.95
n = len(df)
ci_lower = []
ci_upper = []
for col in feature_cols:
 mean = stats_summary.loc[col, 'Mean']
 sem = stats_summary.loc[col, 'Std Error (SEM)']
 h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
 ci_lower.append(mean - h)
 ci_upper.append(mean + h)
stats_summary['95% CI Lower'] = ci_lower
stats_summary['95% CI Upper'] = ci_upper
print("--- PART (i): DESCRIPTIVE STATISTICS ---")
print(stats_summary, "\n")
corr_matrix = df[feature_cols].corr()
print("--- PART (ii): CORRELATION MATRIX ---")
print(corr_matrix, "\n")
# Plotting the Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Iris Features')
plt.show()
# Using pandas covariance method
cov_matrix = df[['sepal length (cm)', 'petal length (cm)']].cov()
covariance_value = cov_matrix.loc['sepal length (cm)', 'petal length (cm)']
print("--- PART (iii): COVARIANCE ---")
print(f"Covariance between Sepal Length and Petal Length: {covariance_value:.4f}\n")
# Since Iris has continuous numerical features and one class feature ('species'),
# we can bin one continuous feature (e.g., 'sepal length (cm)') into categories
# to build a contingency table with the 'species' class feature.
df['sepal_length_binned'] = pd.cut(df['sepal length (cm)'],
 bins=3,
 labels=['Short', 'Medium', 'Long'])
# Building the contingency table (Cross-tabulation)
contingency_table = pd.crosstab(df['species'], df['sepal_length_binned'])
print("--- PART (iv): CONTINGENCY TABLE (Species vs Binned Sepal Length) ---")
print(contingency_table)
