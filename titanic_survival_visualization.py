import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
# Set high-performance display theme
sns.set_theme(style="whitegrid")
# 1. Efficient Data Loading & Cleaning
print("Loading Titanic dataset...")
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame
# Vectorized type casting and cleaning
df["survived"] = df["survived"].astype(int)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["fare"] = pd.to_numeric(df["fare"], errors="coerce")
# Drop unnecessary rows for numerical plots once to save repeated computation
df_clean = df.dropna(subset=["age", "fare", "sex", "survived"]).copy()
df_clean["survived_label"] = df_clean["survived"].map(
 {0: "Non-Survivor", 1: "Survivor"}
)
# Set up a multi-panel figure for clear comparative layout
fig = plt.figure(figsize=(16, 12))
# 1. Bar Chart: Survival Frequency by Gender
ax1 = plt.subplot(2, 2, 1)
sns.countplot(
 data=df_clean,
 x="sex",
 hue="survived_label",
 palette=["salmon", "skyblue"],
 ax=ax1,
)
ax1.set_title(
 "Survival Frequency by Gender", fontsize=12, fontweight="bold"
)
ax1.set_xlabel("Gender", fontsize=11)
ax1.set_ylabel("Passenger Count", fontsize=11)
ax1.legend(title="Outcome")
# 2. Scatter Plot: Age vs. Fare Colored by Survival
ax2 = plt.subplot(2, 2, 2)
sns.scatterplot(
 data=df_clean,
 x="age",
 y="fare",
 hue="survived",
 palette="Set1",
 alpha=0.6,
 ax=ax2,
)
ax2.set_title("Age vs. Passenger Fare", fontsize=12, fontweight="bold")
ax2.set_xlabel("Age (Years)", fontsize=11)
ax2.set_ylabel("Fare ($)", fontsize=11)
ax2.legend(title="Survived", labels=["No (0)", "Yes (1)"])
# 3. Density Distribution: Age & Fare Side-by-Side
ax3 = plt.subplot(2, 2, 3)
sns.kdeplot(
 data=df_clean["age"], color="blue", fill=True, alpha=0.3, ax=ax3
)
ax3.set_title("Density Distribution: Age", fontsize=12, fontweight="bold")
ax3.set_xlabel("Age", fontsize=11)
ax3.set_ylabel("Density", fontsize=11)
ax4 = plt.subplot(2, 2, 4)
sns.kdeplot(
 data=df_clean["fare"],
 color="green",
 fill=True,
 alpha=0.3,
 log_scale=True,
 ax=ax4,
)
ax4.set_title(
 "Density Distribution: Fare (Log Scale)", fontsize=12, fontweight="bold"
)
ax4.set_xlabel("Fare (Log Scale)", fontsize=11)
ax4.set_ylabel("Density", fontsize=11)
plt.tight_layout()
plt.show()
# 4. Pair Plot: Pairwise Bivariate Distribution
# Optimized subset selection to minimize rendering lag
pair_subset = df_clean[["age", "fare", "pclass", "survived_label"]]
pairplot_fig = sns.pairplot(
 pair_subset,
 hue="survived_label",
 palette={"Non-Survivor": "salmon", "Survivor": "skyblue"},
 diag_kind="kde",
 corner=False,
 height=2.5,
)
pairplot_fig.fig.suptitle(
 "Pairwise Bivariate Distribution of Titanic Features", y=1.02, fontsize=14,
fontweight="bold"
)
plt.show()
