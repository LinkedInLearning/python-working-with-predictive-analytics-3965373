# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values
# Check for missing values
print("\nMissing values before imputation:")
print(data.isnull().sum())

# Fill missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
data["bmi"] = imputer.fit_transform(data[["bmi"]])

# Check for missing values after imputation
print("\nMissing values after imputation:")
print(data.isnull().sum())

# Visualization: Age vs Charges
plt.figure(figsize=(12, 8))
sns.lmplot(
    x="age", y="charges", hue="smoker", data=data, palette="coolwarm",
    height=6, aspect=2, facet_kws={"legend_out": False}  # Avoid deprecated warning
)
plt.title("Age vs Charges (Smokers vs Non-Smokers)", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Charges", fontsize=12)

# Adjust legend position
plt.legend(title="Smoker", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12)

plt.tight_layout()
plt.savefig("output/01_03_age_vs_charges.png")
plt.close()

# Visualization: Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=[np.number])
corr = numeric_data.corr()
sns.heatmap(corr, cmap="Wistia", annot=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("output/01_03_correlation_heatmap.png")
plt.close()
