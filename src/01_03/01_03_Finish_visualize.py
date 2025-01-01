import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Check the Missing Value Count
print("How many values are missing?")
print(data.isnull().sum())

# Handling Missing Values

# Option 3: Fill missing values with mean (SimpleImputer)
data_option3 = data.copy()
imputer = SimpleImputer(strategy="mean")
data_option3["bmi"] = imputer.fit_transform(data_option3[["bmi"]])
print("\nOption 3: Fill missing values with mean (SimpleImputer)")
print(data_option3.isnull().sum())

# Visualization

# Create subplots to analyze distributions
fig, ax = plt.subplots(4, 2, figsize=(12, 24))

# Distribution of numerical columns
sns.histplot(data['charges'], kde=True, ax=ax[0, 0], color="blue")
ax[0, 0].set_title("Distribution of Charges (Right Skewed)")

sns.histplot(data['age'], kde=True, ax=ax[0, 1], color="green")
ax[0, 1].set_title("Distribution of Age")

sns.histplot(data['bmi'], kde=True, ax=ax[1, 0], color="purple")
ax[1, 0].set_title("Distribution of BMI")

sns.histplot(data['children'], kde=False, ax=ax[1, 1], color="orange")
ax[1, 1].set_title("Number of Children")

# Count plots for categorical columns
sns.countplot(x=data['sex'], ax=ax[2, 0], hue=None)
ax[2, 0].set_title("Count of Male vs Female")

sns.countplot(x=data['smoker'], ax=ax[2, 1], hue=None)
ax[2, 1].set_title("Count of Smokers vs Non-Smokers")

sns.countplot(x=data['region'], ax=ax[3, 0], hue=None)
ax[3, 0].set_title("Count of Regions")

# Add spacing between plots
plt.tight_layout()
plt.savefig("output/01_03_distribution_analysis.png")
plt.close()

# Visualizing Skewness with Pairplots
sns.pairplot(data, diag_kind="kde", plot_kws={'alpha': 0.6})
plt.savefig("output/01_03_pairplot.png")
plt.close()

# Scatterplot: Smokers vs Non-Smokers on Age vs Charges
sns.lmplot(x="age", y="charges", hue="smoker", data=data, palette="muted", height=7, aspect = 2)
plt.title("Age vs Charges (Smokers vs Non-Smokers)")
plt.tight_layout()
plt.savefig("output/01_03_age_vs_charges.png")
plt.close()

# Correlation Heatmap
# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])
corr = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="Wistia", annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/01_03_correlation_heatmap.png")
plt.close()