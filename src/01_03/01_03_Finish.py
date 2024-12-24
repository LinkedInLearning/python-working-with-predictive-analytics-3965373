import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values

# Option 1: Drop the entire column with missing values
data_option1 = data.copy()
data_option1.drop('bmi', axis=1, inplace=True)
print("\nOption 1: Drop the 'bmi' column")
print(data_option1.isnull().sum())

# Option 2: Drop rows with missing values
data_option2 = data.copy()
data_option2.dropna(inplace=True)
data_option2.reset_index(drop=True, inplace=True)
print("\nOption 2: Drop rows with missing values")
print(data_option2.isnull().sum())

# Option 3: Fill missing values with mean (SimpleImputer)
data_option3 = data.copy()
imputer = SimpleImputer(strategy="mean")
data_option3["bmi"] = imputer.fit_transform(data_option3[["bmi"]])
print("\nOption 3: Fill missing values with mean (SimpleImputer)")
print(data_option3.isnull().sum())