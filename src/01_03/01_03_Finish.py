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
count_nan = data.isnull().sum()  # The number of missing values for each column
print("\nMissing values before imputation:")
print(count_nan[count_nan > 0])

# Fill missing values
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Check for missing values after imputation
count_nan = data.isnull().sum()  # The number of missing values for each column
print("\nMissing values after imputation:")
print(count_nan[count_nan > 0])
