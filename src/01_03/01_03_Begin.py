# Load necessary libraries
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

# Handling Missing Values

# TODO: Check how many values are missing (NaN)

# Option 1: Drop the entire column with missing values
# TODO: Add code to drop the 'bmi' column and verify

# Option 2: Drop rows with missing values
# TODO: Add code to drop rows with missing values and verify

# Option 3: Fill missing values with mean (SimpleImputer)
# TODO: Add code to fill missing values in the 'bmi' column using SimpleImputer

# Visualization

# TODO: Create a scatterplot (Age vs. Charges)

# TODO: Create a correlation heatmap
