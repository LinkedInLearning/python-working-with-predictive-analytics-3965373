import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values
# Option 3: Fill missing values with mean (SimpleImputer)
# Reload dataset to ensure fresh state
imputer = SimpleImputer(strategy="mean")
data["bmi"] = imputer.fit_transform(data[["bmi"]])
print("\nOption 3: Fill missing values with mean (SimpleImputer)")
print(data.isnull().sum())

# Label Encoding: Encode 'sex' and 'smoker' columns
# TODO: Create a label encoder instance and encode the 'sex' column

# TODO: Create a label encoder instance and encode the 'smoker' column

# One Hot Encoding: Encode the 'region' column
# TODO: Create a one hot encoder instance and encode the 'region' column

# TODO: Convert the result into a DataFrame with appropriate column names
