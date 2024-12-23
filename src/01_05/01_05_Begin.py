# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values

# Check how many values are missing (NaN) before applying methods
count_nan = data.isnull().sum()
print("\nMissing values before handling:")
print(count_nan[count_nan > 0])

# Fill missing values with mean (SimpleImputer)
imputer = SimpleImputer(strategy='mean')
data['bmi'] = imputer.fit_transform(data[['bmi']])
print("\nMissing values after filling with mean (SimpleImputer):")
print(data.isnull().sum())

# Convert Categorical Data into Numbers

# Pandas factorize (Label Encoding)
region = data["region"]
region_encoded, region_categories = pd.factorize(region)
print("\nPandas factorize results for 'region':")
print("Region Mapping:", dict(zip(region_categories, region_encoded)))

# Pandas get_dummies (One-Hot Encoding)
region_encoded_df = pd.get_dummies(region, prefix='', prefix_sep='')
print("\nPandas get_dummies results for 'region':")
print(region_encoded_df.head(10))

# Sklearn Label Encoding
le = LabelEncoder()
data['sex_encoded'] = le.fit_transform(data['sex'])
data['smoker_encoded'] = le.fit_transform(data['smoker'])
print("\nSklearn Label Encoding results:")
print(data[['sex', 'sex_encoded']].head())
print(data[['smoker', 'smoker_encoded']].head())

# Sklearn One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)
region_encoded = ohe.fit_transform(data[['region']])
region_encoded_df = pd.DataFrame(region_encoded, columns=ohe.get_feature_names_out(['region']))
print("\nSklearn One-Hot Encoding results for 'region':")
print(region_encoded_df.head(10))

# Dividing the Data into Train and Test Sets

# Combine numerical and encoded categorical data
# Placeholder for combining numerical and encoded data

# Define target variable (y) and features (X)
# Placeholder for defining X and y

# Split the data into training and testing sets
# Placeholder for train-test split
