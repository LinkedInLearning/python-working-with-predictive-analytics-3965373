# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values

# Option 0: Drop the entire column with missing values
print("\nOption 0: Drop the 'bmi' column")
data_option0 = data.copy()
data_option0.drop('bmi', axis=1, inplace=True)
print("Missing values after dropping the 'bmi' column:")
print(data_option0.isnull().sum())

# Option 1: Drop rows with any missing values
print("\nOption 1: Drop rows with missing values")
data_option1 = data.copy()
data_option1.dropna(inplace=True)
data_option1.reset_index(drop=True, inplace=True)
print("Missing values after dropping rows with missing values:")
print(data_option1.isnull().sum())

# Option 2: Fill missing values with mean (using SimpleImputer)
print("\nOption 2: Fill missing values with mean (SimpleImputer)")
data_option2 = data.copy()
imputer = SimpleImputer(strategy="mean")
data_option2['bmi'] = imputer.fit_transform(data_option2[['bmi']])
print("Missing values after filling with mean (SimpleImputer):")
print(data_option2.isnull().sum())

# Option 3: Fill missing values with mean (using pandas)
print("\nOption 3: Fill missing values with mean (pandas)")
data_option3 = data.copy()
data_option3['bmi'] = data_option3['bmi'].fillna(data_option3['bmi'].mean())
print("Missing values after filling with mean (pandas):")
print(data_option3.isnull().sum())

# Convert Categorical Data into Numbers

# Option 0: Pandas factorize (Label Encoding)
print("\nOption 0: Pandas factorize for label encoding")
region = data["region"]
region_encoded, region_categories = pd.factorize(region)
factor_region_mapping = dict(zip(region_categories, region_encoded))
print("Original Region Categories:", region[:10])
print("Region Categories:", region_categories)
print("Encoded Region Values:", region_encoded[:10])
print("Factor Region Mapping:", factor_region_mapping)

# Option 1: Pandas get_dummies (One-Hot Encoding)
print("\nOption 1: Pandas get_dummies for one-hot encoding")
region_encoded_df = pd.get_dummies(region, prefix='', prefix_sep='')
print("Original Region Categories:", region[:10])
print("One-Hot Encoded Values:")
print(region_encoded_df[:10])

# Option 2: Sklearn Label Encoding
print("\nOption 2: Sklearn Label Encoding for 'sex' and 'smoker'")
le = LabelEncoder()

# Encoding 'sex'
data['sex_encoded'] = le.fit_transform(data['sex'])
sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sex Encoding Mapping:", sex_mapping)
print(data[['sex', 'sex_encoded']].head())

# Encoding 'smoker'
data['smoker_encoded'] = le.fit_transform(data['smoker'])
smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Smoker Encoding Mapping:", smoker_mapping)
print(data[['smoker', 'smoker_encoded']].head())

# Option 3: Sklearn One-Hot Encoding
print("\nOption 3: Sklearn One-Hot Encoding for 'region'")
ohe = OneHotEncoder(sparse_output=False)
region_encoded = ohe.fit_transform(data[['region']])
region_encoded_df = pd.DataFrame(region_encoded, columns=ohe.get_feature_names_out(['region']))
print("One-Hot Encoded Region Values:")
print(region_encoded_df.head())
