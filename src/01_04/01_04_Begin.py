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

# Check how many values are missing (NaN) before we apply the methods below
count_nan = data.isnull().sum()  # The number of missing values for every column
print("\nMissing values before handling:")
print(count_nan[count_nan > 0])

# Fill in the missing values (we will look at 4 options for this course)

# Option 0: Drop the entire column
data_option0 = data.copy()
data_option0.drop('bmi', axis=1, inplace=True)
count_nan = data_option0.isnull().sum()  # Check missing values after dropping 'bmi'
print("\nOption 0: Drop the 'bmi' column")
print(count_nan[count_nan > 0])

# Option 1: Drop rows with missing values
data_option1 = data.copy()
data_option1.dropna(inplace=True)
data_option1.reset_index(drop=True, inplace=True)
count_nan = data_option1.isnull().sum()  # Check missing values after dropping rows
print("\nOption 1: Drop rows with missing values")
print(count_nan[count_nan > 0])

# Option 2: Fill missing values with mean (using SimpleImputer)
data_option2 = data.copy()
imputer = SimpleImputer(strategy='mean')
data_option2['bmi'] = imputer.fit_transform(data_option2[['bmi']])
count_nan = data_option2.isnull().sum()  # Check missing values after filling with mean
print("\nOption 2: Fill missing values with mean (SimpleImputer)")
print(count_nan[count_nan > 0])

# Option 3: Fill missing values with mean (using pandas)
data_option3 = data.copy()
data_option3['bmi'] = data_option3['bmi'].fillna(data_option3['bmi'].mean())
count_nan = data_option3.isnull().sum()  # Check missing values after filling with pandas
print("\nOption 3: Fill missing values with mean (Pandas)")
print(count_nan[count_nan > 0])

# Convert Categorical Data into Numbers

# Sklearn Label Encoding: Maps each category to a different integer

# Create ndarray for label encoding
sex = data.iloc[:, 1:2].values  # Select the 'sex' column
smoker = data.iloc[:, 4:5].values  # Select the 'smoker' column

# Label Encoder for 'sex'
le = LabelEncoder()
sex[:, 0] = le.fit_transform(sex[:, 0])
sex = pd.DataFrame(sex, columns=['sex'])
le_sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nSklearn Label Encoding Results for 'sex':")
print(le_sex_mapping)
print(sex.head(10))

# Label Encoder for 'smoker'
# Add code here to encode 'smoker' using LabelEncoder

# Sklearn One-Hot Encoding: Maps each category to binary vectors
# Create ndarray for one-hot encoding
# Add code here to encode 'region' using OneHotEncoder
