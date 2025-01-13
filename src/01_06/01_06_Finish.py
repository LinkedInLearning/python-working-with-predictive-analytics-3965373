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
# Reload dataset to simulate a clean state after missing value handling
imputer = SimpleImputer(strategy="mean")
data["bmi"] = imputer.fit_transform(data[["bmi"]])
print("\nOption 3: Fill missing values with mean (SimpleImputer)")
print(data.isnull().sum())

# Label Encoding: Encode 'sex' and 'smoker' columns
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
print("\nSklearn Label Encoding for 'sex':")
print(dict(zip(le.classes_, le.transform(le.classes_))))
print(data[['sex']].head(10))

data['smoker'] = le.fit_transform(data['smoker'])
print("\nSklearn Label Encoding for 'smoker':")
print(dict(zip(le.classes_, le.transform(le.classes_))))
print(data[['smoker']].head(10))

# One Hot Encoding: Encode the 'region' column
ohe = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
region_encoded = ohe.fit_transform(data[['region']])
region_columns = ohe.get_feature_names_out(['region'])

# Add one hot encoded columns back to the DataFrame
region_df = pd.DataFrame(region_encoded, columns=region_columns)
data = pd.concat([data.reset_index(drop=True), region_df.reset_index(drop=True)], axis=1)
data.drop(columns=['region'], inplace=True)

print("\nSklearn One Hot Encoding for 'region':")
print(region_df.head(10))

# Display the updated DataFrame
print("\nFinal DataFrame after encoding:")
print(data.head(15))
