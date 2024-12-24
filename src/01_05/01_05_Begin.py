import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values
# Fill missing values in 'bmi' with mean (Option 3 from previous module)
imputer = SimpleImputer(strategy="mean")
data["bmi"] = imputer.fit_transform(data[["bmi"]])
print("\nOption 3: Fill missing values with mean (SimpleImputer)")
print(data.isnull().sum())

# Encoding Categorical Variables
# Label encode 'sex' and 'smoker'
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])

# One hot encode 'region'
ohe = OneHotEncoder(sparse_output=False, drop='first')
region_encoded = ohe.fit_transform(data[['region']])
region_columns = ohe.get_feature_names_out(['region'])
region_df = pd.DataFrame(region_encoded, columns=region_columns)

# TODO: Combine numerical columns (age, bmi, children) with encoded columns (region, sex, smoker)

# TODO: Assign response variable ('charges') to y_final

# TODO: Split the data into train and test sets using train_test_split
