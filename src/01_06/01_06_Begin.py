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
# Option 3: Fill missing values with mean (SimpleImputer)
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

# Combine numerical and encoded columns
X_num = data[['age', 'bmi', 'children']].copy()
X_final = pd.concat([X_num, region_df, data['sex'], data['smoker']], axis=1)

# Assign response variable
y_final = data['charges']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=0)

# TODO: Normalize the training and test sets using MinMaxScaler

# TODO: Standardize the training and test sets using StandardScaler
