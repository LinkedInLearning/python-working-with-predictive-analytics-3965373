import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Fill missing values in the 'bmi' column with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Encoding Categorical Variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])  # Female=0, Male=1
data['smoker'] = le.fit_transform(data['smoker'])  # No=0, Yes=1
region_df = pd.get_dummies(data['region'], drop_first=True)  # One-hot encode 'region'

# Prepare the data
X_num = data[['age', 'bmi', 'children']].copy()  # Numerical features
X_final = pd.concat([X_num, region_df, data['sex'], data['smoker']], axis=1)  # Combine all features
y_final = data['charges']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.33, random_state=0
)

# Standardize the features
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))


# TODO: Fit the Linear Regression model on the training data

# TODO: Predict on both training and test datasets

# Print the coefficients, intercept, and R-squared scores
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print(
    "lr train score %.3f, lr test score: %.3f"
    % (lr.score(X_train, y_train), lr.score(X_test, y_test))
)