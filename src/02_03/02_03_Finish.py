import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Fill missing values in the 'bmi' column with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Encoding Categorical Variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])  # e.g., Female=0, Male=1
data['smoker'] = le.fit_transform(data['smoker'])  # e.g., No=0, Yes=1
region_df = pd.get_dummies(data['region'], drop_first=True)  # One-hot encode 'region'

# Prepare the data
X_num = data[['age', 'bmi', 'children']].copy()  # Numerical features
X_final = pd.concat([X_num, region_df, data['sex'], data['smoker']], axis=1)  # Combine all features
y_final = data['charges']  # Target variable

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_final)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_final, test_size=0.33, random_state=0
)

# Standardize the polynomial features
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

# Fit the polynomial regression model on the training data
poly_lr = LinearRegression()
poly_lr.fit(X_train, y_train)

# Predict on both training and test datasets
y_train_pred = poly_lr.predict(X_train)
y_test_pred = poly_lr.predict(X_test)

# Print final coefficients, intercept, and R-squared scores
print("Polynomial Regression (degree=2)")
print("Coefficients:", poly_lr.coef_)
print("Intercept:", poly_lr.intercept_)
print(
    "poly train score %.3f, poly test score: %.3f"
    % (poly_lr.score(X_train, y_train), poly_lr.score(X_test, y_test))
)