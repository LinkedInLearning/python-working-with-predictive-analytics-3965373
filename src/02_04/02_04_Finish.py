import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Fill missing values in the 'bmi' column with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Encode categorical variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])      # e.g., Female=0, Male=1
data['smoker'] = le.fit_transform(data['smoker'])# e.g., No=0, Yes=1
region_df = pd.get_dummies(data['region'], drop_first=True)

# Prepare the data
X_num = data[['age', 'bmi', 'children']]
X_final = pd.concat([X_num, region_df, data['sex'], data['smoker']], axis=1)
y_final = data['charges']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.33, random_state=0
)

# Standardize the features
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

# Instantiate SVR
svr = SVR(kernel='linear', C=300)

# Fit the SVR model on the training data
svr.fit(X_train, y_train)

# Predict on both training and test datasets
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# Print final scores
print("SVR (linear kernel, C=300)")
print("Train R-squared: %.3f" % svr.score(X_train, y_train))
print("Test R-squared:  %.3f" % svr.score(X_test, y_test))