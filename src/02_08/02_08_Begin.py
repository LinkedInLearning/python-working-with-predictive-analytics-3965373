import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Fill missing values in the 'bmi' column with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Encode categorical variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])       # e.g., Female=0, Male=1
data['smoker'] = le.fit_transform(data['smoker']) # e.g., No=0, Yes=1
region_df = pd.get_dummies(data['region'], drop_first=True)  # One-hot encode 'region'

# Prepare the data
X_num = data[['age', 'bmi', 'children']]  # Numerical features
X_final = pd.concat([X_num, region_df, data['sex'], data['smoker']], axis=1)
y_final = data['charges']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.33, random_state=0
)

# Standardize the features
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

# Instantiate RandomForestRegressor
forest = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',
    random_state=1,
    n_jobs=-1
)

# TODO: Fit the model on the training data
# e.g.:
# forest.fit(X_train, y_train)

# TODO: Predict on both training and test datasets
# y_train_pred = forest.predict(X_train)
# y_test_pred = forest.predict(X_test)

# TODO: Print the final scores (e.g., R-squared for train/test)
print(
    "forest train score %.3f, forest test score: %.3f"
    % (forest.score(X_train, y_train), forest.score(X_test, y_test))
)