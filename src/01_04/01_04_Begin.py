#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:46:31 2019

@author: berkunis
"""
##############################################01_02_PythonLibraries#####################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder





#import data
data = pd.read_csv("input/insurance.csv")

#see the first 15 lines of data
print(data.head(15))

############################################01_03_HandlingMissingValues###################################################

#check how many values are missing (NaN) before we apply the methods below 
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#fill in the missing values (we will look at 4 options for this course - there are so many other methods out there.)

#option0 for dropping the entire column
data = pd.read_csv("input/insurance.csv") # reloading fresh dataset for option 0
data.drop('bmi', axis = 1, inplace = True)
#check how many values are missing (NaN) - after we dropped 'bmi'
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#option1 for dropping NAN
data = pd.read_csv("input/insurance.csv") # reloading fresh dataset for option 1
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#option2 for filling NaN # reloading fresh dataset for option 2
data = pd.read_csv("input/insurance.csv")
imputer = SimpleImputer(strategy='mean')
imputer.fit(data['bmi'].values.reshape(-1, 1))
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#option3 for filling NaN # reloading fresh dataset for option 3
data = pd.read_csv("input/insurance.csv")
data['bmi'].fillna(data['bmi'].mean(), inplace = True)
print(data.head(15))
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])


############################################01_04_ConvertCategoricalDataintoNumbers##############################################


# sklearn label encoding: maps each category to a different integer

#create ndarray for label encodoing (sklearn)
sex = data.iloc[:,1:2].values
smoker = data.iloc[:,4:5].values


#label encoder = le

## le for sex
le = LabelEncoder()
sex[:,0] = le.fit_transform(sex[:,0])
sex = pd.DataFrame(sex)
sex.columns = ['sex']
le_sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for sex:") 
print(le_sex_mapping)
print(sex[:10])


## le for smoker


#sklearn one hot encoding: maps each category to 0 (cold) or 1 (hot) 

#one hot encoder = ohe

#create ndarray for one hot encodoing (sklearn)


## ohe for region






