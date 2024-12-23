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
import seaborn as sns

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


############################################Vizualization################################################################

figure, ax = plt.subplots(4,2, figsize=(12,24))

#See the distrubution of the data
sns.distplot(data['charges'],ax= ax[0,0])
sns.distplot(data['age'],ax=ax[0,1])
sns.distplot(data['bmi'],ax= ax[1,0])
sns.distplot(data['children'],ax= ax[1,1])


sns.countplot(data['sex'],ax=ax[2,0])
sns.countplot(data['smoker'],ax= ax[2,1])
sns.countplot(data['region'],ax= ax[3,0])



#visualizeing skewness
sns.pairplot(data)

#Lets look at smokers vs non-smokers on age vs charges:

sns.lmplot(x="age", y="charges", hue="smoker", data=data, palette = 'muted', height = 7)
plt.show(sns)

#Lets look at correlation:

corr = data.corr()

sns.heatmap(corr, cmap = 'Wistia', annot= True)
plt.show(sns)
