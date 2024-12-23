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

#fill in the missing values 
data['bmi'].fillna(data['bmi'].mean(), inplace = True)

#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

