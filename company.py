# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:13:59 2020

@author: chandan
"""
##importing the dataset
import pandas as pd
data = pd.read_csv("E:\\assignment\\decisiontree\\Company_Data.csv")
data.head()
data.describe()

##checking the null value inside the data and replacing them with the mean value
data.isnull().any()
data['Sales']=data['Sales'].replace(0,data['Sales'].mean())
data['Advertising']=data['Advertising'].replace(0,data['Advertising'].mean())



##finding the unique values of the sales column
data['Sales'].unique()
##converting the sales column in catagorical form and catagorising them
category= pd.cut(data.Sales,bins=[0,3,10,20],labels=['low','medium','high'])
##inserting the column inside the dataframe
data.insert(1,'sales_type',category)
data.sales_type.value_counts()

##droping the columns
data.drop(columns=['Sales','ShelveLoc','Urban','US'],inplace= True)

colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)

##importing decision tree and using entropy value to get the information
from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])


preds = model.predict(test[predictors])
type(preds)
pd.Series(preds).value_counts()

pd.crosstab(test[target],preds)

import numpy as np

np.mean(preds==test.sales_type)
