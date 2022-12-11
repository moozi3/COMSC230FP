# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:31:19 2022

@author: moozi
Name: Lauren Moulaison, Wyatt VanDyk, Dan Le
Course: COMSC230
Prof. Name: Prof. Rivera Morales
Assignment : Final Project
Program Name: RaisinsLWD
Program brief description: Run an analysis of a raisin dataset
"""
#imports
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#import dataset
raisins = pd.read_csv('C:/Users/moozi/Downloads/raisins.csv')

#just testing we see what we want to see
print(raisins.columns)

#Okay, this dataset includes raisins of two different kinds, and records 
#the size of each raisin, among other factors. Let's see if we can determine
#some factors that differentiate each raisin. 

#Fortunately, this dataset is already clean. All of the entries are similarly
#formatted which means we can just jump into the analysis. For the sake of
#determining correlation, I will change the species column which returns
#Kecimen or Besni, into 0 and 1 respectively.
raisins['Class'].replace(['Kecimen', 'Besni'],[0, 1], inplace=True)
#check the edit went through:
print(raisins)


#This line allows us to quickly see what variables (if any) are correlated to 
#one another which will help us choose how to go about our linear regression
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (raisins.corr())

#It's obvious the raisins' pixel area is correlated to its length and width
#but what about these measurements compared to the species of raisin?
#It appears that MajorAxisLength and Perimeter have a significant correlation
#to the species of raisin, producing results of .673 and .666. Let's explore
#these with lin reg
x = raisins['Class'].values 
y = raisins['MajorAxisLength'].values
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7)

lr = LinearRegression()
lr_model = lr.fit(x_train, y_train)
coefficients = lr_model.coef_
intercept = lr_model.intercept_

y_pred = lr_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) 

x = raisins['Class']

y = raisins['MajorAxisLength']
plt.scatter(x, y)

plt.show()

plt.plot(x_test, y_pred, color="blue", linewidth=3)

#Now to compare Class with Perimeter
x1 = raisins['Class'].values 
y1 = raisins['Perimeter'].values
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, train_size=.7)

lr1 = LinearRegression()
lr_model1 = lr.fit(x_train, y_train)
coefficients1 = lr_model.coef_
intercept1 = lr_model.intercept_

y_pred1 = lr_model.predict(x_test)
mse1 = mean_squared_error(y_test, y_pred)
r21 = r2_score(y_test, y_pred) 

x1 = raisins['Class']

y1 = raisins['Perimeter']
plt.scatter(x1, y1)

plt.plot(x1_test, y_pred1, color="red", linewidth=3)

plt.show()