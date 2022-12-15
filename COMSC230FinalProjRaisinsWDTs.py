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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


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
#print(raisins)


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
"""
x1 = raisins['Class'].values 
y1 = raisins['Perimeter'].values
x1 = x1.reshape(1, -1)
y1 = y1.reshape(1, -1)
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, train_size=.7)

lr1 = LinearRegression()
lr_model1 = lr.fit(x1_train, y1_train)
coefficients1 = lr_model.coef_
intercept1 = lr_model.intercept_

y_pred1 = lr_model.predict(x_test)
mse1 = mean_squared_error(y_test, y_pred)
r21 = r2_score(y_test, y_pred) 

x1 = raisins['Class']

y1 = raisins['Perimeter']
plt.scatter(x1, y1)

#plt.plot(x1_test, y_pred1, color="red", linewidth=3)

plt.show()
"""
features =['Perimeter','MajorAxisLength','Eccentricity','ConvexArea']
X2 = raisins[features].values #don't actually need .values now, but it suppresses a Future Warning
Y = raisins['Class'].values
print(np.corrcoef(X2.T, Y.T)) #need to take the transpose to get column wise

fig, axes = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        k = j + i * 2
        feature = features[k]
        axes[i, j].scatter(X2[:, k], y, s=1) #s is marker size
        axes[i, j].set_xlabel(feature)
        axes[i, j].set_ylabel('Class')
plt.tight_layout()
plt.show()

X3 = raisins[raisins.columns[:-1]].values
Y = raisins['Class'].values

X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y, train_size=.7)
lr = LinearRegression()
lr_model3 = lr.fit(X_train3, Y_train3)
coefficients3 = lr_model3.coef_
intercept3 = lr_model3.intercept_

y_pred3 = lr_model3.predict(X_test3)
mse3 = mean_squared_error(Y_test3, y_pred3)
r2_3 = r2_score(Y_test3, y_pred3)

print(coefficients3)
print(intercept3)
print(r2_3)
print(mse3)

#############################################################################
x2 = raisins['MajorAxisLength'].values      
y2 = raisins['Class'].values
x2 = x2.reshape(-1, 1)
y2 = y2.reshape(-1, 1)
dtc = tree.DecisionTreeClassifier()
dtc = DecisionTreeClassifier(max_depth = 3, random_state = 0)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=.7)

dtc.fit(x2_train, y2_train)
y2_pred = dtc.predict(x2_test)

#Check the performace on test data
print('Test accuracy of decision tree 1:' , accuracy_score(y_true = y2_test, y_pred = y2_pred))

#View the tree
tree.plot_tree(dtc)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x2_train, y2_train)

y_pred = classifier.predict(x2_test)
#############################################################################
x3 = raisins['MajorAxisLength'].values      
y3 = raisins['Class'].values
x3 = x3.reshape(-1, 1)
y3 = y3.reshape(-1, 1)
dtc2 = tree.DecisionTreeClassifier()
dtc2 = DecisionTreeClassifier(max_depth = 5, random_state = 0)

x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, train_size=.7)

dtc2.fit(x3_train, y3_train)
y3_pred = dtc.predict(x3_test)

#Check the performace on test data
print('Test accuracy of decision tree 2:' , accuracy_score(y_true = y3_test, y_pred = y3_pred))

#View the tree
tree.plot_tree(dtc2)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x3_train, y3_train)

y_pred = classifier.predict(x3_test)
#############################################################################
x4 = raisins['MajorAxisLength'].values      
y4 = raisins['Class'].values
x4 = x4.reshape(-1, 1)
y4 = y4.reshape(-1, 1)
dtc3 = tree.DecisionTreeClassifier()
dtc3 = DecisionTreeClassifier(max_depth = 3, random_state = 0)

x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, train_size=.85)

dtc3.fit(x4_train, y4_train)
y4_pred = dtc.predict(x4_test)

#Check the performace on test data
print('Test accuracy of decision tree 3:' , accuracy_score(y_true = y4_test, y_pred = y4_pred))

#View the tree
tree.plot_tree(dtc3)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x4_train, y4_train)

y_pred = classifier.predict(x4_test)