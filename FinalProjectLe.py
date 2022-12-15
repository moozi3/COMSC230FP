#Name: Danh Le, Lauren Moulaison, Wyatt VanDyk
#Course: COMSC230
#Prof. Name: Prof. Rivera Morales
#Program Name: FinalProjectLe
#Program brief description: Run an analysis of the raisin dataset


import pandas
import numpy 
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

raisins = pandas.read_csv('C:/Users/ledan/Downloads/Raisin_Dataset.csv')
raisins['Class'].replace(['Kecimen', 'Besni'],[0, 1], inplace=True)

x = raisins['Class'].values #This variable stayed consistent throughout the production of the graphs
y = raisins['Perimeter'].values #Changed this variable to each of the variables in the raisin data set
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7)

lr = LinearRegression()
lr_model= lr.fit(x_train, y_train)
coefficients = lr_model.coef_
intercept = lr_model.intercept_

y_pred = lr_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

x = raisins['Class'] #Same as above

y = raisins['Perimeter'] #Same as above

plt.scatter(x, y)
plt.plot(x_test, y_pred, color = "blue", linewidth=3)
plt.show()
