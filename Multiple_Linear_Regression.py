# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 22:43:57 2021

@author: swapn
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#import dataset in form of csv file 
dataset = pd.read_csv("50_startups.csv")

#seprate depending and independing veriable 
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

states=pd.get_dummies(x['State'], drop_first=True)
x=x.drop('State', axis=1)

x=pd.concat([x,states], axis=1)

#spliting the dataset into the Training and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)

#fitting multiple linear regression into training set 
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train, y_train)

#prediction the test data result
y_pred=regression.predict(x_test)

#r2_score is use to mesure model is good or not
from sklearn.metrics import r2_score
score=r2_score(y_test, y_pred) 