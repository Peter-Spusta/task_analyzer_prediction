# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e1P2rOgOZM1K-Pz25QRpCI5-1Ad8CNAS
"""

# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle


url = "user-tasks.csv"
names = ['cmdCnt','duration','score','cmdRedund','rating']
dataframe = pandas.read_csv(url, names=names)

dataframe.head(5)

array = dataframe.values
X = array[:,0:4]
Y = array[:,3]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
filename = 'finalized_model_trainings.sav'
pickle.dump(model, open(filename, 'wb'))
 
print('model_Training done')
# some time later...

