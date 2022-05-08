import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

url = "tasks-for-training-prediction.csv"
names = ['cmdCnt','duration','score','cmdRedund','rating']
dataframe = pandas.read_csv(url, names=names)

dataframe.head(5)

array = dataframe.values
X = array[:,0:4]

filename = 'finalized_model_trainings.sav'
model = pickle.load(open(filename, 'rb'))

Ynew = model.predict(X)
dataframe.complexion = Ynew

import csv

dataframe.to_csv("./predicted-training.csv",",")

print(dataframe)
