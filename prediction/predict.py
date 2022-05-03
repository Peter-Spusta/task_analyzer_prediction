import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

url = "tasks-for-prediction.csv"
names = ['id','cmdCnt','duration','score','cmdRedund','complexion']
dataframe = pandas.read_csv(url, names=names)

dataframe.head(5)

array = dataframe.values
X = array[:,1:5]

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

Ynew = model.predict(X)
dataframe.complexion = Ynew

import csv

dataframe.to_csv("./predicted-task.csv",",")

print(dataframe)
