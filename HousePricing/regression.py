import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, scale
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import csv

df = pd.read_csv("train.csv",index_col=0)
df2 = pd.read_csv("test.csv",index_col=0)
# Since there are many NaN values, we have a choice to make.
# The most straightforward thing to do is to drop all of the rows that have NaN, like below.
# However, this gets rid of 75% of the data, which may make our model less generalizable.

df = df.apply(LabelEncoder().fit_transform)
df2 = df2.apply(LabelEncoder().fit_transform)
#dfX = df.loc[:,'MSSubClass':'SaleCondition']
#dfY = df['SalePrice']



dataset = df.values
dataset2 = df2.values
X = dataset[:,:-1]
y = dataset[:,-1]
X2 = dataset2
id = range(1461, 2920)
#y2 = dataset2[:,-1]

#lb = LabelBinarizer()
#y_b = lb.fit_transform(y)

log_reg = LogisticRegression(penalty="l2")
log_reg.fit(X, y)

#y_pred = log_reg.predict(X)
y_pred = log_reg.predict(X2)
y_pred = y_pred * 1000
#print(y_pred)
y_pred = y_pred.tolist()
#id = id.tolist()
df = pd.DataFrame(y_pred,id)
# ~95.5% accuracy
#print("Model accuracy:", accuracy_score(y,y_pred))

df.to_csv('prediction.csv', sep='\t', quoting=csv.QUOTE_ALL)
