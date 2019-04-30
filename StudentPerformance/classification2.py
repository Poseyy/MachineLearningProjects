import numpy as np
import pandas as pd

from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn import datasets

import theano as th
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, scale
from sklearn.metrics import mean_squared_error

df = pd.read_csv("StudentsPerformance.csv",index_col=0)
df.dropna(inplace=True)
f = range(0,20)
d = range(20,40)
c = range(40,50)
b = range(50,80)
a = range(80,101)
f2 = [str(x) for x in f]
d2 = [str(x) for x in d]
c2 = [str(x) for x in c]
b2 = [str(x) for x in b]
a2 = [str(x) for x in a]

# seaborn viz for math score
sns.distplot(df['math score'], bins=30)
plt.xlabel('math score')
plt.ylabel('count')
plt.title('Math Scores')
#plt.show()

# seaborn viz for gender difference
df['gender'] = df.index
sns.barplot(x='gender', y='math score', data=df)
#plt.show()

# math score replace
df.replace(f, 'F', inplace=True)
df.replace(d, 'D', inplace=True)
df.replace(c, 'C', inplace=True)
df.replace(b, 'B', inplace=True)
df.replace(a, 'A', inplace=True)
#test prep
df.replace('none', 0, inplace=True)
df.replace('completed', 1, inplace=True)
#df['test preparation course'] = df.index
sns.barplot(x = 'test preparation course', y = 'math score', data=df)
#plt.show()
# education replace
df.replace('some high school', 0, inplace=True)
df.replace('high school', 1, inplace=True)
df.replace('some college', 2, inplace=True)
df.replace("associate's degree", 3, inplace=True)
df.replace("bachelor's degree", 4, inplace=True)
df.replace("master's degree", 5, inplace=True)
sns.barplot(x='parental level of education',y='math score', data=df)
plt.xticks(rotation=90)
# plt.show()
# df.replace("associate's degree", 3, inplace=True)
# lunch replace
df.replace('free/reduced', 0, inplace=True)
df.replace('standard', 1, inplace=True)
# race
df.replace('group A', 0, inplace=True)
df.replace('group B', 1, inplace=True)
df.replace('group C', 2, inplace=True)
df.replace("group D", 3, inplace=True)
df.replace("group E", 4, inplace=True)
# gender
df.replace('female', 0, inplace=True)
df.replace('male', 1, inplace=True)
df

sel_feature = ['lunch','parental level of education', 'test preparation course'] # Select features
X = df[sel_feature].values
Y = df.pop('math score')

sel_feature = ['lunch','parental level of education','test preparation course', 'gender', 'race/ethnicity'] # Select features
X1 = df1[sel_feature].values
Y1 = df1['math score'].values
Y1 = Y1.flatten()
X_scale1 = scale(X1)
#X2 = df2[sel_feature].values
#Y2 = df2['math score'].values
#Y2 = Y2.flatten()
#X_scale2 = scale(X2)
#print(X)

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X1,Y1)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(df.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(5,'Score'))  #print 10 best features

sns.pairplot(data=df_small, hue="math score")

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
ylabel = encoder.transform(['A','B','C','D','F'])
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# build model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=5, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# train model
model1 = baseline_classification_model();
history = model1.fit(X,dummy_y,epochs=100,verbose=0)
y_pred = model1.predict_classes(X)
print(y_pred)
print("Model accuracy:", accuracy_score(dummy_y,y_pred))

# build model
def baseline_regression_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=3, activation='relu'))
	model.add(Dense(1, activation='linear'))
	# Compile model
	model.compile(loss='mse', optimizer='adam')
	return model

# train model
model2 = baseline_regression_model();
history = model2.fit(X,Y,epochs=100,verbose=0)
y_pred = model2.predict(X)
print(y_pred)

m = mean_squared_error(Y, y_pred)
print(sqrt(m))
