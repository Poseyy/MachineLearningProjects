
# coding: utf-8

# # Project M1 - Cell Type Classification using Neural Network
# ### The objective of this project is to classify two cell types (spiny/aspiny) according to their electrophysiology features using both logistic regression and neural network.
# #### The data set is downloaded from the __[Allen Institute data base](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Computing-Electrophysiology-Features)__ and is already saved in the file "ElecPhyFeatures.csv".
# #### Two examples for classifying Iris data set using logistic regression and neural network are given in notebooks "logistic-regression-for-iris-classification.ipynb" and "Iris_NeuralNetworkTutorial.ipynb" respectively. You can start with these two examples before you work on the Allen's data set.

# ## Getting start with the Allen's data set
# Use python library Pandas to read the csv file. The data set is now stored in Pandas dataframe.

# In[9]:


import numpy as np
import pandas as pd

df = pd.read_csv("ElecPhyFeatures.csv",index_col=0)

# Since there are many NaN values, we have a choice to make.
# The most straightforward thing to do is to drop all of the rows that have NaN, like below.
# However, this gets rid of 75% of the data, which may make our model less generalizable.
df.dropna(inplace=True)

# Get rid of sparsely spiny cells
df = df[df.dendrite_type!='sparsely spiny']
df2 = df
df2 = df2.drop(['dendrite_type'], axis=1)
dataset = df.values
print(dataset.shape)
#df.head()
df2.head()


# Find the features with no missing data. You will choose any combination of features you want (avoid using those 'features' that are actually IDs) as input to the classifier. You will get bonus points if you figure out which features are most useful in distinguishing the cell type.

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df_full=df.drop(columns=['specimen_id','rheobase_sweep_id','thumbnail_sweep_id','id'])
df_full.head()


# The cell type is determined by the dendrite type in the last column of the data set. There is a minority type called "sparsely spiny". You can do either a 3-class classification or a binary classification excluding the "sparsely spiny".

# In[11]:



X = abs(df_full.iloc[:,:-1]) # Need to take absolute value for SelectKBest to work
y = df_full.iloc[:,-1]

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(df_full.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features


# Plot classes based on the two features selected. As you can see, the classes are more seperable under the feature "f_i_curve_slope" rather then "Vrest".

# ### Now you have defined the training data set and the class labels. Next train the logistic regression classifier and the neural network like in the two examples and compare the performance of these two methods.

# In[12]:


pick_feats = list(featureScores.nlargest(10,'Score').Specs) # make a list of the ten best features
pick_feats.append('dendrite_type') # add dendrite_type to the list

df_small = df[pick_feats] # Make a new DataFrame with our selected features
df_small2 = df_small
df_small2.drop(['dendrite_type'], axis=1)
#sns.pairplot(data=df_small, hue="dendrite_type")


# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, scale, Normalizer
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

dataset = df_small.values
dataset2 = df_small2.values
X = dataset[:,:-1]
y = dataset[:,-1]

lb = LabelBinarizer()
y_b = lb.fit_transform(y)

log_reg = LogisticRegression(penalty="l2")
log_reg.fit(X,y_b)

#X = normalize(X, norm=l2, axis=1, copy=True, return_norm=False)

y_pred = log_reg.predict(X)

print("Model accuracy:", accuracy_score(y_b,y_pred))

print(y_pred)


df_small2 = df_small2.drop(['dendrite_type'], axis=1)
df_small2.head()


# In[14]:


dataset = df_small.values
X = dataset[:,:-1]
y = dataset[:,-1]
# First we need to scale X so that the features are all on the same scale.
# Remember feature scaling from the machine learning coursera videos? That's what we're doing here.
#transformer = Normalizer().fit(X)
#Normalizer(copy=True, norm='l2')
#X = transformer.transform(X)
X_scale = scale(X)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=10, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

df_small2.head()


# In[15]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

df_small2.head()


# In[16]:


print(X_scale)
model = baseline_model();
model.fit(X_scale,dummy_y,epochs=25,verbose=1)
y_pred2 = model.predict_classes(X_scale)

df_small2.head()


# In[17]:


df_small2['neuralpred'] = y_pred2
df_small2['logisticpred'] = y_pred
#df_small2.drop(['dendrite_type'], axis=1)
df_small2.head()


# In[18]:


def baseline_model():
    model = Sequential()
    model.add(Dense(14, input_dim=12, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

df_small2.head()

dataset3 = df_small2.values

X_scale3 = scale(dataset3)

print(X_scale3)

model = baseline_model();
model.fit(X_scale3,dummy_y,epochs=300,verbose=1)
ensemble_pred = model.predict_classes(X_scale3)
