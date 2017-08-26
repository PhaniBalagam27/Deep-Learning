# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:29:08 2017

@author: phani
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing  the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Importing Keras library
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

# Adding the input layer and first Hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))    

# Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))    

# output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))    

# Compiling the ANN
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit ANN to taining set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))    
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))    
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))    
    classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)


mean = accuracies.mean()
Variance = accuracies.std()


# Parameter tuning


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))    
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))    
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))    
    classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'nd_epoch': [100, 500]}




