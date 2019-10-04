# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 04:22:28 2019

@author: Dilip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#import the data set
chun_dataframe=pd.read_csv('churn_modelling.csv')

#analysing data
sns.countplot(x="Exited", hue="Gender", data=chun_dataframe)
sns.countplot(x="Exited", hue="Geography", data=chun_dataframe)
chun_dataframe["Age"].plot.hist()
chun_dataframe["HasCrCard"].plot.hist()

#data wrangling
chun_dataframe.isnull()
chun_dataframe.isnull().sum()
sns.heatmap(chun_dataframe.isnull(), yticklabels=False,cmap="viridis" )


x= chun_dataframe.iloc[:,3:13]
y= chun_dataframe.iloc[:,13]


#creating dummie variables
Sex=pd.get_dummies(x ["Gender"], drop_first=True)
geography=pd.get_dummies(x["Geography"], drop_first=True)


#concate the data frame
x=pd.concat([x, Sex, geography], axis=1)

#drop unwanted coloumns
x.drop(["Gender", "Geography"], axis=1, inplace=True)

#spliting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#importing the keras libraries

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


#Initializing the ANN
classifier = Sequential()

#Adding the input layer and First hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation= 'relu', input_dim = 11))

'''classifier.add(Dropout(0.3))'''

#Adding the input layer and second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu'))

'''classifier.add(Dropout(0.4))'''

#Adding the out put layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

#Compailing the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
model_history = classifier.fit(x_train, y_train, validation_split = 0.33, batch_size = 10, nb_epoch = 100)

#llist all data in  histpry
print(model_history.history.keys())


# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predicting the test result
y_pred=classifier.predict(x_test)
y_pred=(y_pred > 0.5)

#calculate the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score 
cm = confusion_matrix(y_test, y_pred)
print('\n')
ac = accuracy_score(y_test, y_pred)








