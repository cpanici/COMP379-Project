#!/usr/bin/python3

#libary imports

import pandas as pd #uses pandas library and labels it as pd
import numpy as np #uses numpy library and labels it as np
from sklearn.svm import SVC #uses SVC from sklearn.svm libary
from sklearn import metrics #uses metrics from sklearn 
from sklearn.model_selection import train_test_split # uses train_test_split from sklearn.model_selection

def svm() -> None:
  #data processing

  #importing

  data = pd.read_csv('data/complete.csv') #importing of cleaned data set

  #cleaning

  data = data.drop('customer_id', axis=1) #drops costmers ids

  #spliting

  X = data.loc[:, data.columns != 'card_offer'].values #gets all columns in cleaned data set but there label
  y = data['card_offer'].values #gets labels from cleaned data set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y) #spliting in to test and training sets

  print('===========SUPPORT VECTOR MACHINE MODELS ON TEST SET==========')
  #SVM

  #default

  svmClassifer = SVC() #creates a support vector machine from sklearn.svm with default parameters
  svmClassifer.fit(X_train, y_train.ravel()) #fits the svm to the training data set
  devPredict = svmClassifer.predict(X_test) #predicts the labels for test
  print("Default parameter accuracy:",metrics.accuracy_score(y_test, devPredict)) #gets accuracy score for svm with default parameters

  #Gaussian kernel

  svmGaussianClassifer = SVC(kernel='rbf') #creates a support vector machine from sklearn.svm with Gaussian kernel
  svmGaussianClassifer.fit(X_train, y_train.ravel()) #fits the svm to the training data set
  devGaussianPredict = svmGaussianClassifer.predict(X_test) #predicts the labels for test
  print("Gaussian kernel accuracy:",metrics.accuracy_score(y_test, devGaussianPredict)) #gets accuracy score for svm with Gaussian kernel

  #Sigmoid kernel

  svmSigmoidClassifer = SVC(kernel='sigmoid') #creates a support vector machine from sklearn.svm with Sigmoid kernel
  svmSigmoidClassifer.fit(X_train, y_train.ravel()) #fits the svm to the training data set
  devSigmoidPredict = svmSigmoidClassifer.predict(X_test) #predicts the labels for test
  print("Sigmoid kernel accuracy:",metrics.accuracy_score(y_test, devSigmoidPredict)) #gets accuracy score for svm with Sigmoid kernel

if __name__ == "__main__":
  svm()