#!/usr/bin/python3

#libary imports

import pandas as pd #uses pandas library and labels it as pd
import numpy as np #uses numpy library and labels it as np
from sklearn.svm import SVC #uses SVC from sklearn.svm libary
from sklearn import metrics #uses metrics from sklearn
from sklearn.model_selection import train_test_split #uses train_test_split from sklearn.model_selection
from sklearn.preprocessing import StandardScaler #uses standard scaler to standardizer the date from sklearn.preprocessing
from sklearn.model_selection import StratifiedKFold #uses StratifiedKFold from sklearn.model_selection
from sklearn.model_selection import GridSearchCV #uses GridSearchCV from sklearn.model_selection

#data processing

#importing

data = pd.read_csv('data/complete.csv') #importing of cleaned data set

#cleaning

data = data.drop('customer_id', axis=1) #drops costmers ids

#spliting

X = data.loc[:, data.columns != 'card_offer'].values #gets all columns in cleaned data set but there label
y = data['card_offer'].values #gets labels from cleaned data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y) #spliting in to test and training sets

#Standardizing 

stdsc = StandardScaler().fit(X_train) #fits X_train to a standard sclaer
X_train_std = stdsc.transform(X_train) #transforms X_train to the standardized set
X_test_std = stdsc.transform(X_test) #transforms X_train to the standardized set

#SVM

#default

svmClassifer = SVC() #creates a support vector machine from sklearn.svm with default parameters
svmClassifer.fit(X_train_std, y_train.ravel()) #fits the svm to the training data set
devPredict = svmClassifer.predict(X_test_std) #predicts the labels for test
print("Default parameter accuracy:", metrics.accuracy_score(y_test, devPredict)) #gets accuracy score for svm with default parameters
print("Test f1 score of default parameter:", metrics.f1_score(y_test, devPredict))#printing the f1 scores of the default svc model

#Gaussian kernel

svmGaussianClassifer = SVC(kernel='rbf') #creates a support vector machine from sklearn.svm with Gaussian kernel
svmGaussianClassifer.fit(X_train_std, y_train.ravel()) #fits the svm to the training data set
devGaussianPredict = svmGaussianClassifer.predict(X_test_std) #predicts the labels for test
print("Gaussian kernel accuracy:", metrics.accuracy_score(y_test, devGaussianPredict)) #gets accuracy score for svm with Gaussian kernel
print("Test f1 score of gaussian kernel accuracy parameter", metrics.f1_score(y_test, devGaussianPredict))#printing the f1 scores of the Gaussian kernel model

#Sigmoid kernel

svmSigmoidClassifer = SVC(kernel='sigmoid') #creates a support vector machine from sklearn.svm with Sigmoid kernel
svmSigmoidClassifer.fit(X_train_std, y_train.ravel()) #fits the svm to the training data set
devSigmoidPredict = svmSigmoidClassifer.predict(X_test_std) #predicts the labels for test
print("Sigmoid kernel accuracy:", metrics.accuracy_score(y_test, devSigmoidPredict)) #gets accuracy score for svm with Sigmoid kernel
print("Test f1 score of sigmoid kernel:", metrics.f1_score(y_test, devSigmoidPredict))#printing the f1 scores of the sigmoid kernel model

#Tunning

tuned_parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},] #Exhaustive Grid Search parameters
skf = StratifiedKFold(n_splits=10) #defines how many folds to do on find best parameters
svmTuned = SVC(kernel='rbf') #creates a support vector machine from sklearn.svm with Gaussian kernel
svmTunedClassifier = GridSearchCV(svmTuned, tuned_parameters, cv = skf) #preforms grid search to find optimal paramaters
svmTunedClassifier.fit(X_train_std, y_train.ravel()) #fits the svm to the training data set
svmTunedClassifier.best_estimator_ #returns best estiamte of optimal parameters 

#Predicting with best svc parameters

svmBestClassifier = SVC(C=1000, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False) #creates a support vector machine from sklearn.svm with best kernel from tunning
svmBestClassifier.fit(X_train_std, y_train.ravel()) #fits the svm to the training data set
svmBestPredict = svmBestClassifier.predict(X_test_std) #predicts the labels for test
print("Best kernel accuracy:", metrics.accuracy_score(y_test, svmBestPredict)) #gets accuracy score for svm with the best kernel
print("Test f1 score of best kernel accuracy parameter", metrics.f1_score(y_test, svmBestPredict)) #printing the f1 scores of the best model

#Best model prediction on unseen data

unseen = pd.read_csv('data/unseen.csv') #importing of unseen data set
unseen = unseen.drop('customer_id', axis=1) #drops costmers ids
unseen_std = stdsc.transform(unseen) #transforms X_train to the standardized set
unseenPredict = svmBestClassifier.predict(unseen_std) #predicts the labels for test
print("Predicting on unseen data... \nUnseen data predicted counts:", unseenPredict) #printing of predictions of unseen data
