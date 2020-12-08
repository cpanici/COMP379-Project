import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

# Load and pre-process
data = pd.read_csv('data/complete.csv')
data = data.drop('customer_id', axis=1)

X = data.loc[:, data.columns != 'card_offer'].values
y = data['card_offer'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=0,
                                                  stratify=y)

stdsc = StandardScaler().fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

def logistic() -> None:
  # Default model
  model = LogisticRegression(random_state=0)
  model.fit(X_train_std, y_train)
  print(f"The f1 score of the default LR model is: {f1_score(y_test, model.predict(X_test_std))}")

  # Optimizing hyperparameters


  model = LogisticRegression(solver='liblinear', random_state=0)
  parameters = {'penalty': ['l1', 'l2'], 
                'C': [.0001, .0005, .001, .005,  .01, .05,  1, 5, 10, 50, 100, 500, 1000]}

  skf = StratifiedKFold(n_splits=10)
  classifier = GridSearchCV(model, parameters, cv=skf, scoring="f1")
  classifier.fit(X_train_std, y_train)

  pd.DataFrame(classifier.cv_results_).sort_values(by='rank_test_score').head(5)


  # Best model
  best_model = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=0)
  best_model.fit(X_train_std, y_train)
  print('===========LOGISTIC REGRESSION MODEL ON TEST SET==========')
  print(f"The f1 score of the best LR model is: {f1_score(y_test, best_model.predict(X_test_std))}")
  print(f'Test set accuracy of best LR model: {best_model.score(X_test_std, y_test)}')

def dummy() -> None:
  # Baseline
  dummy1 = DummyClassifier(strategy='stratified')
  dummy2 = DummyClassifier(strategy='most_frequent')

  dummy1.fit(X_train_std, y_train)
  dummy2.fit(X_train_std, y_train)

  print('===========DUMMY MODELS ON TEST SET==========')
  print('Dummy 1: Stratified')
  print('f1 score:', f1_score(y_test, dummy1.predict(X_test_std)))

  print('Dummy 2: Most Frequent')
  print('f1 score:', f1_score(y_test, dummy2.predict(X_test_std)))

if __name__ == "__main__":
  dummy()
  print("\n" + "=" * 80 + "\n")
  logistic()