import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data/complete.csv')

X = data.loc[:, data.columns != 'card_offer'].values
y = data['card_offer'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=0,
                                                   stratify=y)

stdsc = StandardScaler().fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

model = LogisticRegression(solver='liblinear', random_state=0)
parameters = {'penalty': ['l1', 'l2'], 
              'C': [.0001, .0005, .001, .005,  .01, .05,  1, 5, 10, 50, 100, 500, 1000]}

skf = StratifiedKFold(n_splits=10)
classifier = GridSearchCV(model, parameters, cv=skf)
classifier.fit(X_train, y_train)

pd.DataFrame(classifier.cv_results_).sort_values(by='rank_test_score').head(5)

best_model = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=0)
best_model.fit(X_train, y_train)
print(f'Test set accuracy of best model: {best_model.score(X_test, y_test)}')