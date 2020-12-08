import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from config import Data

def knn() -> None:
  data = Data()

  model = KNeighborsClassifier()
  model.fit(data.X_train, data.y_train)
  print(f"The baseline F1 score for KNN is: {f1_score(data.y_test, model.predict(data.X_test))}")
  parameters = {
    'n_neighbors': [nbr for nbr in range(1, 10)], 
    'weights': ["uniform", "distance"]
  }

  skf = StratifiedKFold(n_splits=10)
  classifier = GridSearchCV(model, parameters, cv=skf, scoring="f1")
  classifier.fit(data.X_train, data.y_train)
  print(f"The best holdout F1 score is {classifier.best_score_}")
  model = KNeighborsClassifier(**classifier.best_params_)
  model.fit(data.X_train, data.y_train)
  print(f"The best parameters were {classifier.best_params_}")
  print('===========K-NEAREST NEIGHBORS MODEL ON TEST SET==========')
  print(f"The F1 score of the best KNN model is: {f1_score(data.y_test, model.predict(data.X_test))}")
  print(f'The accuracy of best KNN model is: {model.score(data.X_test, data.y_test)}')

if __name__ == "__main__":
  knn()