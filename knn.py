import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from config import Data

def main() -> None:
  data = Data()

  model = KNeighborsClassifier(n_neighbors=3)

  parameters = {
    'n_neighbors': [nbr for nbr in range(1, 10)], 
    'weights': ["uniform", "distance"],
    'leaf_size': [10 * nbr for nbr in range(1, 5)]
  }

  skf = StratifiedKFold(n_splits=10)
  classifier = GridSearchCV(model, parameters, cv=skf, scoring="f1")
  classifier.fit(data.X_train, data.y_train)

  print(f"The best parameters were {classifier.best_params_}")
  print(f"The f1 score of the best KNN model is: {f1_score(data.y_test, classifier.predict(data.X_test))}")
  print(f'The accuracy of best KNN model is: {classifier.score(data.X_test, data.y_test)}')

if __name__ == "__main__":
  main()