from knn import knn
from logistic_and_dummy import dummy, logistic
from randomforest import random_forest
from SVM import svm

models = [dummy, knn, logistic, random_forest, svm]

if __name__ == "__main__":
  for model in models:
    model()
    print("\n" + "=" * 80 + "\n")
