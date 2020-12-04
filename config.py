import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
  def __init__(self):
    data = pd.read_csv('data/complete.csv')

    X = data.loc[:, data.columns != 'card_offer'].values
    y = data['card_offer'].values

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
      X, y,
      test_size=0.2,
      random_state=0,
      stratify=y
    )

