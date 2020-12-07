import pandas as pd
import numpy as np

# Training dataset
data = pd.read_csv('data/Test 1.csv')

# Recode target variable
data['card_offer'] = data['card_offer'].replace({True: 1, False: 0})

# Replace categorical features with dummy variables
categoricals = data.select_dtypes(include='object')
dummies = pd.get_dummies(categoricals)

numerics = data.select_dtypes(exclude='object')
complete = pd.merge(numerics, dummies, left_on=numerics.index, right_on=dummies.index)
complete = complete.drop('key_0', axis=1)

complete.to_csv('data/complete.csv')

# Unseen dataset
data = pd.read_csv('data/Test 2.csv')

data = data.drop('card_offer', axis=1)

categoricals = data.select_dtypes(include='object')
dummies = pd.get_dummies(categoricals)

numerics = data.select_dtypes(exclude='object')
complete = pd.merge(numerics, dummies, left_on=numerics.index, right_on=dummies.index)
complete = complete.drop('key_0', axis=1)

complete.to_csv('data/unseen.csv', index=False)