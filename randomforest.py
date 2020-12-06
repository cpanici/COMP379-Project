
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

dataset = pd.read_csv("data/complete.csv")

# Excluding customer ID and index because they're not relevant features
new_dataset = dataset.iloc[:, 2:18]
features = new_dataset.loc[:, new_dataset.columns != 'card_offer'].values
labels = new_dataset['card_offer'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.2,
                                                    random_state=23,
                                                    stratify=labels)

# Scale both the training features and the test features

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a random forest classifier using grid search onto the training data

rf = RandomForestClassifier()
params = {'n_estimators': [100, 200, 400, 600, 800, 1000], 'max_depth': [10, 20, 30]}
rf_classifier = GridSearchCV(rf, params)
rf_classifier.fit(X_train_scaled, y_train)

print(rf_classifier.best_estimator_)

# The best model for a random forest uses 800 trees and a max depth of 30 in this case

# Make predictions of the y values for the test set now
predictions_train = rf_classifier.predict(X_train_scaled)
predictions_test = rf_classifier.predict(X_test_scaled)

print(f'Test accuracy of best random forest model: {accuracy_score(y_test, predictions_test)}')
print(f'Test f1 score of best random forest model: {f1_score(y_test, predictions_test)}')

# Test accuracy of best random forest model: 0.9775
# Test f1 score of best random forest model: 0.9231
