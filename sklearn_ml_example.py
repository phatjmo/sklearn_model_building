# Step 2: Import libraries and modules.
# NumPy
import numpy as np
# Pandas
import pandas as pd
# Import sampling helper
from sklearn.model_selection import train_test_split
# Import preprocessing modules
from sklearn import preprocessing
# Import random forest model
from sklearn.ensemble import RandomForestRegressor
# Import cross-validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# Import evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
# Import module for saving scikit-learn models
from sklearn.externals import joblib

# Step 3: Load red wine data.
# Load wine data from remote URL
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# data = pd.read_csv(dataset_url)
# Read CSV with semicolon separator
data = pd.read_csv(dataset_url, sep=';')
# Output the first 5 rows of data
# print data.head()
# print data.shape # (1599, 12) <-- 1,599 samples x 12 features
# Summary statistics
# print data.describe()

# Step 4: Split data into training and test sets.
# Separate target from training features
y = data.quality
X = data.drop('quality', axis=1)
# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)
# Step 5: Declare data preprocessing steps.
# Fitting the Transformer API
scaler = preprocessing.StandardScaler().fit(X_train)
# Applying transformer to training data
X_train_scaled = scaler.transform(X_train)
# print X_train_scaled.mean(axis=0)
# print X_train_scaled.std(axis=0)
# Applying transformer to test data
X_test_scaled = scaler.transform(X_test)
# print X_test_scaled.mean(axis=0)
# print X_test_scaled.std(axis=0)
# Pipeline with preprocessing and model
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# Step 6: Declare hyperparameters to tune
# List tunable hyperparameters
# print pipeline.get_params()
# Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth' : [None, 5, 3, 1]}

# Step 7: Tune model using a cross-validation pipeline.
# Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)
print clf.best_params_

#Step 8: Refit on the entire training set.
#Confirm model will be retrained
print clf.refit

# Step 9: Evaluate model pipeline on test data.
# Predict a new set of data
y_pred = clf.predict(X_test)
print r2_score(y_test, y_pred)

print mean_squared_error(y_test, y_pred)

# Step 10: Save model for future use.
# Save model to a .pkl file
joblib.dump(clf, 'rf_regressor.pkl')

# Load model from .pkl file
clf2 = joblib.load('rf_regressor.pkl')

# Predict data set using loaded model
clf2.predict(X_test)

