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

print "Load Magic Bus data."
# Load MB data from CSV
mbdata = './MBDataWithSerenity_cleaned.csv'
# data = pd.read_csv(dataset_url)
# Read CSV with semicolon separator
data = pd.read_csv(mbdata, nrows=100) # <-- limit load to 10,000 while we test!
# Output the first 5 rows of data
## print data.head()
print data.shape # (1599, 12) <-- 1,599 samples x 12 features
# Summary statistics
# print data.describe()
# Double ## indicates commands that we will need to uncomment as we progress...
# print "Step 4: Split data into training and test sets."
# Separate target from training features
y = data.intSPITStatus
X = data.drop(['intSPITStatus', 'SPITStatus', 'SPITCause', 'WaitStatus', 'DID', 'ANI'], axis=1)

print 'Load model: mbdata_rf_regressor.pkl'
clf = joblib.load('mbdata_rf_regressor.pkl')

print 'Predict data set using loaded model'
result = clf.predict(X)
# print result
# print y
print "Model Score: {0}".format(clf.score(X, y))
target_id = 0
print_lines = True # False 
if print_lines:
  for item in result:
    print "Prediction: {0} <----> Observation: {1} --> Diff: {2}".format(item, y[target_id], np.diff([item, y[target_id]])[0])
    target_id += 1



