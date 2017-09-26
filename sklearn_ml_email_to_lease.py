# Step 2: Import libraries and modules.
# NumPy
import numpy as np
# Pandas
import pandas as pd
# Import sampling helper
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
# Import preprocessing modules
from sklearn import preprocessing
# Import random forest model
from sklearn.ensemble import RandomForestRegressor
# Import cross-validation pipeline
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
# Import evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
# Import module for saving scikit-learn models
from sklearn.externals import joblib
# Text tools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def xml2df(xml_data):
    root = ET.XML(xml_data) # element tree
    all_records = []
    for i, child in enumerate(root):
        record = {}
        for subchild in child:
            record[subchild.tag] = subchild.text
            all_records.append(record)
    return pd.DataFrame(all_records)


print "Step 3: Load Lease data."
# Load MB data from CSV
# leasedata = './Emails_to_Leases_09252017.csv'
# data = pd.read_csv(dataset_url)
# Read CSV with semicolon separator
# data = pd.read_csv(leasedata) #, nrows=100) # <-- limit load to 10,000 while we test!

## Reading from XML for full Email Body <-- Doesn't work, ugly export 
# xml_file = open('./Emails_to_Leases_09252017.xml').read()
# data = xml2df(xml_file)

## Read from XLSX <-- Requires pip install xlrd
xlsx_file = pd.ExcelFile('./Emails_to_Leases_09252017.xlsx')
data = xlsx_file.parse("Export Worksheet")

# Output the first 5 rows of data
## print data.head()
print data.shape # (1599, 12) <-- 1,599 samples x 12 features
# Summary statistics
print data.describe()
# Double ## indicates commands that we will need to uncomment as we progress...
print "Step 4: Split data into training and test sets."
# Separate target from training features
y = data.turnedintolease
# Drop data we can't use with this regression model. Better textual analysis is required
X = data.drop(['turnedintolease',
                'pmsEmail',
                'siteid',
                'adsource',
                'Subject',
                #'Body',
                'Comments',
                'EmailFormatTypeId',
                'RelationshipTypeId',
                'TimelineStatusTypeId',
                'Beds',
                'Baths',
                'ResponseRequired',
                'ActionRequired'],
                axis=1)
# Testing Vectorizers

# Try the CountVectorizer --> Max .29 with bigrams
# vectorizer = CountVectorizer()
#vectorizer = CountVectorizer(ngram_range=(1, 4),
#                             token_pattern=r'\b\w+\b', min_df=1)
# Try the TfidVectorizer --> Max .32 with bigrams, and smooth_idf
vectorizer = TfidfVectorizer(ngram_range=(1, 2), smooth_idf=True,
                             token_pattern=r'\b\w+\b', min_df=1)
X['Body'] = vectorizer.fit_transform(X['Body'].values.astype('U')).toarray()
# print X['Comments']
# X['Comments'] = vectorizer.fit_transform(X['Comments'].values.astype('U')).toarray()
# X['Subject'] = vectorizer.fit_transform(X['Subject'].values.astype('U')).toarray()
# X['adsource'] = vectorizer.fit_transform(X['adsource'].values.astype('U')).toarray()
# print X['Comments']
# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    stratify=y)
print "Step 5: Declare data preprocessing steps."
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

print "Step 6: Declare hyperparameters to tune."
# List tunable hyperparameters
# print pipeline.get_params()
# Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                    'randomforestregressor__max_depth' : [None, 5, 3, 1]}

print "Step 7: Tune model using a cross-validation pipeline."
# Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10, verbose=True)

# Fit and tune model
clf.fit(X_train, y_train)
print clf.best_params_

print "Step 8: Refit on the entire training set."
# Confirm model will be retrained
print clf.refit

print "Step 9: Evaluate model pipeline on test data."
# Predict a new set of data
y_pred = clf.predict(X_test)
print r2_score(y_test, y_pred)

print mean_squared_error(y_test, y_pred)

print "Step 10: Save model for future use."
# Save model to a .pkl file
joblib.dump(clf, 'leasedata_rf_regressor.pkl')

# Load model from .pkl file
clf2 = joblib.load('leasedata_rf_regressor.pkl')

# Predict data set using loaded model
clf2.predict(X_test)
