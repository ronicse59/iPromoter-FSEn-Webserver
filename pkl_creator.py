from __future__ import division
import methods
import random

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

# Define Important Library
#====================================================================
import pandas as pd

# scikit-learn library import
#====================================================================
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib

#====================================================================
# Common error to set file path
# http://help.pythonanywhere.com/pages/NoSuchFileOrDirectory/
# There are two approach to setting csv file path
# One is to use absolute path: in our case: /home/ipro70/mysite
'''
import urllib2
url = 'https://raw.githubusercontent.com/ronicse59/iPromoter70/master/selected_feature.csv'
file_path = urllib2.urlopen(url)
#D = pd.read_csv(file_path, header=None)
'''

#====================================================================
# First time add belows comment out code for make pkl file by your model fitted data

features_file_name = "FeaturesEnsembleVote.csv"
D = pd.read_csv(features_file_name, header=None)

# Divide features (X) and classes (y) :
#====================================================================
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

# Encoding y :
#====================================================================
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

print ('-> After transform total features : ',len(X[1]))
print ('-> Start classification   ...')

# Define classifiers within a list
#====================================================================
import method_ensemble


# Spliting with 10-FCV :
#====================================================================
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)

#====================================================================
from sklearn.pipeline import Pipeline
a = []
b = []
c = []
d = []
e = []

l = len(X[0])

from methods import add_index
a = add_index(a,0,16202)
b = add_index(b,0,8)
b = add_index(b,5546,16384)
c = add_index(c,0,5)
c = add_index(c,16201,16384)


pipe1 = Pipeline([
               ('sel', method_ensemble.ColumnSelector(a)),
               ('clf', SVC(kernel='rbf', C=4, probability=True, decision_function_shape='ovo', tol=0.1, cache_size=200))])
pipe2 = Pipeline([
               ('sel', method_ensemble.ColumnSelector(b)),
               ('clf', LinearDiscriminantAnalysis(n_components=500))])
pipe3 = Pipeline([
               ('sel', method_ensemble.ColumnSelector(c)),
               ('clf', LogisticRegression(random_state=0, n_jobs=1000))])

# Define classifiers within a list
#====================================================================

model = EnsembleVoteClassifier(clfs=[pipe1,pipe2,pipe3], weights=[0.42,0.20,0.38], voting='soft')

# Save to file in the current working directory
joblib_file = "model.pkl"
joblib.dump(model.fit(X, y), joblib_file)