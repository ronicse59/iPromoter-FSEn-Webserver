from sklearn.svm import SVC

model = SVC()

from sklearn.externals import joblib
with open('model_dump.pkl', 'wb') as File:
     joblib.dump(model, File)

### Retrieve Model with joblib ###
from sklearn.externals import joblib
with open('model_dump.pkl', 'rb') as File:
     model = joblib.load(File)