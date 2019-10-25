import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


X = pd.read_csv('../data/processed-X.csv')
y = pd.read_csv('../data/processed-y.csv', header=None)
y.drop(0, axis=1, inplace=True)
X.drop('Unnamed: 0', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=1)

y_train = y_train.values.ravel()
y_val = y_val.values.ravel()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

max_depth = [2, 10, None]
max_features = ['auto', 'sqrt']
n_estimators = [100, 200, 400]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 5]
bootstrap = [True, False]

hyperF = dict(max_depth = max_depth, max_features = max_features, n_estimators = n_estimators, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, bootstrap = bootstrap)
clf = RandomForestClassifier()

gridF = GridSearchCV(clf, hyperF, cv = 3, verbose = 1,  n_jobs = -1)

bestClf = gridF.fit(X_train, y_train)
