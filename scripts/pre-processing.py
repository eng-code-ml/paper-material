import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV


X = pd.read_csv('/Users/otaviolemos/Dropbox/academic/projects/eng-vs-noneng/data-sloc-3/data-sloc-3.csv')
y = pd.read_csv('/Users/otaviolemos/Dropbox/academic/projects/eng-vs-noneng/data-sloc-3/target-sloc-3.csv', header=None)
y.drop(0, axis=1, inplace=True)
X.drop('Unnamed: 0', axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=1)

y_train = y_train.values.ravel()
y_val = y_val.values.ravel()


# Feature Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Feature Selection
# Feature ranking with recursive feature elimination and cross-validated selection of the best number of features
clf = RandomForestClassifier()
selector = RFECV(clf, step=1, cv=5)
selector = selector.fit(X, y)
