import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix

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

names = ['Logistic Regression', 'Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'Multi-layer Perceptron', 'Stochastic Gradient Descent', 'AdaBoost', 'Gradient Boosting', 'Random Forest']
classifiers = [LogisticRegression(), BernoulliNB(), MultinomialNB(), MLPClassifier(), SGDClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]

file=open("ml-results.txt","w+")

for name, clf in zip(names, classifiers):
  clf.fit(X_train, y_train) 
  print('\r\n' + name + ' results:')
  file.write('\r\n' + name + ' results:')
  score_test = clf.score(X_test, y_test)
  score_val = clf.score(X_val, y_val)
  print('\r\nAccuracy (test): ' + str(score_test))
  file.write('\r\nAccuracy (test): ' + str(score_test))
  print('\r\nAccuracy (val): ' + str(score_val))
  file.write('\r\nAccuracy (val): ' + str(score_val))
  y_pred = clf.predict(X_test)  
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  print('\r\ntn = ' + str(tn) + ', fp = ' + str(fp) + ', fn = ' + str(fn) + ', tp = ' + str(tp))
  file.write('\r\ntn = ' + str(tn) + ', fp = ' + str(fp) + ', fn = ' + str(fn) + ', tp = ' + str(tp))
  fnr = fn / (fn + tn) # false-negative rate
  fpr = fp / (fp + tp) # false-positive rate
  print('\r\nfnr = ' + str(fnr) + ', fpr = ' + str(fpr))
  file.write('\r\nfnr = ' + str(fnr) + ', fpr = ' + str(fpr))
  p = tp / (tp + fp) # precision
  r = tp / (tp + fn) # recall
  f = 2 * ((p * r) / (p + r)) # f-score
  print('\r\nprecision = ' + str(p) + ', recall = ' + str(r) + ', f-score = ' + str(f))
  file.write('\r\nprecision = ' + str(p) + ', recall = ' + str(r) + ', f-score = ' + str(f))
file.close()
