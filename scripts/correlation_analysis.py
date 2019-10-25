import pandas as pd

X = pd.read_csv('../data/processed-X.csv')
y = pd.read_csv('../data/processed-y.csv', header=None)
y.drop(0, axis=1, inplace=True)
X.drop('Unnamed: 0', axis=1, inplace=True)

X['eng'] = y
X.corr()