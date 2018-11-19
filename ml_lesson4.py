

import pandas as pd
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('indians-diabetes.csv', names=names)


import numpy as np
def stratified_split(y, proportion=0.8):
    y = np.array(y)

    train_inds = np.zeros(len(y), dtype=bool)
    test_inds = np.zeros(len(y), dtype=bool)

    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        np.random.shuffle(value_inds)

        n = int(proportion * len(value_inds))

        train_inds[value_inds[:n]] = True
        test_inds[value_inds[n:]] = True

    return train_inds, test_inds

train, test = stratified_split(df['class'])
X_train = df.iloc[train, 0:8]
X_test = df.iloc[test, 0:8]

y_train = df['class'][train]
y_test = df['class'][test]

from sklearn.linear_model import LogisticRegression
def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))


print(accuracy(y_test, y_pred))