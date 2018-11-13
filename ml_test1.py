import pandas as pd


titanic_df = pd.read_csv('titanic.csv')



def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


dat = pd.read_csv('titanic.csv')
df = pd.DataFrame(dat)
a=df[df['Age'].notnull()]
b=a['Age']
c=norm_arr(b)

def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()
    normalized = (arr - mean) / std
    return normalized
dat = pd.read_csv('titanic.csv')
df = pd.DataFrame(dat)
a=df[df['Age'].notnull()]


names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('indians-diabetes.csv', names=names)

import matplotlib.pyplot as plt
df.boxplot()
df.hist()
df.groupby('class').hist()

df.groupby('class').age.hist(alpha=0.4)
plt.show()