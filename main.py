import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
import pandas as pd
import os

def load_data(path, num):
    df1 = pd.read_csv(path, sep = '\t', header=None)
    df1.columns = ['time', 'Pw', 'distance']
    df1['sensor_nr'] = num
    df1 = df1.dropna() #filter
    df1 = df1[df1['distance'] < 20]
    df1 = df1[((df1['Pw'] < 50)&(df1['distance'] <10)).map(lambda x: not x)]
    return df1

df = []
fns = [i for i in os.listdir('../data') if 'distance' in i]

for i, j in enumerate(fns):
    print(j)
    df.append(load_data('../data/' + j, i))
    df[i] = df[i][['Pw', 'sensor_nr', 'distance']]


X_test = df[0].iloc[:, :-1]

m = pickle.load(open('../data/svr.m', 'rb'))
pred = m.predict(X_test)
error = m.score(X_test, pred)
print(pred, error)