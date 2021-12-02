#!/usr/bin/env
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report


def runmodel():
    data = pd.read_csv('test.csv',sep=';')
    X_test = data.drop(['quality'], axis = 1)
    y_test = data['quality']
    from sklearn.preprocessing import StandardScaler
    norm = StandardScaler()
    XT = norm.fit_transform(X_test)
    yT = data['quality']
    model = load('./savedmodel')
    y_pred = model.predict(XT)
    print(classification_report(yT, y_pred))
        
if __name__ == '__main__':
    runmodel()