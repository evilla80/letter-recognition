import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('letter-recognition.csv')
print(df.shape)

x = df.drop(columns="letter",axis=1)
y = df["letter"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True, random_state=42)

svm = svm.SVC(kernel='rbf', C = 1000.0, random_state=42)
svm.fit(X_train, y_train)


start_val = 0.1
c_vec = [start_val]
for i in range(1):
    c_vec.append(c_vec[-1] * 1.5)

print(c_vec)
grid = {
    'kernel': ['rbf'],
    'C': c_vec,
    'gamma' : [0.1]
}

gscv = GridSearchCV(
    svm,
    grid,
    scoring='accuracy',
    cv=5,
    verbose=10
)

gscv.fit(X_train, y_train)

print("Best params", gscv.best_params_)
y_pred = svm.predict(X_test)

accuracy = svm.score(X_test, y_test)

print("accuracy1 = ", accuracy)