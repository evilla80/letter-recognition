from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV

import pandas as pd
df = pd.read_csv('letter-recognition.csv')
print(df.shape)
print(df.head)
x = df.drop(columns="letter",axis=1)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

y_orig = df["letter"]
l = LabelEncoder()
y = l.fit_transform(y_orig)
print("Target Label Encoded: \n", y)
y = to_categorical(y)
print("Target One Hot Encoded: \n", y)
X_train, X_test, y_train, y_test = train_test_split(
    x, #Intero dataset senza la colonna target
    y, #Colonna target codificata in One-Hot-Encoding
    test_size=0.2,
    shuffle=True,
    random_state=42
)
from keras.src.layers import Dense
from keras.src.models import Sequential

model = Sequential()
model.add(Dense(256, input_shape=(16, ), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

from keras.optimizers import Adam
import time
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=8e-5),
    metrics=['accuracy']
)
default_weights = model.get_weights()
start = time.time()
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    verbose=1
)
end = time.time()
print("Training time: ", end - start)

from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
import scipy.stats as st
k = 5
folds = KFold(n_splits=k, shuffle=True, random_state=42)
folds.split(x, y)
accuracy_k_fold_m = []
accuracy_k_fold_a = []

model.set_weights(default_weights)

for i, (train_index, valid_index) in enumerate(folds.split(x, y)):
        X_train, X_valid = x.iloc[train_index], x.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=150,
            batch_size=32,
            verbose=1,
            callbacks=[EarlyStopping(patience=7)]
        )
        y_pred_valid = model.predict(X_valid)
        accuracy = np.mean(np.argmax(y_pred_valid, axis=1) == np.argmax(y_valid, axis=1))
        accuracy_k_fold_m.append(accuracy)
        accuracy_k_fold_a.append(history.history['val_accuracy'][-1])


confint = st.t.interval(
    confidence=0.90,
    df=len(accuracy_k_fold_m) - 1,
    loc=np.mean(accuracy_k_fold_m),
    scale=st.sem(accuracy_k_fold_m)
)

print("Intervallo di confidenza = [", float(confint[0]), ", ", float(confint[1]), "]")
print(list(map(float, accuracy_k_fold_m)))
print(accuracy_k_fold_a)