import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
from keras.src.layers import Dense
from keras.src.models import Sequential
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def geterror(l):
    r = []
    for e in l:
        r.append(1. - e)
    return r

DF = pd.read_csv("letter-recognition.csv")

x = DF.drop(columns="letter",axis=1)
y = DF["letter"]
l = LabelEncoder()
y = l.fit_transform(y)
y = keras.utils.to_categorical(y)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)
pca = PCA().fit(scaled_data)
plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Componenti della PCA')
plt.ylabel('Varianza spiegata')
plt.title("Varianza spiegata da ogni componente")
plt.show()

# Applica la PCA per ridurre la dimensionalità dei dati
pca = PCA(n_components=2).fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Quantifichiamo  la percentuale di varianza spiegata da ciascuno delle componenti.
print(pca.explained_variance_ratio_)


#Verifichiamo il coefficienti degli autovalori
eigenvalues = pca.explained_variance_
n_samples = scaled_data.shape[0]
cov_matrix = np.dot(scaled_data.T, scaled_data) / n_samples
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
    #print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)

ax = sns.heatmap(pca.components_,
                 cmap='YlGnBu',
                 yticklabels=[ "PCA"+str(x) for x in range(1,pca.n_components_+1)],
                 xticklabels=list(DF.columns[1 : 17]),
                 cbar_kws={"orientation": "vertical"})
ax.set_aspect("equal")
plt.show()

print(abs(pca.components_))

plt.clf()
corr_matrix = x.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
'''
model = Sequential()
model.add(Dense(300, input_shape=(16, ), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

plt.title('Learning Rate = 0.00008 (8e-5)')
plt.ylabel('Loss sul training set')
plt.xlabel('Epoche')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True, random_state=42)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=8e-5), metrics=['accuracy'])
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1
)






plt.plot(geterror(history.history['accuracy']))
plt.plot(geterror(history.history['val_accuracy']))
plt.title('Errori su training set/test set')
plt.ylabel('Errore')
plt.xlabel('Epoche')
plt.legend(['Errore sui dati di training', 'Errore sui dati di test'], loc='upper left')
plt.savefig("plots/precisione.png")



plt.clf()

plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("plots/perdita.png")







'''
