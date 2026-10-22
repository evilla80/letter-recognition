import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Caricamento del dataset
# Il file è disponibile all'URL del repository UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
# Il dataset ha 17 colonne: la prima è la lettera target, le altre 16 sono le features
col_names = ['letter'] + [f'feature_{i}' for i in range(1, 17)]
data = pd.read_csv(url, header=None, names=col_names)
print("Prime righe del dataset:")
print(data.head())

# 2. Separazione delle features e della variabile target
X = data.iloc[:, 1:]   # Le colonne delle features
y = data.iloc[:, 0]    # La lettera (target)

# 3. Standardizzazione delle features
# La PCA è influenzata dalle scale delle variabili, quindi è buona pratica standardizzare i dati
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 4. Applicazione della PCA
# Usiamo tutte le 16 componenti per poter vedere la varianza spiegata da ciascuna
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X_std)

# 5. Varianza spiegata ed autovalori
# pca.explained_variance_ratio_ restituisce la frazione della varianza spiegata da ciascuna componente
varianza_spiegata = pca.explained_variance_ratio_
# pca.explained_variance_ restituisce gli autovalori (la quantità di varianza catturata)
autovalori = pca.explained_variance_

print("\nVarianza spiegata per ciascuna componente:")
print(varianza_spiegata)

print("\nAutovalori:")
print(autovalori)

# 6. Plot della varianza cumulativa spiegata (scree plot)
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(varianza_spiegata), marker='o', linestyle='--')
plt.xlabel('Numero di componenti principali')
plt.ylabel('Varianza cumulativa spiegata')
plt.title('PCA - Varianza cumulativa spiegata')
plt.grid(True)
plt.show()

# 7. Visualizzazione dei dati proiettati sui primi due componenti principali
# Per visualizzare le classi, convertiamo la lettera in un numero (A -> 0, B -> 1, ...)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=classi, cmap='viridis', alpha=0.5)
plt.xlabel('Prima componente principale')
plt.ylabel('Seconda componente principale')
plt.title('PCA: Prime due componenti del dataset Letter Recognition')
plt.colorbar(scatter, label='Classe lettera (A=0, B=1, ...)')
plt.grid(True)
plt.show()