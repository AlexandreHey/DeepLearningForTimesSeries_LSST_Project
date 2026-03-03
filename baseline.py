from tslearn.datasets import UCR_UEA_datasets
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Chargement
print("Chargement des données...")
ds = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

# Note: sktime préfère parfois un format spécifique, 
# mais RocketClassifier accepte souvent le numpy (n, time, dim)

# 2. Définition du modèle Baseline (Rocket est très puissant)
baseline = RocketClassifier(num_kernels=2000) # 2000 kernels pour aller vite

# 3. Entraînement
print("Entraînement de la baseline (Rocket)...")
baseline.fit(X_train, y_train)

# 4. Évaluation
print("Évaluation...")
y_pred = baseline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"--- RÉSULTATS BASELINE ---")
print(f"Accuracy sur le test set : {acc * 100:.2f}%")