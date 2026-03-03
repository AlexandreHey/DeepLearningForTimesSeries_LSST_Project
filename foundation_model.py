import torch
from momentfm import MOMENTPipeline
from tslearn.datasets import UCR_UEA_datasets
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Préparation des données
print("Chargement et padding des données...")
ds = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

# Padding à 512
X_train_pad = np.pad(X_train, ((0,0), (0, 512-36), (0,0)), mode='constant')
X_test_pad = np.pad(X_test, ((0,0), (0, 512-36), (0,0)), mode='constant')

X_train_torch = torch.from_numpy(X_train_pad).float().transpose(1, 2)
X_test_torch = torch.from_numpy(X_test_pad).float().transpose(1, 2)

le = LabelEncoder()
y_train_idx = torch.tensor(le.fit_transform(y_train))
y_test_idx = torch.tensor(le.transform(y_test))

# 2. Extraction des caractéristiques (On fait ça une seule fois)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
base_model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={"task_name": "reconstruction"}).to(device)
base_model.eval()

def get_embeddings(dataloader):
    all_embeddings = []
    base_model.eval()
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            res = base_model.embed(x_enc=batch_x)
            
            # Gestion robuste de la sortie MOMENT
            embeds = res.embeddings if hasattr(res, 'embeddings') else res
            
            # Si c'est du (Batch, Time, Features), on fait la moyenne sur Time (dim 1)
            if len(embeds.shape) == 3:
                embeds = embeds.mean(dim=1)
            
            all_embeddings.append(embeds.cpu())
    
    final_tensor = torch.cat(all_embeddings, dim=0)
    print(f"DEBUG: Shape des features extraites: {final_tensor.shape}")
    return final_tensor

print("Extraction des features via MOMENT (cela peut prendre 1-2 min)...")
train_loader_pre = DataLoader(TensorDataset(X_train_torch, y_train_idx), batch_size=32)
test_loader_pre = DataLoader(TensorDataset(X_test_torch, y_test_idx), batch_size=32)

train_features = get_embeddings(train_loader_pre)
test_features = get_embeddings(test_loader_pre)

# Sécurité : on récupère la dimension d'entrée dynamiquement
input_dim = train_features.shape[1]
print(f"Entraînement du classifieur sur les features extraites (Dim: {input_dim})...")

# 3. Entraînement du classifieur final
print(f"Entraînement du classifieur sur les features extraites...")
train_ds = TensorDataset(train_features, y_train_idx)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

classifier = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 14)
).to(device)

optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    classifier.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(classifier(batch_x), batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20 terminée")

# 4. Évaluation
classifier.eval()
with torch.no_grad():
    logits = classifier(test_features.to(device))
    preds = logits.argmax(dim=1)
    acc = (preds.cpu() == y_test_idx).float().mean()
    print(f"\n--- RÉSULTATS FINAL FOUNDATION MODEL ---")
    print(f"Accuracy : {acc.item()*100:.2f}%")
    print(f"Baseline était : 26.28%")
    
    
# 1. Calcul des prédictions sur tout le set de test
classifier.eval()
with torch.no_grad():
    logits = classifier(test_features.to(device))
    preds = logits.argmax(dim=1).cpu().numpy()
    true_labels = y_test_idx.numpy()

# 2. Génération du rapport textuel (précision/rappel par classe)
print("\n--- RAPPORT DE CLASSIFICATION DÉTAILLÉ ---")
print(classification_report(true_labels, preds, target_names=[str(c) for c in le.classes_]))

# 3. Création de la matrice de confusion visuelle
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de Confusion : MOMENT sur LSST')
plt.ylabel('Vrais Labels')
plt.xlabel('Prédictions')
plt.savefig('confusion_matrix.png')
print("\nMatrice de confusion sauvegardée sous 'confusion_matrix.png'")
plt.show()