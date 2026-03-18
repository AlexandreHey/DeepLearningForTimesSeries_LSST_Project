import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from mantis.architecture import Mantis8M
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🚀 Running on: {device}")


ds = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ds.load_dataset("LSST")
mean_per_channel = X_train.mean(axis=(0, 1), keepdims=True)
std_per_channel = X_train.std(axis=(0, 1), keepdims=True)

X_train = (X_train - mean_per_channel) / (std_per_channel + 1e-8)
X_test = (X_test - mean_per_channel) / (std_per_channel + 1e-8)

model_save_path = "mantis_6ch_smooth_final_lr04.pth"


class MantisMultiChannel(nn.Module):
    def __init__(self, target_length=512, n_channels=6):
        super().__init__()
        self.mantis = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        self.target_length = target_length
        self.n_channels = n_channels
        
        for param in self.mantis.parameters(): param.requires_grad = False
        for param in list(self.mantis.parameters())[-10:]: param.requires_grad = True

        self.reconstructor = nn.Linear(1536, target_length * n_channels) 

    def forward(self, x, mask_ratio=0.2, return_512=True):
        B, T_orig, C = x.shape  
        x_masked = x.clone()

        if self.training and mask_ratio > 0:
            mask = torch.rand(B, T_orig, C).to(device) < mask_ratio
            x_masked[mask] = 0.0 

        x_v = x_masked.permute(0, 2, 1).reshape(B * C, 1, T_orig)
        x_resized = F.interpolate(x_v, size=self.target_length, mode='linear', align_corners=True)
        
        out = self.mantis(x_resized)
        out_pool = (out.mean(dim=1) + out.max(dim=1)[0]) if out.ndim == 3 else out
        out_f = out_pool.reshape(B, C, -1).reshape(B, -1)
        preds_512 = self.reconstructor(out_f).reshape(B, self.n_channels, self.target_length)
        
        if return_512:
            return preds_512
        else:
            return F.interpolate(preds_512, size=T_orig, mode='linear', align_corners=True)

model_pt = MantisMultiChannel().to(device)

if not os.path.exists(model_save_path):
    loader_pt = DataLoader(torch.tensor(X_train, dtype=torch.float32), batch_size=64, shuffle=True)
    
    opt_pt = optim.Adam([
        {'params': model_pt.mantis.parameters(), 'lr': 1e-6},
        {'params': model_pt.reconstructor.parameters(), 'lr': 1e-4}
    ])
    
    print("\n--- 🧠 Phase 1 : Masked Modeling ---")
    for epoch in range(400):
        model_pt.train()
        epoch_loss = 0
        for batch_x in loader_pt:
            batch_x = batch_x.to(device) # [B, 36, 6]
            opt_pt.zero_grad()
            target = batch_x.permute(0, 2, 1) 
            
            pred_36 = model_pt(batch_x, mask_ratio=0.2, return_512=False)
            
            loss = F.mse_loss(pred_36, target)
            
            loss.backward()
            opt_pt.step()
            epoch_loss += loss.item()

        if (epoch+1)%10 == 0: 
            print(f"Epoch {epoch+1}/400 - Loss: {epoch_loss/len(loader_pt):.5f}")
            
    torch.save(model_pt.state_dict(), model_save_path)
else:
    model_pt.load_state_dict(torch.load(model_save_path, map_location=device))


def extract_features(model, data):
    model.eval()
    all_f = []
    with torch.no_grad():
        for i in range(0, len(data), 64):
            bx = torch.tensor(data[i:i+64], dtype=torch.float32).to(device)
            B, T, C = bx.shape
            xv = bx.permute(0, 2, 1).reshape(B * C, 1, T)
            xr = F.interpolate(xv, size=512, mode='linear', align_corners=True)
            out = model.mantis(xr)
            res = (out.mean(dim=1) + out.max(dim=1)[0]) if out.ndim == 3 else out
            all_f.append(res.reshape(B, C, -1).reshape(B, -1).cpu().numpy())
    return np.concatenate(all_f)

X_train_f = extract_features(model_pt, X_train)
X_test_f = extract_features(model_pt, X_test)

le = LabelEncoder()
y_train_idx = le.fit_transform(y_train)
y_test_idx = le.transform(y_test)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_f, y_train_idx)

train_loader = DataLoader(TensorDataset(torch.tensor(X_res, dtype=torch.float32), torch.tensor(y_res, dtype=torch.long)), batch_size=128, shuffle=True)

classifier = nn.Sequential(
    nn.Linear(X_train_f.shape[1], 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(le.classes_))
).to(device)

opt_cls = optim.Adam(classifier.parameters(), lr=1e-3)
for epoch in range(50):
    classifier.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        opt_cls.zero_grad()
        F.cross_entropy(classifier(bx), by).backward()
        opt_cls.step()

classifier.eval()
with torch.no_grad():
    test_preds = classifier(torch.tensor(X_test_f, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()

print(f"\n✅ ACCURACY FINALE : {accuracy_score(y_test_idx, test_preds)*100:.2f}%")
target_names = [str(cls) for cls in le.classes_]
print(classification_report(y_test_idx, test_preds, target_names=target_names))
