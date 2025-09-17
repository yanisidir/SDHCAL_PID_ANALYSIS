import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

#####################################
#### 0. Configuration Parameters ####
#####################################

config = {
    'pion_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv",
    'proton_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_proton_E1to100_flat.csv",
    'data_num': '2k',
    'date'        : datetime.now().strftime("%Y_%m_%d-%H_%M"),
    'nb_features' : 7,      # thr count as 3 one-hot
    'nb_classes'  : 2,
    'sigma_gaus'  : 0.1,     # ns   
    'seed_gaus'   : 131,
    'hidden_dim'  : 256,
    'batch_size'  : 64,
    'lr'          : 1e-3,
    'epochs'      : 50,
    'patience'    : 10,
    'seed'        : 42,
}

# -- ID et logging de config (inchangé)

parameters_file   = "run_parameters.csv"
comments_file     = "run_comments.csv"
performances_file = "CNN_performances.csv"
run_id = 1  # adapter avec votre fonction get_next_run_id, inchangé

# Define output paths
best_model_path = f"models/best_model_{run_id}.pt"
plot1_path      = f"plots/loss_acc_{run_id}.png"
plot2_path      = f"plots/confusion_{run_id}.png"
# Ensure output directories exist
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
os.makedirs(os.path.dirname(plot1_path), exist_ok=True)
os.makedirs(os.path.dirname(plot2_path), exist_ok=True)

# Save parameters
def save_run_parameters():
    keys = ["date","data_num","nb_features","nb_classes","mode",
            "hidden_dim","batch_size","lr","epochs","seed","patience"]
    row = {"run_id": run_id}
    row.update({k: config[k] for k in keys})
    exists = os.path.exists(parameters_file)
    with open(parameters_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["run_id"]+keys)
        if not exists: writer.writeheader()
        writer.writerow(row)

# Save comment
def save_run_comment(comment_text):
    exists = os.path.isfile(comments_file)
    with open(comments_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists: writer.writerow(["run_id","comment"])
        writer.writerow([run_id, comment_text])

#####################################
### 1. Data loading and splitting ###
#####################################

def load_and_split_data(pion_path, proton_path, test_size=0.2, val_size=0.1):
    pion_df = pd.read_csv(pion_path).assign(label=0, event_id=lambda x: x['eventNumber'].astype(str)+'_pion')
    proton_df = pd.read_csv(proton_path).assign(label=1, event_id=lambda x: x['eventNumber'].astype(str)+'_proton')
    for df in (pion_df, proton_df):
        df.columns = df.columns.str.strip()
    full_df = pd.concat([pion_df, proton_df]).reset_index(drop=True)
    # smearing time
    # np.random.seed(config['seed_gaus'])
    # full_df['time'] += np.random.normal(0.0, config['sigma_gaus'], size=full_df.shape[0])
    full_df['time'] = (full_df['time'] * 100).astype(int) / 100

    event_labels = full_df.groupby('event_id')['label'].first()
    ids    = event_labels.index.values
    labels = event_labels.values

    train_ids, test_ids = train_test_split(ids, test_size=test_size,
                              stratify=labels, random_state=config['seed'])
    train_ids, val_ids  = train_test_split(train_ids,
                              test_size=val_size/(1-test_size),
                              stratify=event_labels.loc[train_ids],
                              random_state=config['seed'])

    return {
      'train': full_df[full_df['event_id'].isin(train_ids)],
      'val'  : full_df[full_df['event_id'].isin(val_ids)],
      'test' : full_df[full_df['event_id'].isin(test_ids)],
    }

#####################################
########## 2. Preprocessor ##########
#####################################

class DataPreprocessor:
    def __init__(self):
        self.spatial_scaler = None
        self.time_scaler    = None

    def fit(self, graphs):
        coords = np.vstack([g['x'][:,:3] for g in graphs])
        times  = np.hstack([g['x'][:,3]  for g in graphs]).reshape(-1,1)
        self.spatial_scaler = RobustScaler().fit(coords)
        self.time_scaler    = QuantileTransformer(output_distribution='normal').fit(times)

    def time_smearing(self, graphs):
        for g in graphs:
            g['x'][:,3] += np.random.normal(0.0, config['sigma_gaus'], size=g['x'][:,3].shape)

    def transform(self, graph):
        spatial = self.spatial_scaler.transform(graph['x'][:,:3])
        time_   = self.time_scaler.transform(graph['x'][:,3].reshape(-1,1))
        thr     = graph['x'][:,-1].astype(int)
        thr_onehot = np.eye(3)[thr-1]
        return np.hstack([spatial, time_, thr_onehot]).astype(np.float32)

#####################################
########## 3. Dataset CNN ###########
#####################################

class ParticleDataset(Dataset):
    def __init__(self, df, preprocessor, is_train=False):
        self.samples = []
        # construire la liste de raw_graphs
        raw_graphs = []
        for eid, grp in df.groupby('event_id'):
            coords = grp[['x_coords','y_coords','z_coords']].values
            times  = grp['time'].values.reshape(-1,1)
            thr    = grp['thr'].values
            raw_graphs.append({
                'x': np.hstack([coords, times, thr.reshape(-1,1)]).astype(np.float32),
                'y': int(grp['label'].iloc[0])
            })
        if is_train:
            preprocessor.time_smearing(raw_graphs)
            preprocessor.fit(raw_graphs)
        # application du transform
        for raw in raw_graphs:
            feat = preprocessor.transform(raw)  # shape [n_hits, nb_features]
            self.samples.append((feat, raw['y']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """Padder la dimension nombre de hits au max du batch"""
    feats, labels = zip(*batch)
    lengths = [f.shape[0] for f in feats]
    max_len = max(lengths)
    C = feats[0].shape[1]
    X = torch.zeros(len(batch), C, max_len, dtype=torch.float32)
    for i, f in enumerate(feats):
        # f: [n_hits, C] -> transpose -> [C, n_hits]
        fh = torch.from_numpy(f.T)
        X[i, :, :fh.shape[1]] = fh
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

#####################################
########### 4. CNN model ############
#####################################

class ParticleCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, nb_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2  = nn.GELU()
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(hidden_dim, nb_classes)
        )

    def forward(self, x):
        # x: [batch, C, L]
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [batch, hidden_dim]
        return self.classifier(x)

#####################################
######### 5. Training utils #########
#####################################

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, device, threshold=0.5):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_probs = []
    all_labels= []
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * X.size(0)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
    val_loss = total_loss / len(loader.dataset)
    preds_class = (np.array(all_probs) >= threshold).astype(int)
    acc = accuracy_score(all_labels, preds_class)
    auc = roc_auc_score(all_labels, all_probs)
    return val_loss, acc, auc, all_probs, all_labels

#####################################
######### 6. Full Pipeline ##########
#####################################

def main():
    t0 = time.time()
    raw = load_and_split_data(config['pion_path'], config['proton_path'])
    preproc = DataPreprocessor()

    train_ds = ParticleDataset(raw['train'], preproc, is_train=True)
    val_ds   = ParticleDataset(raw['val'],   preproc, is_train=False)
    test_ds  = ParticleDataset(raw['test'],  preproc, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                              shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleCNN(config['nb_features'], config['hidden_dim'], config['nb_classes']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    best_val_auc = 0
    counter = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(config['epochs']):
        tl = train_one_epoch(model, train_loader, optimizer, device)
        vl, va, vauc, _, _ = validate(model, val_loader, device)
        scheduler.step(vl)

        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)

        print(f"Epoch {epoch+1}/{config['epochs']} "
              f"Train Loss: {tl:.4f} | Val Loss: {vl:.4f} "
              f"| Val Acc: {va:.4f} | Val AUC: {vauc:.4f}")

        if vauc > best_val_auc:
            best_val_auc = vauc
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            if counter >= config['patience']:
                print("Early stopping.")
                break

    # Test final
    model.load_state_dict(torch.load(best_model_path))
    tloss, tacc, tauc, tprobs, tlabels = validate(model, test_loader, device)
    print(f"Test Loss: {tloss:.4f} | Test Acc: {tacc:.4f} | Test AUC: {tauc:.4f}")

    # Sauvegarde performances et plots (même code que précédemment)...

if __name__ == '__main__':
    main()