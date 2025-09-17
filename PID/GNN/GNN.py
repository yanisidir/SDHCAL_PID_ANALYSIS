import os
import sys
import argparse
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import time
from tqdm import tqdm
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, TAGConv, GraphConv, GraphNorm, global_max_pool, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Argparse setup
parser = argparse.ArgumentParser()
parser.add_argument('--comment', default='', help='Comment for this run')
args = parser.parse_args()

all_dts = []

#####################################
#### 0. Configuration Parameters ####
#####################################
config = {
    'pion_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv",
    'proton_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_proton_E1to100_flat.csv",
    'data_num': '2k',
    'date': datetime.now().strftime("%Y_%m_%d-%H_%M"),
    'nb_features': 8,
    'nb_classes': 2,
    'knn_k': 1,
    'mode': 'route',
    'sigma_gaus': 0,
    'time_window': 2.0,
    'hidden_dim': 64,
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 50,
    'patience': 10,
    'layer_z_bins': 10,
    'seed': 42,
    'seed_gaus': 131,
}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Fix seeds
random.seed(config['seed'])
np.random.seed(config['seed_gaus'])
torch.manual_seed(config['seed'])

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

#####################################
#### 00. Run ID & Logging CSVs #####
#####################################
# Helpers to store run metadata
parameters_file = "run_parameters.csv"
comments_file = "run_comments.csv"
performances_file = "GNN_performances.csv"

# Get next run ID
def get_next_run_id(path):
    if not os.path.exists(path):
        return 1
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return 1 if not rows else max(int(r['run_id']) for r in rows) + 1

run_id = get_next_run_id(parameters_file)

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
    keys = ["date","data_num","nb_features","nb_classes","mode","time_window",
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
### 1. Data loading & splitting #####
#####################################
def load_and_split_data(pion_path, proton_path, test_size=0.2, val_size=0.1):
    # Load and label
    pion = pd.read_csv(pion_path).assign(label=0,
                 event_id=lambda df: df['eventNumber'].astype(str)+"_pion")
    proton = pd.read_csv(proton_path).assign(label=1,
                   event_id=lambda df: df['eventNumber'].astype(str)+"_proton")
    for df in (pion, proton): df.columns = df.columns.str.strip()
    full = pd.concat([pion, proton]).reset_index(drop=True)

    # Apply a smearing 
    full['time'] += np.random.normal(loc=0.0, scale=config['sigma_gaus'], size=full['time'].shape)

    # Truncate the value to 0.01 of precision ( bins of 10ps )
    full['time'] = (full['time'] * 100).astype(int) / 100

    labels = full.groupby('event_id')['label'].first()
    ev_ids, ev_labels = labels.index.values, labels.values

    train_ev, test_ev = train_test_split(ev_ids, test_size=test_size,
                                         stratify=ev_labels, random_state=config['seed'])
    train_ev, val_ev = train_test_split(train_ev,
                    test_size=val_size/(1-test_size),
                    stratify=labels.loc[train_ev], random_state=config['seed'])

    return {
        'train': full[ full['event_id'].isin(train_ev) ],
        'val': full[ full['event_id'].isin(val_ev) ],
        'test': full[ full['event_id'].isin(test_ev) ]
    }

#####################################
#### 2. Preprocessor ###############
#####################################
class GraphPreprocessor:
    def __init__(self):
        self.spatial_scaler = None
        self.time_scaler = None
        self.layer_scalers = {}
        self.z_bins = None

    # def time_smearing(self, graphs, sigma):
    #     for g in graphs:
    #         g['x'][:,3] += np.random.normal(0, sigma, size=g['x'][:,3].shape)

    def fit(self, graphs):
        """Apprend les paramètres de normalisation (spatiale, temporelle, et z par couche) à partir des graphes"""
        
        # 1. Collecte des données
        coords_list = []    # Pour x, y, z
        time_list = []      # Pour time
        z_list = []         # Pour z séparément (pour normalisation par couche)
        
        for g in graphs:
            coords_list.append(g['x'][:, :3])        # x, y, z
            time_list.append(g['x'][:, 3])           # time
            z_list.append(g['z_coords'])             # uniquement z

        # 2. Normalisation des coordonnées spatiales (x, y, z)
        coords_array = np.vstack(coords_list)
        self.spatial_scaler = RobustScaler().fit(coords_array)

        # 3. Normalisation du temps (distribution normale)
        time_array = np.hstack(time_list).reshape(-1, 1)
        self.time_scaler = QuantileTransformer(output_distribution='normal').fit(time_array)

        # 4. Normalisation par couche en z
        z_array = np.hstack(z_list)
        self.z_bins = np.linspace(z_array.min(), z_array.max(), config['layer_z_bins'] + 1)

        for bin_idx in range(config['layer_z_bins']):
            # masque pour sélectionner les z dans la couche i
            mask = (z_array >= self.z_bins[bin_idx]) & (z_array < self.z_bins[bin_idx + 1])
            if mask.sum() > 0:
                self.layer_scalers[bin_idx] = RobustScaler().fit(z_array[mask].reshape(-1, 1))



    def transform(self, graph):
        # 1. Normalisation des coordonnées spatiales x, y, z
        spatial = self.spatial_scaler.transform(graph['x'][:, :3])
        
        # 2. Normalisation du temps
        time_n = self.time_scaler.transform(graph['x'][:, 3].reshape(-1, 1))

        # 3. Normalisation de z par couche (z = profondeur)
        z = graph['z_coords'].copy()  # z est donné séparément
        for bin_idx in range(config['layer_z_bins']):
            mask = (z >= self.z_bins[bin_idx]) & (z < self.z_bins[bin_idx + 1])
            if mask.sum() > 0 and bin_idx in self.layer_scalers:
                z[mask] = self.layer_scalers[bin_idx].transform(z[mask].reshape(-1, 1)).flatten()
        
        # 4. Encodage one-hot du seuil thr
        thr = graph['x'][:, -1].astype(int)
        thr_onehot = np.eye(3)[thr - 1]

        # 5. Concaténation finale : [x, y, z (spatial)] + [time] + [z (normalisé par couche)] + [thr_onehot]
        return np.hstack([spatial, time_n, z.reshape(-1, 1), thr_onehot]).astype(np.float32)


#####################################
#### 3. Graph building #############
#####################################
def knn_z(features, k, mode, device):
    coords = features[:,:3].to(device)
    if mode=='route': z = coords[:,2]
    else:           z = features[:,3].to(device)
    # mask and pairwise distances
    mask = z.view(-1,1)>z.view(1,-1)
    if mode=='route': diff = coords.unsqueeze(1)-coords.unsqueeze(0)
    else:            diff = z.unsqueeze(1)-z.unsqueeze(0)
    dist2 = (diff**2).sum(-1) if mode=='route' else diff**2
    inf = torch.tensor(float('inf'), device=device)
    dist2 = torch.where(mask, dist2, inf)
    # # top-k neighbors
    vals, idxs = torch.topk(dist2, k, largest=False)
    # vals, idxs = torch.min(dist2, dim=1, keepdim=True) 
    src, dst = [], []
    for i,row in enumerate(idxs):
        for j in row.tolist():
            if dist2[i,j]!=float('inf'):
                src.append(i); dst.append(j)
    return torch.stack([torch.tensor(src), torch.tensor(dst)]).to(device)

# def knn_z(features, k, mode, device):
#     coords = features[:, :3].to(device)
#     z = coords[:, 2] if mode == 'route' else features[:, 3].to(device)

#     # Créer le masque causal
#     mask = z.view(-1, 1) > z.view(1, -1)

#     # Calcul des distances
#     if mode == 'route':
#         diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 3]
#         dist2 = (diff ** 2).sum(-1)
#     else:
#         diff = z.unsqueeze(1) - z.unsqueeze(0)
#         dist2 = diff ** 2

#     # Masquer les distances invalides
#     inf = torch.tensor(float('inf'), device=device)
#     dist2 = torch.where(mask, dist2, inf)

#     # Trouver le plus proche voisin valide (k=1)
#     min_dist, min_indices = torch.min(dist2, dim=1)
#     valid_mask = min_dist != float('inf')

#     # Construire les arêtes valides
#     sources = torch.where(valid_mask)[0]
#     targets = min_indices[valid_mask]

#     edge_index = torch.stack([sources, targets])
#     return edge_index.to(device)


# def build_graph(feat, raw_time, k, mode, time_window):
#     edge_idx = knn_z(torch.tensor(feat), k, mode, device)
#     t = torch.tensor(raw_time, device=device)
#     dt = (t[edge_idx[0]] - t[edge_idx[1]]).abs()
#     mask = dt < time_window
#     return edge_idx[:, mask.cpu()]

def build_graph(feat, raw_time, k, mode, time_window):
    # 1. kNN causal
    edge_idx = knn_z(torch.tensor(feat), k, mode, device)

    # 2. (optionnel) filtre temporel :
    t = torch.tensor(raw_time, device=device)
    dt = (t[edge_idx[0]] - t[edge_idx[1]]).abs()

    # on stocke tous les dt
    all_dts.extend(dt.tolist())

    mask = dt < time_window
    edge_idx = edge_idx[:, mask.cpu()]

    # 3. transformer en graphe non‑orienté + self‑loops
    edge_idx = to_undirected(edge_idx, num_nodes=feat.shape[0])
    edge_idx, _ = add_self_loops(edge_idx, num_nodes=feat.shape[0])
    return edge_idx
#####################################
#### 4. PyG data creation ##########
#####################################
def create_pyg_data(df, preproc, is_train):
    raw_graphs = []
    for ev, g in df.groupby('event_id'):
        coords = g[['x_coords','y_coords','z_coords']].values
        times = g['time'].values.reshape(-1,1)
        thr   = g['thr'].values
        raw_graphs.append({'x': np.hstack([coords, times, thr.reshape(-1,1)]).astype(np.float32),
                           'y': int(g['label'].iloc[0]),
                           'z_coords': coords[:,2]})
        
    # if is_train: preproc.time_smearing(raw_graphs, config['sigma_gaus'])
    if is_train: preproc.fit(raw_graphs)

    processed_graphs = []
    for raw in tqdm(raw_graphs, desc=('train' if is_train else 'eval')):
        feat = preproc.transform(raw)
        ei = build_graph(feat, raw['x'][:,3], config['knn_k'], config['mode'], config['time_window'])
        data = Data(x=torch.tensor(feat), edge_index=ei,
                    y=torch.tensor([raw['y']], dtype=torch.long), num_nodes=feat.shape[0])
        processed_graphs.append(data)
    return processed_graphs

from torch_geometric.utils import to_undirected, add_self_loops

def audit_graphs(graphs, name):
    for i, g in enumerate(graphs[:5]):  # premiers 5 events
        E = g.edge_index.size(1)
        N = g.num_nodes
        # nombres de nœuds sources et cibles connectés
        conn = torch.unique(g.edge_index).numel()
        iso = N - conn
        print(f"[{name}] graph #{i}: nodes={N}, edges={E}, isolated={iso}")

#####################################
#### 5. GNN model ##################
#####################################
class ParticleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        h = config['hidden_dim']
        self.enc  = nn.Sequential(nn.Linear(config['nb_features'],h), nn.GELU())
        self.c1   = GATConv(h,h//4,heads=4)
        self.c2   = TAGConv(h,h)
        self.c3   = GraphConv(h,h)
        self.n1   = GraphNorm(h)
        self.n2   = GraphNorm(h)
        self.cls  = nn.Sequential(nn.Linear(h*2,h), nn.Dropout(0.3), nn.GELU(), nn.Linear(h,config['nb_classes']))

    def forward(self, data):
        x, ei, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        x = self.enc(x)
        x = self.n1(self.c1(x,ei), batch).relu()
        x = self.n2(self.c2(x,ei), batch).relu()
        x = self.c3(x,ei).relu()
        xm = global_max_pool(x,batch)
        xM = global_mean_pool(x,batch)
        out = self.cls(torch.cat([xm,xM],dim=1))
        return out.log_softmax(dim=-1)

#####################################
#### 6. Train & eval utils #########
#####################################
def train_one_epoch(model, loader, opt):
    model.train(); tot=0
    for data in loader:
        opt.zero_grad()
        out = model(data)
        loss = nn.functional.nll_loss(out, data.y.to(device))
        loss.backward(); opt.step()
        tot += loss.item()*data.num_graphs
    return tot/len(loader.dataset)

def evaluate(model, loader, thresh=0.5):
    model.eval(); preds, labs=[],[]; tot=0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = nn.functional.nll_loss(out, data.y.to(device))
            tot += loss.item()*data.num_graphs
            prob = out.exp()[:,1].cpu().numpy()
            preds.extend(prob); labs.extend(data.y.numpy())
    loss = tot/len(loader.dataset)
    cls = (np.array(preds)>=thresh).astype(int)
    acc = accuracy_score(labs,cls)
    auc = roc_auc_score(labs,preds)
    return loss, acc, auc, preds, labs

#####################################
#### 7. Main pipeline ##############
#####################################
def main():
    save_run_parameters()
    save_run_comment(args.comment)

    data_dict = load_and_split_data(config['pion_path'], config['proton_path'])
    preproc = GraphPreprocessor()
    train_ds = create_pyg_data(data_dict['train'], preproc, True)
    audit_graphs(train_ds, "train")
    val_ds   = create_pyg_data(data_dict['val'],   preproc, False)
    test_ds  = create_pyg_data(data_dict['test'],  preproc, False)

    import numpy as np

    # Affichage des stats de dt collectés sur l'entraînement
    dts = np.array(all_dts)
    print("=== Stats de dt (en même unité que raw_time) ===")
    for p in [0, 10, 25, 50, 75, 90, 100]:
        print(f"  percentile {p:>3}% : {np.percentile(dts, p):.3f}")
    print(f"  mean = {dts.mean():.3f},  std = {dts.std():.3f}")

    # Optionnel : tracer un histogramme
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(dts, bins=100, density=True, alpha=0.7)
    plt.xlabel("dt")
    plt.ylabel("densité")
    plt.title("Distribution de dt sur l'ensemble d'entraînement")
    plt.show()

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'])
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'])

    model = ParticleGNN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)

    best_auc, counter = 0, 0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}

    t0 = time.time()
    for epoch in range(1, config['epochs']+1):
        tl = train_one_epoch(model, train_loader, opt)
        vl, va, va_auc, _, _ = evaluate(model, val_loader)
        sched.step(vl)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        logging.info(f"Epoch {epoch}/{config['epochs']} TL={tl:.4f} VL={vl:.4f} VA={va:.4f} AUC={va_auc:.4f}")

        if va_auc>best_auc:
            best_auc=va_auc; counter=0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter+=1
            if counter>=config['patience']:
                logging.info("Early stopping")
                break

    # Test set evaluation
    model.load_state_dict(torch.load(best_model_path))
    tl, ta, tauc, tpreds, tlabs = evaluate(model, test_loader)
    logging.info(f"Test L={tl:.4f} Acc={ta:.4f} AUC={tauc:.4f}")

    # Save performances
    exists = os.path.isfile(performances_file)
    with open(performances_file,'a',newline='') as f:
        w = csv.writer(f)
        if not exists: w.writerow(["run_id","test_loss","test_acc","test_auc"])
        w.writerow([run_id, tl, ta, tauc])

    # Plots
    epochs = range(1,len(history['train_loss'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, history['val_acc'],    label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    plt.tight_layout(); plt.savefig(plot1_path); plt.close()

    # Probability hist + confusion
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.hist([p for p,l in zip(tpreds,tlabs) if l==0], bins=50, alpha=0.5, label='Pion')
    plt.hist([p for p,l in zip(tpreds,tlabs) if l==1], bins=50, alpha=0.5, label='Proton')
    plt.xlabel('Proton Prob'); plt.legend(); plt.title('Prob Distribution')
    plt.subplot(122)
    cm = confusion_matrix(tlabs, np.round(tpreds))
    cmn = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    plt.imshow(cmn, interpolation='nearest'); plt.title('Confusion')
    plt.colorbar();
    for i in range(2):
        for j in range(2):
            plt.text(j,i,f"{cmn[i,j]*100:.1f}%", ha='center', va='center',
                     color='white' if cmn[i,j]>0.5 else 'black')
    plt.xticks([0,1],['Pion','Proton']); plt.yticks([0,1],['Pion','Proton'])
    plt.tight_layout(); plt.savefig(plot2_path); plt.close()

if __name__ == '__main__':
    main()