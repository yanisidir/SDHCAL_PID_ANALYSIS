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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

# Argparse setup
parser = argparse.ArgumentParser()
parser.add_argument('--comment', default='', help='Comment for this run')
args = parser.parse_args()

#####################################
#### 0. Configuration Parameters ####
#####################################
config = {
    'pion_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv",
    'proton_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_proton_E1to100_flat.csv",
    'kaon_path':   "/gridgroup/ilc/midir/Timing/files/csv/1k_kaon_E1to100_flat.csv",
    'data_num': '20k',
    'date': datetime.now().strftime("%Y_%m_%d-%H_%M"),
    'nb_features': 7,
    'nb_classes': 3,
    'knn_k': 8,
    'mode': 'route',
    'sigma_gaus': 0.1,
    'seed_gaus': 131,
    'time_window': 2.0,
    'hidden_dim': 64,
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 50,
    'patience': 10,
    'seed': 42,
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
def load_and_split_data(pion_path, proton_path, kaon_path,
                        test_size=0.2, val_size=0.1, random_state=42):
    # 1) Lecture + label + event_id
    pion = (pd.read_csv(pion_path)
              .dropna()
              .assign(label=0,
                      event_id=lambda df: df['eventNumber'].astype(str)+"_pion"))
                
    proton = (pd.read_csv(proton_path)
                .dropna()
                .assign(label=1,
                        event_id=lambda df: df['eventNumber'].astype(str)+"_proton"))

    kaon = (pd.read_csv(kaon_path)
              .dropna()  
              .assign(label=2,
                      event_id=lambda df: df['eventNumber'].astype(str)+"_kaon"))

    for df in (pion, proton, kaon):
        df.columns = df.columns.str.strip()
    full = pd.concat([pion, proton, kaon], ignore_index=True)

    full['time'] = (full['time'] * 100).astype(int) / 100

    # 1) Préparer df_events
    df_events = full[['event_id','label']].drop_duplicates().reset_index(drop=True)

    # 2) train/test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(df_events, df_events['label']))
    train_events = df_events.loc[train_idx, 'event_id'].values
    test_events  = df_events.loc[test_idx,  'event_id'].values

    # 3) train/val
    train_sub = df_events.loc[train_idx].reset_index(drop=True)   # on réindexe de 0 à N_train-1
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size/(1-test_size),
        random_state=random_state
    )
    sub_idx, val_idx = next(sss_val.split(train_sub, train_sub['label']))

    train_events = train_sub.iloc[sub_idx]['event_id'].values
    val_events   = train_sub.iloc[val_idx]['event_id'].values

    # 4) retourner les DataFrames
    return {
        'train': full[ full['event_id'].isin(train_events) ],
        'val':   full[ full['event_id'].isin(val_events) ],
        'test':  full[ full['event_id'].isin(test_events) ]
    }

#####################################
#### 2. Preprocessor ###############
#####################################
class GraphPreprocessor:
    def __init__(self):
        self.spatial_scaler = None
        self.time_scaler = None

    # def time_smearing(self, graphs, sigma):
    #     for g in graphs:
    #         g['x'][:,3] += np.random.normal(0, sigma, size=g['x'][:,3].shape)

    def fit(self, graphs):
        coords = np.vstack([g['x'][:,:3] for g in graphs])
        times = np.hstack([g['x'][:,3] for g in graphs]).reshape(-1,1)
        self.spatial_scaler = RobustScaler().fit(coords)
        self.time_scaler = QuantileTransformer(output_distribution='normal').fit(times)

    def transform(self, graph):
        spatial = self.spatial_scaler.transform(graph['x'][:,:3])
        time_n = self.time_scaler.transform(graph['x'][:,3].reshape(-1,1))
        thr = graph['x'][:,-1].astype(int)
        thr_onehot = np.eye(3)[thr-1]
        return np.hstack([spatial, time_n, thr_onehot]).astype(np.float32)

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
    # top-k neighbors
    vals, idxs = torch.topk(dist2, k, largest=False)
    src, dst = [], []
    for i,row in enumerate(idxs):
        for j in row.tolist():
            if dist2[i,j]!=float('inf'):
                src.append(i); dst.append(j)
    return torch.stack([torch.tensor(src), torch.tensor(dst)])


def build_graph(feat, raw_time, k, mode, time_window):
    edge_idx = knn_z(torch.tensor(feat), k, mode, device)
    t = torch.tensor(raw_time, device=device)
    dt = (t[edge_idx[0]] - t[edge_idx[1]]).abs()
    mask = dt < time_window
    return edge_idx[:, mask.cpu()]

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

def evaluate(model, loader):
    model.eval()
    tot_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            out = model(data)  # log-softmax sur 3 classes
            loss = nn.functional.nll_loss(out, data.y.to(device))
            tot_loss += loss.item() * data.num_graphs

            # argmax pour la prédiction
            preds = out.exp().argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.numpy())

    avg_loss = tot_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, cm, all_preds, all_labels

#####################################
#### 7. Main pipeline ##############
#####################################
def main():
    save_run_parameters()
    save_run_comment(args.comment)

    data_dict = load_and_split_data(
        config['pion_path'],
        config['proton_path'],
        config['kaon_path'],
        test_size=0.2,       # ou ce que vous voulez
        val_size=0.1,
        random_state=config['seed']
    )

    for split in ['train','val','test']:
        counts = data_dict[split]['label'].value_counts().sort_index()
        print(f"{split} set :")
        print(counts)
        print("---")

    # 2) Préparation des graphes
    preproc      = GraphPreprocessor()
    train_ds     = create_pyg_data(data_dict['train'], preproc, True)
    val_ds       = create_pyg_data(data_dict['val'],   preproc, False)
    test_ds      = create_pyg_data(data_dict['test'],  preproc, False)
    # sys.exit(0) 

    preproc = GraphPreprocessor()
    train_ds = create_pyg_data(data_dict['train'], preproc, True)
    val_ds   = create_pyg_data(data_dict['val'],   preproc, False)
    test_ds  = create_pyg_data(data_dict['test'],  preproc, False)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'])
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'])

    model = ParticleGNN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)

    best_acc, counter = 0, 0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}

    t0 = time.time()
    for epoch in range(1, config['epochs']+1):
        tl = train_one_epoch(model, train_loader, opt)
        vl, va, cm_val, _, _ = evaluate(model, val_loader)
        sched.step(vl)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        logging.info(f"Epoch {epoch}/{config['epochs']} TL={tl:.4f} VL={vl:.4f} VA={va:.4f}")

        if va>best_acc:
            best_acc=va; counter=0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter+=1
            if counter>=config['patience']:
                logging.info("Early stopping")
                break

    # Test set evaluation
    model.load_state_dict(torch.load(best_model_path))
    tl, ta, cm_test, tpreds, tlabs = evaluate(model, test_loader)
    logging.info(f"Test L={tl:.4f} Acc={ta:.4f}")

    # Save performances
    exists = os.path.isfile(performances_file)
    with open(performances_file,'a',newline='') as f:
        w = csv.writer(f)
        if not exists: w.writerow(["run_id","test_loss","test_acc"])
        w.writerow([run_id, tl, ta])

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

    # Confusion matrix plot
    classes = ['Pion','Proton','Kaon']
    plt.figure(figsize=(6,5))
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix – Test')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{cm_test[i,j]}",
                     ha='center', va='center',
                     color='white' if cm_test[i,j] > cm_test.max()/2 else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(plot2_path)
    plt.close()

if __name__ == '__main__':
    main()