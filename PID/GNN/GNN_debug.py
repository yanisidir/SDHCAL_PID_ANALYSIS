#!/usr/bin/env python3
import os, sys, argparse, csv, random, time, logging
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv, TAGConv, GraphConv, GraphNorm,
    global_max_pool, global_mean_pool
)
from torch_geometric.utils import to_undirected, add_self_loops

from torch_geometric.nn import NNConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# -------------------------
#  Argparse & config
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--comment', default='', help='Comment for this run')
parser.add_argument('--debug', action='store_true', help='Activate verbose debug logs')
args = parser.parse_args()

all_dts = []
dbg_hist_dt = []  # pour stocker quelques dt par event
dbg = defaultdict(list)  # sac à dos debug
np.set_printoptions(precision=3, suppress=True)

config = {
    'pion_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv",
    'proton_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_proton_E1to100_flat.csv",
    'data_num': '2k',
    'date': datetime.now().strftime("%Y_%m_%d-%H_%M"),
    'nb_features': 8,
    'nb_classes': 2,
    'knn_k': 8,
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
    'debug': args.debug,
}

# -------------------------
#  Logging
# -------------------------
logging.basicConfig(
    level=logging.DEBUG if config['debug'] else logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------
#  Utils
# -------------------------
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.debug(f"[TIMER] {name}: {time.time()-t0:.3f}s")

def percentiles(x, ps=[0,10,25,50,75,90,95,99,100]):
    x = np.asarray(x)
    return {p: np.percentile(x, p) for p in ps}

def check_tensor(name, t):
    if torch.isnan(t).any() or torch.isinf(t).any():
        logger.error(f"[NaN/Inf] {name} contient NaN/Inf !")
        raise ValueError(f"NaN/Inf in {name}")

def grad_hook(name):
    def hook(grad):
        if grad is None:
            return
        g = grad.detach()
        dbg['grad_norms'].append((name, g.norm().item()))
        if torch.isnan(g).any() or torch.isinf(g).any():
            logger.error(f"[NaN/Inf] Gradient {name} a NaN/Inf !")
            raise ValueError(f"NaN/Inf gradient {name}")
    return hook

# -------------------------
#  Seeding / device
# -------------------------
random.seed(config['seed'])
np.random.seed(config['seed_gaus'])
torch.manual_seed(config['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# -------------------------
#  CSV logs
# -------------------------
parameters_file   = "run_parameters.csv"
comments_file     = "run_comments.csv"
performances_file = "GNN_performances.csv"

def get_next_run_id(path):
    if not os.path.exists(path):
        return 1
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
        return 1 if not rows else max(int(r['run_id']) for r in rows) + 1

run_id = get_next_run_id(parameters_file)

best_model_path = f"models/best_model_{run_id}.pt"
plot1_path      = f"plots/loss_acc_{run_id}.png"
plot2_path      = f"plots/confusion_{run_id}.png"
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
os.makedirs(os.path.dirname(plot1_path), exist_ok=True)
os.makedirs(os.path.dirname(plot2_path), exist_ok=True)

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

def save_run_comment(comment_text):
    exists = os.path.isfile(comments_file)
    with open(comments_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists: writer.writerow(["run_id","comment"])
        writer.writerow([run_id, comment_text])

# -------------------------
#  1. Data
# -------------------------
def load_and_split_data(pion_path, proton_path, test_size=0.2, val_size=0.1):
    with timer("load csv"):
        pion = pd.read_csv(pion_path).assign(label=0,
                 event_id=lambda df: df['eventNumber'].astype(str)+"_pion")
        proton = pd.read_csv(proton_path).assign(label=1,
                 event_id=lambda df: df['eventNumber'].astype(str)+"_proton")
    for df in (pion, proton): df.columns = df.columns.str.strip()
    full = pd.concat([pion, proton]).reset_index(drop=True)
    logger.debug(f"full shape={full.shape}, pion={len(pion)}, proton={len(proton)}")

    with timer("smearing + trunc"):
        full['time'] += np.random.normal(0.0, config['sigma_gaus'], size=full['time'].shape)
        full['time'] = (full['time'] * 100).astype(int) / 100

    labels = full.groupby('event_id')['label'].first()
    ev_ids, ev_labels = labels.index.values, labels.values

    logger.debug(f"class balance before split: {np.bincount(ev_labels)} (0=pion,1=proton)")
    train_ev, test_ev = train_test_split(ev_ids, test_size=test_size,
                                         stratify=ev_labels, random_state=config['seed'])
    train_ev, val_ev = train_test_split(train_ev,
                                        test_size=val_size/(1-test_size),
                                        stratify=labels.loc[train_ev], random_state=config['seed'])
    d = {
        'train': full[ full['event_id'].isin(train_ev) ],
        'val':   full[ full['event_id'].isin(val_ev)   ],
        'test':  full[ full['event_id'].isin(test_ev)  ],
    }
    for k in d:
        c = d[k].groupby('event_id')['label'].first().values
        logger.info(f"[{k}] events={len(c)} | class balance={np.bincount(c)}")
    return d

# -------------------------
#  2. Preprocessor
# -------------------------
# class GraphPreprocessor:
#     def __init__(self):
#         self.spatial_scaler = None
#         self.time_scaler = None
#         self.layer_scalers = {}
#         self.z_bins = None

#     def fit(self, graphs):
#         coords_list, time_list, z_list = [], [], []
#         for g in graphs:
#             coords_list.append(g['x'][:, :3])
#             time_list.append(g['x'][:, 3])
#             z_list.append(g['z_coords'])
#         coords_array = np.vstack(coords_list)
#         time_array   = np.hstack(time_list).reshape(-1,1)
#         z_array      = np.hstack(z_list)

#         self.spatial_scaler = RobustScaler().fit(coords_array)
#         self.time_scaler    = QuantileTransformer(output_distribution='normal').fit(time_array)

#         self.z_bins = np.linspace(z_array.min(), z_array.max(), config['layer_z_bins'] + 1)
#         for bin_idx in range(config['layer_z_bins']):
#             mask = (z_array >= self.z_bins[bin_idx]) & (z_array < self.z_bins[bin_idx + 1])
#             if mask.sum() > 0:
#                 self.layer_scalers[bin_idx] = RobustScaler().fit(z_array[mask].reshape(-1, 1))

#         if config['debug']:
#             logger.debug(f"[Preproc] coords medians={np.median(coords_array,axis=0)}")
#             logger.debug(f"[Preproc] time q=[0,50,100]%={np.percentile(time_array,[0,50,100])}")
#             logger.debug(f"[Preproc] z_bins={self.z_bins}")

#     def transform(self, graph):
#         spatial = self.spatial_scaler.transform(graph['x'][:, :3])
#         time_n  = self.time_scaler.transform(graph['x'][:, 3].reshape(-1, 1))

#         z = graph['z_coords'].copy()
#         for bin_idx in range(config['layer_z_bins']):
#             mask = (z >= self.z_bins[bin_idx]) & (z < self.z_bins[bin_idx + 1])
#             if mask.sum() > 0 and bin_idx in self.layer_scalers:
#                 z[mask] = self.layer_scalers[bin_idx].transform(z[mask].reshape(-1, 1)).flatten()

#         thr = graph['x'][:, -1].astype(int)
#         if thr.min() < 1 or thr.max() > 3:
#             logger.warning(f"[transform] thr outside [1,3]: min={thr.min()}, max={thr.max()}")
#         thr_onehot = np.eye(3)[thr - 1]

#         feat = np.hstack([spatial, time_n, z.reshape(-1, 1), thr_onehot]).astype(np.float32)

#         # Debug sample
#         if config['debug'] and random.random() < 0.001:
#             logger.debug(f"[transform] sample feat[0]: {feat[0]}")
#         return feat

class GraphPreprocessor:
    def __init__(self):
        self.scaler = None
    def fit(self, graphs):
        X = np.vstack([g['x'] for g in graphs])
        self.scaler = RobustScaler().fit(X)
    def transform(self, graph):
        return self.scaler.transform(graph['x']).astype(np.float32)


# -------------------------
#  3. Graph building
# -------------------------
def knn_z(features, k, mode, device):
    coords = features[:,:3].to(device)
    z = coords[:,2] if mode=='route' else features[:,3].to(device)

    mask = z.view(-1,1) > z.view(1,-1)
    diff = coords.unsqueeze(1)-coords.unsqueeze(0) if mode=='route' else (z.unsqueeze(1)-z.unsqueeze(0))
    dist2 = (diff**2).sum(-1) if mode=='route' else diff**2

    inf = torch.tensor(float('inf'), device=device)
    dist2 = torch.where(mask, dist2, inf)

    vals, idxs = torch.topk(dist2, k, largest=False)
    src, dst = [], []
    for i,row in enumerate(idxs):
        for j in row.tolist():
            if dist2[i,j]!=float('inf'):
                src.append(i); dst.append(j)
    ei = torch.stack([torch.tensor(src), torch.tensor(dst)]).to(device)
    return ei

def build_graph(feat, raw_time, k, mode, time_window, keep_dt_samples=False):
    edge_idx = knn_z(torch.tensor(feat), k, mode, device)
    t = torch.tensor(raw_time, device=device)
    dt = (t[edge_idx[0]] - t[edge_idx[1]]).abs()

    # store dt
    all_dts.extend(dt.tolist())
    if keep_dt_samples and len(dbg_hist_dt) < 20:
        dbg_hist_dt.append(dt.cpu().numpy())

    mask = dt < time_window
    edge_idx = edge_idx[:, mask.cpu()]

    edge_idx = to_undirected(edge_idx, num_nodes=feat.shape[0])
    edge_idx, _ = add_self_loops(edge_idx, num_nodes=feat.shape[0])
    return edge_idx

# -------------------------
#  4. PyG data creation
# -------------------------
def create_pyg_data(df, preproc, is_train):
    raw_graphs = []
    for ev, g in df.groupby('event_id'):
        coords = g[['x_coords','y_coords','z_coords']].values
        times  = g['time'].values.reshape(-1,1)
        thr    = g['thr'].values
        raw_graphs.append({
            'x': np.hstack([coords, times, thr.reshape(-1,1)]).astype(np.float32),
            'y': int(g['label'].iloc[0]),
            'z_coords': coords[:,2]
        })

    if is_train: preproc.fit(raw_graphs)

    processed_graphs = []
    for i, raw in enumerate(tqdm(raw_graphs, desc=('train' if is_train else 'eval'))):
        feat = preproc.transform(raw)
        ei = build_graph(feat, raw['x'][:,3], config['knn_k'], config['mode'], config['time_window'],
                         keep_dt_samples=is_train)
        data = Data(x=torch.tensor(feat),
                    edge_index=ei,
                    y=torch.tensor([raw['y']], dtype=torch.long),
                    num_nodes=feat.shape[0])
        processed_graphs.append(data)

        if config['debug'] and i < 3:
            logger.debug(f"[graph dbg] nodes={data.num_nodes}, edges={data.edge_index.size(1)}")
    return processed_graphs

def audit_graphs(graphs, name):
    degs = []
    for i, g in enumerate(graphs[:10]):
        E = g.edge_index.size(1)
        N = g.num_nodes
        conn = torch.unique(g.edge_index).numel()
        iso = N - conn
        degs.append(E/N)
        logger.info(f"[{name}] graph #{i}: nodes={N}, edges={E}, isolated={iso}, edges/node={E/N:.2f}")
    logger.info(f"[{name}] avg edges/node on 10 samples = {np.mean(degs):.2f}")

# -------------------------
#  5. Model
# -------------------------
class ParticleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        h = config['hidden_dim']
        dp = 0.4
        self.enc  = nn.Sequential(nn.Linear(config['nb_features'], h), nn.GELU(), nn.Dropout(dp))
        self.c1   = GATConv(h, h//4, heads=4, dropout=dp)
        self.c2   = TAGConv(h, h)
        self.c3   = GraphConv(h, h)
        self.n1   = GraphNorm(h)
        self.n2   = GraphNorm(h)
        self.drop = nn.Dropout(dp)
        self.cls  = nn.Sequential(nn.Linear(h*2, h), nn.GELU(), nn.Dropout(dp),
                                  nn.Linear(h, config['nb_classes']))

    def forward(self, data):
        x, ei, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        check_tensor("x_in", x)
        x = self.enc(x)
        x = self.n1(self.c1(x, ei), batch).relu()
        x = self.drop(x)
        x = self.n2(self.c2(x, ei), batch).relu()
        x = self.drop(x)
        x = self.c3(x, ei).relu()
        xm = global_max_pool(x, batch)
        xM = global_mean_pool(x, batch)
        out = self.cls(torch.cat([xm, xM], dim=1))
        return out.log_softmax(dim=-1)

# -------------------------
#  6. Train & eval
# -------------------------
def train_one_epoch(model, loader, opt, epoch):
    model.train()
    tot = 0
    for b, data in enumerate(loader):
        opt.zero_grad()
        out = model(data)
        check_tensor("out_train", out)
        loss = nn.functional.nll_loss(out, data.y.to(device))
        loss.backward()

        # gradient debugging
        if config['debug'] and b % 10 == 0:
            for n, p in model.named_parameters():
                if p.grad is not None:
                    dbg['grad_norms_batch'].append((epoch, b, n, p.grad.norm().item()))
        opt.step()
        tot += loss.item()*data.num_graphs

        if config['debug'] and b % 50 == 0:
            logger.debug(f"[train] epoch {epoch} batch {b}/{len(loader)} loss={loss.item():.4f}")
    return tot/len(loader.dataset)

def evaluate(model, loader, thresh=0.5, split_name="val"):
    model.eval()
    preds, labs = [], []
    tot = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = nn.functional.nll_loss(out, data.y.to(device))
            tot += loss.item()*data.num_graphs
            prob = out.exp()[:,1].cpu().numpy()
            preds.extend(prob); labs.extend(data.y.numpy())

    loss = tot/len(loader.dataset)
    preds = np.array(preds); labs = np.array(labs)
    acc = accuracy_score(labs, (preds>=thresh).astype(int))
    auc = roc_auc_score(labs, preds)

    # debug distribution
    if config['debug']:
        logger.debug(f"[{split_name}] prob stats: mean={preds.mean():.3f}, std={preds.std():.3f}, "
                     f"min={preds.min():.3f}, max={preds.max():.3f}")
    return loss, acc, auc, preds, labs

def optimal_threshold(preds, labs):
    # petit grid-search pour info
    ts = np.linspace(0,1,101)
    best_t, best_acc = 0.5, 0
    for t in ts:
        acc = accuracy_score(labs, (preds>=t).astype(int))
        if acc > best_acc:
            best_acc = acc; best_t = t
    return best_t, best_acc

# -------------------------
#  7. Main
# -------------------------
def main():
    save_run_parameters()
    save_run_comment(args.comment)

    data_dict = load_and_split_data(config['pion_path'], config['proton_path'])
    preproc = GraphPreprocessor()

    with timer("create train data"):
        train_ds = create_pyg_data(data_dict['train'], preproc, True)
    audit_graphs(train_ds, "train")

    with timer("create val/test data"):
        val_ds   = create_pyg_data(data_dict['val'],   preproc, False)
        test_ds  = create_pyg_data(data_dict['test'],  preproc, False)

    # # ---- TEST OVERFIT ----
    # small_train = train_ds[:32]  # 32 graphes
    # small_loader = DataLoader(small_train, batch_size=8, shuffle=True)

    # model_small = ParticleGNN().to(device)
    # opt_small = torch.optim.AdamW(model_small.parameters(), lr=1e-3, weight_decay=0)
    # for ep in range(1, 51):
    #     tl = train_one_epoch(model_small, small_loader, opt_small, ep)
    #     vl, va, auc, _, _ = evaluate(model_small, small_loader, split_name="small")
    #     print(f"[OVERFIT] ep={ep} TL={tl:.3f} Acc={va:.3f} AUC={auc:.3f}")
    #     if va > 0.99:
    #         break
    # sys.exit(0)  # stop ici pour ce test

    # dt stats
    dts = np.array(all_dts)
    stats_dt = percentiles(dts)
    logger.info("=== Stats globales de dt ===")
    for k,v in stats_dt.items():
        logger.info(f"  p{k:>2} = {v:.3f}")
    logger.info(f"  mean={dts.mean():.3f}, std={dts.std():.3f}, max={dts.max():.3f}")

    if config['debug']:
        # histogramme rapide
        plt.figure()
        plt.hist(dts, bins=100, density=True, alpha=0.7)
        plt.xlabel("dt"); plt.ylabel("densité"); plt.title("Distribution de dt (train)")
        plt.savefig(f"plots/dt_hist_{run_id}.png"); plt.close()

    def graph_to_stats(d):
        x = d.x.numpy()
        return np.concatenate([x.mean(0), x.std(0), x.max(0), x.min(0), [d.num_nodes]])

    Xtr = np.vstack([graph_to_stats(g) for g in train_ds])
    ytr = np.array([int(g.y) for g in train_ds])
    Xva = np.vstack([graph_to_stats(g) for g in val_ds])
    yva = np.array([int(g.y) for g in val_ds])

    from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(max_iter=200)
    logit.fit(Xtr, ytr)
    auc_base = roc_auc_score(yva, logit.predict_proba(Xva)[:,1])
    print(f"[BASELINE] Logistic AUC={auc_base:.3f}")
    sys.exit(0)  # stop ici pour ce test


    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'])
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'])

    model = ParticleGNN().to(device)
    # hook gradient example
    if config['debug']:
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(grad_hook(n))

    opt   = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)

    best_auc, counter = 0, 0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[], 'val_auc':[]}

    with timer("training loop"):
        for epoch in range(1, config['epochs']+1):
            tl = train_one_epoch(model, train_loader, opt, epoch)
            vl, va, va_auc, vpreds, vlabs = evaluate(model, val_loader, split_name="val")

            sched.step(vl)
            history['train_loss'].append(tl)
            history['val_loss'].append(vl)
            history['val_acc'].append(va)
            history['val_auc'].append(va_auc)

            logger.info(f"Epoch {epoch}/{config['epochs']} TL={tl:.4f} VL={vl:.4f} "
                        f"VA={va:.4f} AUC={va_auc:.4f} LR={opt.param_groups[0]['lr']:.2e}")

            if config['debug']:
                t_opt, t_acc = optimal_threshold(vpreds, vlabs)
                logger.debug(f"[val] best_thresh={t_opt:.2f} acc={t_acc:.3f}")

            if va_auc > best_auc:
                best_auc=va_auc; counter=0
                torch.save(model.state_dict(), best_model_path)
            else:
                counter+=1
                if counter>=config['patience']:
                    logger.info("Early stopping")
                    break

    # Test
    model.load_state_dict(torch.load(best_model_path))
    tl, ta, tauc, tpreds, tlabs = evaluate(model, test_loader, split_name="test")
    logger.info(f"[TEST] L={tl:.4f} Acc={ta:.4f} AUC={tauc:.4f}")

    exists = os.path.isfile(performances_file)
    with open(performances_file,'a',newline='') as f:
        w = csv.writer(f)
        if not exists: w.writerow(["run_id","test_loss","test_acc","test_auc"])
        w.writerow([run_id, tl, ta, tauc])

    # Plots globales
    epochs = range(1,len(history['train_loss'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, history['val_acc'],    label='Val Acc')
    plt.plot(epochs, history['val_auc'],    label='Val AUC')
    plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.legend(); plt.title('Accuracy/AUC')
    plt.tight_layout(); plt.savefig(plot1_path); plt.close()

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.hist([p for p,l in zip(tpreds,tlabs) if l==0], bins=50, alpha=0.5, label='Pion')
    plt.hist([p for p,l in zip(tpreds,tlabs) if l==1], bins=50, alpha=0.5, label='Proton')
    plt.xlabel('Proton Prob'); plt.legend(); plt.title('Prob Distribution')
    plt.subplot(122)
    cm = confusion_matrix(tlabs, (tpreds>=0.5).astype(int))
    cmn = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    plt.imshow(cmn, interpolation='nearest'); plt.title('Confusion')
    plt.colorbar();
    for i in range(2):
        for j in range(2):
            plt.text(j,i,f"{cmn[i,j]*100:.1f}%", ha='center', va='center',
                     color='white' if cmn[i,j]>0.5 else 'black')
    plt.xticks([0,1],['Pion','Proton']); plt.yticks([0,1],['Pion','Proton'])
    plt.tight_layout(); plt.savefig(plot2_path); plt.close()

    # Extra debug dumps
    if config['debug']:
        np.save(f"debug/grad_norms_batch_{run_id}.npy", np.array(dbg['grad_norms_batch'], dtype=object), allow_pickle=True)
        np.save(f"debug/dt_samples_{run_id}.npy", np.array(dbg_hist_dt, dtype=object), allow_pickle=True)
        logger.debug("Saved extra debug npy files in debug/")

if __name__ == '__main__':
    main()
