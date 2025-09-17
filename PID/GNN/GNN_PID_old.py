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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, TAGConv, GraphConv, GraphNorm, global_max_pool, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

#####################################
#### 0. Configuration Parameters ####
#####################################

config = {
#    'pion_path' : "/home/ilc/wvaginay/Machine_Learning/data/csv/20kpi-_Emin1Emax100_continuous_bin_5.csv",
#   'proton_path' : "/home/ilc/wvaginay/Machine_Learning/data/csv/20kproton_Emin1Emax100_continuous_bin_5.csv",
    'pion_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv",
    'proton_path': "/gridgroup/ilc/midir/Timing/files/csv/1k_proton_E1to100_flat.csv",
    'date': datetime.now().strftime("%Y_%m_%d-%H_%M"),
#    'data_num': '20k',
    'data_num': '2k',
    'nb_features' : 7,      # thr count as 3
    'nb_classes' : 2,
    'knn_k': 8,
    'mode': 'route',        # route, time
    'sigma_gaus' : 0.2,     # ns   
    'time_window': 2.0,     # ns
    'hidden_dim': 256,
    'batch_size': 64,
    'lr': 1e-4,
    'epochs': 50,
    'patience' : 10,
    'seed' : 42,
}

# Parameters File Name
parameters_file = "run_parameters.csv"

# Get ID from Parameters file
def get_next_run_id(csv_path):
    if not os.path.exists(csv_path):
        return 1
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return 1
        return max(int(row["run_id"]) for row in rows) + 1

run_id = get_next_run_id(parameters_file)

# File Names
comments_file = "run_comments.csv"
performances_file = "GNN_performances.csv"
best_model_path = f"results/models/raw/GNN_best_model_{run_id}.pth"
plot1_path = f"results/plots/losses_and_accuracy_curves_GNN_{run_id}.png"
plot2_path = f"results/plots/performances_plots_GNN_{run_id}.png"

# Csv header preparation
included_keys = ["date", "data_num", "nb_features", "nb_classes", "mode", "time_window", "hidden_dim", "batch_size", "lr", "epochs", "seed", "patience"]
filtered_config = {k: config[k] for k in included_keys} 
fieldnames = ["run_id"] + included_keys
row_data = {"run_id": run_id}
row_data.update(filtered_config)

file_exists = os.path.exists(parameters_file)
with open(parameters_file, "a", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()  # Si fichier inexistant, écrire l'entête

    writer.writerow(row_data)

# --comment gestion
parser = argparse.ArgumentParser(description="Script de training avec logging de configuration.")
parser.add_argument("--comment", type=str, required=True, help="Commentaire descriptif pour ce run.")
args = parser.parse_args()
comment = args.comment

file_exists = os.path.isfile(comments_file)

with open(comments_file, mode="a", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["run_id", "comment"])  # En-tête
    writer.writerow([run_id, comment])

#####################################
### 1. Data loading and splitting ###
#####################################

def load_and_split_data(pion_path, proton_path, test_size=0.2, val_size=0.1):
    """ Load the data and slice them by group according to eventNumber"""

    # Sets label 0 and 1 for pion and proton, and creates event_id by merging eventNumber + _typeofparticle to get a unique event_id
    pion_df = pd.read_csv(pion_path).assign(label=0, event_id=lambda x: x['eventNumber'].astype(str)+'_pion')
    proton_df = pd.read_csv(proton_path).assign(label=1, event_id=lambda x: x['eventNumber'].astype(str)+'_proton')

    # Get rid of useless spaces before thr and time
    pion_df.columns = pion_df.columns.str.strip() 
    proton_df.columns = proton_df.columns.str.strip()		

    # Merge pion and proton dataframe
    full_df = pd.concat([pion_df, proton_df]).reset_index(drop=True)
    
    # Get unique event ID and label
    event_labels = full_df.groupby('event_id')['label'].first()
    event_ids = event_labels.index.values
    labels = event_labels.values
    
    # Split into train, val, test
    train_events, test_events = train_test_split(event_ids, test_size=test_size, stratify=labels, random_state=config['seed'] )
    train_events, val_events = train_test_split(train_events, test_size=val_size/(1-test_size), stratify=event_labels.loc[train_events], random_state=config['seed'])
    
    # Construct dataset
    return {
        'train': full_df[full_df['event_id'].isin(train_events)],
        'val': full_df[full_df['event_id'].isin(val_events)],
        'test': full_df[full_df['event_id'].isin(test_events)]
    }

#####################################
########## 2. Preprocessor ##########
#####################################

class GraphPreprocessor:
    """Normalise the features"""
    def __init__(self):
        self.spatial_scaler = None
        self.time_scaler = None

    def time_smearing(self, graphs, sigma_gaus_smearing):
        for g in graphs: 
            g['x'][:, 3] += np.random.normal(loc=0.0, scale=sigma_gaus_smearing, size=g['x'][:, 3].shape)

    def fit(self, graphs):
        """ Computes the normalisation parameters using only train data"""
        all_coords = []
        all_time = []
        
        for g in graphs:
            all_coords.append(g['x'][:, :3])
            all_time.append(g['x'][:, 3])

        # Position normalization
        self.spatial_scaler = RobustScaler().fit(np.vstack(all_coords))
        # Time normalization
        self.time_scaler = QuantileTransformer(output_distribution='normal').fit(np.hstack(all_time).reshape(-1,1))

    def transform(self, graph):
        """Apply these parameters to each graph"""
        # Position
        spatial = self.spatial_scaler.transform(graph['x'][:, :3])
        # Time
        time = self.time_scaler.transform(graph['x'][:, 3].reshape(-1,1))

        # Using one-hot to process threshold data
        thr = graph['x'][:, -1].astype(int)
        thr_onehot = np.eye(3)[thr-1]  # thr=[1,3], so thr=2 become [0,1,0]
        
        # Return a numpy array [Nhits, Nfeatures]
        return np.hstack([spatial, time, thr_onehot]).astype(np.float32)

#####################################
######### 3. Graph Building #########
#####################################

def knn_z(features, k, mode):
    """Creates the vertices : return a tensor [2, k*N] with 1st line = list of source nodes, 2nd line = list of target nodes. So far it only takes k=1"""

    if mode == 'route':
	    # features[N,F] (N = nb hits, F = features, >=3. coords[N,3] x_coords, y_coords, z_coords normaly
        coords = features[:, :3]
        z = coords[:, 2]

        # mask[N, N]  Generate z-coord mask by comparison so mask[i,j] = z[i]>z[j]
        mask = z.view(-1, 1) > z.view(1, -1)

        # [N, N]  Calculate square distance between all poles
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 3] 3 is for x y z
        dist_sq = (diff ** 2).sum(dim=-1)  # [N, N]

        # Set the distances that fail to meet the condition (z[i]>z[j])  to inf
        inf = torch.tensor(float('inf'), device=dist_sq.device)
        dist_sq_masked = torch.where(mask, dist_sq, inf)

        # Find nearest indices for each pole
        min_dist_sq, min_indices = torch.min(dist_sq_masked, dim=1)

        # Fileter out the invalid edges
        valid_mask = min_dist_sq != float('inf')
        sources = torch.where(valid_mask)[0]
        targets = min_indices[valid_mask]

    elif mode == 'time':
        z = features[:, 3]

        #  [N, N]  create the time comparison masks
        mask = z.view(-1, 1) > z.view(1, -1)

        #  [N, N]     calculate the square time distances between all poles
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # # 形状 [N, N]
        dist_sq = diff ** 2  # 形状 [N, N]

        #  set the distances that fail to meet the condition to inf
        inf = torch.tensor(float('inf'), device=dist_sq.device)
        dist_sq_masked = torch.where(mask, dist_sq, inf)

        #  find nearest indices for each pole
        min_dist_sq, min_indices = torch.min(dist_sq_masked, dim=1)

        #  fileter out the invalid edges (set to inf)
        valid_mask = min_dist_sq != float('inf')

        sources = torch.where(valid_mask)[0]

        targets = min_indices[valid_mask]

    #   produce edge index
    edge_index = torch.stack([sources, targets])
    return edge_index.cpu()


def build_graph(features, time, k=8, mode='route', time_window=2.0):
    """Build graph and apply a mask time window"""
    edge_features = torch.tensor(features[:, :], dtype=torch.float32)
    edge_index = knn_z(edge_features, k, mode)
    
    # Filter by time window
    src_time = time[edge_index[0]]
    dst_time = time[edge_index[1]]
    time_mask = (torch.abs(src_time - dst_time) < time_window).numpy()
    
    return edge_index[:, time_mask]


def create_pyg_data(raw_df, preprocessor, is_train):
    """Create a Dataset PyG from a DataDrame : input is a panda array with 1 line = all features; output is a list of graphs respresented by a Data object"""
    
    # Raw graph generation (before preprocessor)
    raw_graphs = []
    for event_id, group in raw_df.groupby('event_id'):
        coords = group[['x_coords', 'y_coords', 'z_coords']].values
        time = group['time'].values.reshape(-1,1)
        thr = group['thr'].values
        
        # x is raw features, y is the label of the event
        raw_graphs.append({
            'x': np.hstack([coords, time, thr.reshape(-1,1)]).astype(np.float32),
            'y': group['label'].iloc[0],
            'z_coords': coords[:,2]
        })
    
    # Time smearing
    if is_train:
        preprocessor.time_smearing(raw_graphs, config['sigma_gaus'])

    # Fit the preprocessor on train data
    if is_train:
        preprocessor.fit(raw_graphs)
    
    # Convert all graph data
    processed_graphs = []
    for raw in tqdm(raw_graphs):
        # Preprocessing, so normalisation of the graphs
        features = preprocessor.transform(raw)

        edge_index = build_graph(
            features=features,
            time=torch.tensor(raw['x'][:,3]),
            k=config['knn_k'],
            mode=config['mode'],
            time_window=config['time_window']
        )
        processed_graphs.append(Data(
            x=torch.tensor(features),
            edge_index=edge_index,
            y=torch.tensor([raw['y']], dtype=torch.long),
            num_nodes=len(features)
        ))
    
    return processed_graphs

#def process_dataset(df, preprocessor):
#    graphs = []
#    for event_id, group in df.groupby('event_id'):
#        coords = group[['x_coords', 'y_coords', 'z_coords']].values
#        time = group['time'].values.reshape(-1,1)
#        thr = group['thr'].values
#        
#        raw_graph = {
#            'x': np.hstack([coords, time, thr.reshape(-1,1)]).astype(np.float32),
#            'y': group['label'].iloc[0],
#            'z_coords': coords[:,2]
#        }
#        
#        features = preprocessor.transform(raw_graph)
#        edge_index = build_graph(
#            features=features,
#            time=torch.tensor(raw_graph['x'][:,3]),
#            k=config['knn_k'],
#            time_window=config['time_window']
#        )
#        graphs.append(Data(
#            x=torch.tensor(features, dtype=torch.float32),
#            edge_index=edge_index,
#            y=torch.tensor([raw_graph['y']], dtype=torch.long),
#            num_nodes=len(features)
#        ))
#    return graphs

#####################################
########### 4. GNN model ############
#####################################

class ParticleGNN(nn.Module):
    def __init__(self, nb_features=config['nb_features'],nb_classes=config['nb_classes'], hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(nb_features, hidden),
            nn.GELU()
        )
        self.conv1 = GATConv(hidden, hidden//4, heads=4)
        self.conv2 = TAGConv(hidden, hidden)
        self.conv3 = GraphConv(hidden, hidden)
        self.norm1 = GraphNorm(hidden)
        self.norm2 = GraphNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(hidden, config['nb_classes'])
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch).relu()
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch).relu()
        x = self.conv3(x, edge_index).relu()
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_pool = torch.cat([x_max, x_mean], dim=1)
        return self.classifier(x_pool).log_softmax(dim=-1)

#####################################
######### 5. Training utils #########
#####################################

def train_one_epoch(model, loader, optimizer, device):
    """Train the model for one epoch. Returns the average training loss per graph weighted by batch size"""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = nn.functional.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

def validate(model, loader, device, threshold=0.5):
    """"Evaluate the model on validation (or test) set. Returns: (val_loss, val_acc, val_auc, val_preds, val_labels)"""
    model.eval()
    val_preds = []
    val_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = nn.functional.nll_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs

            probs = out.exp()[:, 1].cpu().numpy()  # probability of class=1
            val_preds.extend(probs)
            val_labels.extend(data.y.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)

    # Binarize predictions at the given threshold
    val_preds_class = (np.array(val_preds) >= threshold).astype(int)
    val_acc = accuracy_score(val_labels, val_preds_class)
    val_auc = roc_auc_score(val_labels, val_preds)

    return val_loss, val_acc, val_auc, val_preds, val_labels

#####################################
######### 6. Full Pipeline# #########
#####################################

def main():

    t0 = time.time()

    # Load and split data
    raw_datasets = load_and_split_data(config['pion_path'], config['proton_path'])
    
    # Initiate the preprocessor once
    preprocessor = GraphPreprocessor()

    # Create all datasets
    train_data = create_pyg_data(raw_datasets['train'], preprocessor, True)
    val_data = create_pyg_data(raw_datasets['val'], preprocessor, False)
    test_data = create_pyg_data(raw_datasets['test'], preprocessor, False)
    
    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    test_loader = DataLoader(test_data, batch_size=config['batch_size'])
    
    # Model + Optimiser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleGNN(hidden=config['hidden_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    time_after_prep = time.time()
    time_preparation = time_after_prep - t0
    mins_prep, secs_preps = divmod(time_preparation, 60)
    print(f"Time preparation : {int(mins_prep)}m {int(secs_preps)}s\n")
    
    # Model Training Parameters
    best_val_auc = 0
    patience = config['patience']
    counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Model Training
    for epoch in range(config['epochs']):
        te = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_auc, val_preds, val_labels = validate(model, val_loader, device)
        epoch_time = time.time() - te
        mins, secs = divmod(epoch_time, 60)

        # Filling the arrays for plots
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:4f} | "
              f"Time for epoch: {int(mins)}m {int(secs)}s")
        
        # Save if best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)      
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    time_all_epochs = time.time() - time_after_prep
    mins_epochs, secs_epochs = divmod(time_all_epochs, 60)
    print(f"Time of all epochs : {int(mins_epochs)}m {int(secs_epochs)}s")

    print(f"\nModel saved in : {best_model_path}")

    # Test evaluation    
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_auc, test_preds, test_labels = validate(model, test_loader, device)

    print(f'\nTest Results:')
    print(f'  Test Loss: {test_loss:.4f}')
    print(f'  Test Accuracy: {test_acc:.4f}')
    print(f'  Test AUC: {test_auc:.4f}')

    # Upload performances to csv file
    file_exists = os.path.isfile(performances_file)

    with open(performances_file, mode="a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run_id", "test_loss", "test_acc", "test_auc"])  # En-tête
        writer.writerow([run_id, test_loss, test_acc, test_auc])

    print(f"\nTest Results saved in : {performances_file}")
    
    # Visualisation

    # Fig 1
    plt.figure(figsize=(10,4))

    # Losses
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot1_path)
    print(f"\nPlot saved in : {plot1_path}")
    plt.close()

    # Fig 2
    plt.figure(figsize=(12,5))

    # Combined hist for proton probability
    plt.subplot(121)
    plt.hist([p for p,l in zip(test_preds, test_labels) if l==0], 
             bins=50, alpha=0.5, label='Pion')
    plt.hist([p for p,l in zip(test_preds, test_labels) if l==1], 
             bins=50, alpha=0.5, label='Proton')
    plt.xlabel('Proton Probability')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(122)
    cm = confusion_matrix(test_labels, np.round(test_preds))
    cm_norm_row = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Normalisation by class

    plt.imshow(cm_norm_row, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm_row[i, j] * 100
            plt.text(j, i, f"{pct:.1f}%", 
                     ha='center', va='center', color='white' if cm_norm_row[i,j]>0.5 else 'black')

    plt.xticks([0,1], ['Pion', 'Proton'])
    plt.yticks([0,1], ['Pion', 'Proton'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    plt.tight_layout()
    plt.savefig(plot2_path)
    print(f"Plot saved in : {plot2_path}")
    plt.close()

if __name__ == '__main__':
    main()
