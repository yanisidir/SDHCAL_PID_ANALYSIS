import torch
import numpy as np
import pandas as pd
from torch import nn
from datetime import datetime
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, TAGConv, GraphConv, GraphNorm, global_max_pool, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数配置
config = {
    'date': datetime.now().strftime("%Y_%m_%d-%H_%M"),
    'data_num': '20k',
    'knn_k': 8,
    'mode': 'route',        # route, time
    'time_window': 2.0,  # ns
    'hidden_dim': 256,
    'batch_size': 64,
    'lr': 3e-4,
    'epochs': 50,
    'layer_z_bins': 10
}

def knn_z(features, k, mode):
    if mode == 'route':
        coords = features[:, :3]
        z = coords[:, 2]

        # 创建z坐标比较掩码 [N, N]  generate z-coord mask by comparison
        mask = z.view(-1, 1) > z.view(1, -1)

        # 计算所有点对的平方距离 [N, N]     calculate square distance between all poles
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # 形状 [N, N, 3]
        dist_sq = (diff ** 2).sum(dim=-1)  # 形状 [N, N]

        # 将不满足条件的距离设为无穷大      set the distances that fail to meet the condition to inf
        inf = torch.tensor(float('inf'), device=dist_sq.device)
        dist_sq_masked = torch.where(mask, dist_sq, inf)

        # 找到每个点的最近邻索引        find nearest indices for each pole
        min_dist_sq, min_indices = torch.min(dist_sq_masked, dim=1)

        # 过滤无效的边（无候选点的情况）    fileter out the invalid edges
        valid_mask = min_dist_sq != float('inf')
        sources = torch.where(valid_mask)[0]
        targets = min_indices[valid_mask]
    elif mode == 'time':
        z = features[:, 3]

        # 创建z坐标比较掩码 [N, N]  create the time comparison masks
        mask = z.view(-1, 1) > z.view(1, -1)

        # 计算所有点对的平方距离 [N, N]     calculate the square time distances between all poles
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # # 形状 [N, N]
        dist_sq = diff ** 2  # 形状 [N, N]

        # 将不满足条件的距离设为无穷大      set the distances that fail to meet the condition to inf
        inf = torch.tensor(float('inf'), device=dist_sq.device)
        dist_sq_masked = torch.where(mask, dist_sq, inf)

        # 找到每个点的最近邻索引        find nearest indices for each pole
        min_dist_sq, min_indices = torch.min(dist_sq_masked, dim=1)

        # 过滤无效的边（无候选点的情况）    fileter out the invalid edges
        valid_mask = min_dist_sq != float('inf')
        sources = torch.where(valid_mask)[0]
        targets = min_indices[valid_mask]

    # 构建边索引矩阵    produce edge index
    edge_index = torch.stack([sources, targets])
    return edge_index.cpu()


def load_and_split_data(pion_path, proton_path, test_size=0.2, val_size=0.1):
    """加载数据并进行事件级别的分层划分 load the data and slice them according to eventNumber"""
    pion_df = pd.read_csv(pion_path).assign(label=0, event_id=lambda x: x['eventNumber'].astype(str)+'_pion')
    proton_df = pd.read_csv(proton_path).assign(label=1, event_id=lambda x: x['eventNumber'].astype(str)+'_proton')

    # Get rid of useless spaces before thr and time
    pion_df.columns = pion_df.columns.str.strip() 
    proton_df.columns = proton_df.columns.str.strip()	

    # Merge pion and proton dataframe
    full_df = pd.concat([pion_df, proton_df]).reset_index(drop=True)

    # Apply a smearing of 100ps
    np.random.seed(152)
    full_df['time'] += np.random.normal(loc=0.0, scale=0.1, size=full_df['time'].shape)

    # Truncate the value to 0.01 of precision ( bins of 10ps )
    full_df['time'] = (full_df['time'] * 100).astype(int) / 100
    
    # 获取唯一事件ID及对应标签  get unique event ID and label
    event_labels = full_df.groupby('event_id')['label'].first()
    event_ids = event_labels.index.values
    labels = event_labels.values
    
    # 分层划分      split into train, val, test
    train_events, test_events = train_test_split(
        event_ids, test_size=test_size, stratify=labels, random_state=42
    )
    train_events, val_events = train_test_split(
        train_events, 
        test_size=val_size/(1-test_size), 
        stratify=event_labels.loc[train_events], 
        random_state=42
    )
    
    # 构建数据集    construct dataset
    return {
        'train': full_df[full_df['event_id'].isin(train_events)],
        'val': full_df[full_df['event_id'].isin(val_events)],
        'test': full_df[full_df['event_id'].isin(test_events)]
    }

# 2. 数据预处理管道
class GraphPreprocessor:
    def __init__(self):
        self.spatial_scaler = None
        self.time_scaler = None
        self.layer_scalers = {}
        self.z_bins = None

    def fit(self, graphs):
        """基于训练数据拟合预处理参数 fit parameters of the preprocessor using only train data"""
        all_coords = []
        all_time = []
        all_z = []
        
        for g in graphs:
            all_coords.append(g['x'][:, :3])
            all_time.append(g['x'][:, 3])
            all_z.append(g['z_coords'])
        
        # 空间坐标归一化    position normalization
        self.spatial_scaler = RobustScaler().fit(np.vstack(all_coords))
        # 时间分位数归一化      time normalization
        self.time_scaler = QuantileTransformer(output_distribution='normal').fit(np.hstack(all_time).reshape(-1,1))
        # 分层z归一化       layer z normalization
        z_all = np.hstack(all_z)
        self.z_bins = np.linspace(z_all.min(), z_all.max(), config['layer_z_bins']+1)
        
        for bin_idx in range(config['layer_z_bins']):
            mask = (z_all >= self.z_bins[bin_idx]) & (z_all < self.z_bins[bin_idx+1])
            if mask.sum() > 0:
                self.layer_scalers[bin_idx] = RobustScaler().fit(z_all[mask].reshape(-1,1))

    def transform(self, graph):
        """应用预处理到单个图 apply preprocessor to each graph"""
        # 空间坐标      position
        spatial = self.spatial_scaler.transform(graph['x'][:, :3])
        # 时间      time
        time = self.time_scaler.transform(graph['x'][:, 3].reshape(-1,1))
        # 分层z     z layer process
        z = graph['z_coords'].copy()
        for bin_idx in range(config['layer_z_bins']):
            mask = (z >= self.z_bins[bin_idx]) & (z < self.z_bins[bin_idx+1])
            if mask.sum() > 0 and bin_idx in self.layer_scalers:
                z[mask] = self.layer_scalers[bin_idx].transform(z[mask].reshape(-1,1)).flatten()
        # 能量等级one-hot       using one-hot to process threshold data
        thr = graph['x'][:, -1].astype(int)
        thr_onehot = np.eye(3)[thr-1]  # thr取值为1,2,3     thr=[1,3]
        
        return np.hstack([spatial, time, z.reshape(-1,1), thr_onehot]).astype(np.float32)

# 3. 图结构构建
def build_graph(features, time, k=8, mode='route', time_window=2.0):
    """构建带时间约束的KNN图结构"""
    edge_features = torch.tensor(features[:, :], dtype=torch.float)
    edge_index = knn_z(edge_features, k, mode)
    
    # 时间一致性过滤        filter by time window
    src_time = time[edge_index[0]]
    dst_time = time[edge_index[1]]
    time_mask = (torch.abs(src_time - dst_time) < time_window).numpy()
    
    return edge_index[:, time_mask]

def create_pyg_data(raw_df):
    """从DataFrame创建PyG数据集"""
    preprocessor = GraphPreprocessor()
    
    # 原始图数据生成        raw graph generation
    raw_graphs = []
    for event_id, group in raw_df.groupby('event_id'):
        coords = group[['x_coords', 'y_coords', 'z_coords']].values
        time = group['time'].values.reshape(-1,1)
        thr = group['thr'].values
        
        raw_graphs.append({
            'x': np.hstack([coords, time, thr.reshape(-1,1)]).astype(np.float32),
            'y': group['label'].iloc[0],
            'z_coords': coords[:,2]
        })
    
    # 仅在训练集上拟合预处理器      only fit the preprocessor on train data
    if raw_df is raw_datasets['train']:
        preprocessor.fit(raw_graphs)
    
    # 转换所有图数据    convert all graph data
    processed_graphs = []
    for raw in tqdm(raw_graphs):
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

# 4. GNN模型 GNN model
class ParticleGNN(nn.Module):
    def __init__(self, input_dim=8, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
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
            nn.Linear(hidden, 2)
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
    
def process_dataset(df, preprocessor):
    graphs = []
    for event_id, group in df.groupby('event_id'):
        coords = group[['x_coords', 'y_coords', 'z_coords']].values
        time = group['time'].values.reshape(-1,1)
        thr = group['thr'].values
        
        raw_graph = {
            'x': np.hstack([coords, time, thr.reshape(-1,1)]).astype(np.float32),
            'y': group['label'].iloc[0],
            'z_coords': coords[:,2]
        }
        
        features = preprocessor.transform(raw_graph)
        edge_index = build_graph(
            features=features,
            time=torch.tensor(raw_graph['x'][:,3]),
            k=config['knn_k'],
            time_window=config['time_window']
        )
        graphs.append(Data(
            x=torch.tensor(features),
            edge_index=edge_index,
            y=torch.tensor([raw_graph['y']], dtype=torch.long),
            num_nodes=len(features)
        ))
    return graphs

# 5. 训练流程
def main():
    # 数据准备
    global raw_datasets  # 用于create_pyg_data判断训练集
#    pion_route = f"D:\\GNN\\simulate_data\\transfer_9435897_files_031472cb\\{config['data_num']}pi-_Emin1Emax100_digitized_hits_continuous_merged.csv"
#    proton_route = f"D:\\GNN\\simulate_data\\transfer_9435897_files_031472cb\\{config['data_num']}proton_Emin1Emax100_digitized_hits_continuous_mergedd.csv"
    pion_route = "/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv"
    proton_route = "/gridgroup/ilc/midir/Timing/files/csv/1k_proton_E1to100_flat.csv"
    raw_datasets = load_and_split_data(pion_route, proton_route)

    # 仅在训练集上创建并拟合预处理器
    train_preprocessor = GraphPreprocessor()
    train_graphs = []
    
    # 处理训练集
    for event_id, group in raw_datasets['train'].groupby('event_id'):
        coords = group[['x_coords', 'y_coords', 'z_coords']].values
        time = group['time'].values.reshape(-1,1)
        thr = group['thr'].values
        train_graphs.append({
            'x': np.hstack([coords, time, thr.reshape(-1,1)]).astype(np.float32),
            'y': group['label'].iloc[0],
            'z_coords': coords[:,2]
        })
    train_preprocessor.fit(train_graphs)  # 关键点：只在训练集上拟合
    
    # 创建所有数据集
    train_data = process_dataset(raw_datasets['train'], train_preprocessor)
    val_data = process_dataset(raw_datasets['val'], train_preprocessor)  # 使用训练集的预处理器
    test_data = process_dataset(raw_datasets['test'], train_preprocessor)  # 使用训练集的预处理器
    
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    test_loader = DataLoader(test_data, batch_size=config['batch_size'])
    
    # 模型训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleGNN(hidden=config['hidden_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    best_val_auc = 0
    patience = 8
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = nn.functional.nll_loss(model(batch), batch.y)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                probs = model(batch).exp()[:,1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch.y.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f'Epoch {epoch+1}/{config["epochs"]} | Val AUC: {val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    # 测试评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            probs = model(batch).exp()[:,1].cpu().numpy()
            test_preds.extend(probs)
            test_labels.extend(batch.y.cpu().numpy())
    
    print(f'\nTest Results:')
    print(f'AUC: {roc_auc_score(test_labels, test_preds):.4f}')
    print(f'Accuracy: {accuracy_score(test_labels, np.round(test_preds)):.4f}')
    
    # 可视化
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.hist([p for p,l in zip(test_preds, test_labels) if l==0], 
             bins=50, alpha=0.5, label='Pion')
    plt.hist([p for p,l in zip(test_preds, test_labels) if l==1], 
             bins=50, alpha=0.5, label='Proton')
    plt.xlabel('Proton Probability'), plt.legend()
    
    plt.subplot(122)
    cm = confusion_matrix(test_labels, np.round(test_preds))
    cm_norm_row = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            pct = cm_norm_row[i, j] * 100
            plt.text(j, i, f"{pct:.1f}%", ha='center', va='center', color='white' if cm_norm_row[i,j]>0.5 else 'black')
    plt.xticks([0,1], ['Pion', 'Proton'])
    plt.yticks([0,1], ['Pion', 'Proton'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    plt.tight_layout()
#    plt.savefig(f"D:\\GNN\\results\\{config['data_num']}_{config['mode']}_performance.png")
    plt.savefig(f"{config['date']}_performance.png")
    plt.close()

    # 确保目标目录存在
    file_name = f"k_results_{config['date']}.txt"

    # 拼接完整的文件路径

    # 保存矩阵为txt文件
    # Save matrix as txt file
    np.savetxt(
        fname=file_name,  # 文件路径  File path
        X=[roc_auc_score(test_labels, test_preds), accuracy_score(test_labels, np.round(test_preds))],         # 要保存的矩阵  Matrix to save
        fmt="%f",
        delimiter=","     # 分隔符（可选，默认是空格）  Delimiter (optional, default is space)
    )

if __name__ == '__main__':
    main()
