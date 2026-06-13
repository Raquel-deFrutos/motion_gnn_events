import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
torch.set_num_threads(os.cpu_count())
import warnings
warnings.filterwarnings("ignore", message="CUDA initialization")
from torch_geometric.nn.aggr import AttentionalAggregation



# ==========================================================
# DATASET
# ==========================================================
class MVSECGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_files, gt_array):
        self.graph_files = graph_files
        self.gt = gt_array

        self.data_cache = [None] * len(graph_files)

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        data_np = np.load(self.graph_files[idx], allow_pickle=True)
        # print(self.graph_files[idx])
        # print(data_np.files)

        x = torch.tensor(data_np["feats"], dtype=torch.float32)
        edge_index = torch.tensor(data_np["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data_np["edge_attr"], dtype=torch.float32)

        ##local
        # flow_path = f"C:\\Users\\Raquel\\Documents\\Doct\\Algoritmo\\MVSEC\\outdoor_day1\\optical_flow\\flow_{get_idx(self.graph_files[idx]):010d}.npy"
        #server
        flow_path = f"/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/optical_flow/flow_{get_idx(self.graph_files[idx]):010d}.npy"
        flow = np.load(flow_path).astype(np.float32)  # (2,H,W)

        flow_vec = torch.tensor([
            np.median(flow[0]),
            np.median(flow[1])
        ], dtype=torch.float32)


        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()



        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(self.gt[idx], dtype=torch.float32)[None, :]
        )

        data.y_flow = flow_vec  # ok

        return data

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    data = Batch.from_data_list(batch)
    data.y_flow = torch.stack([b.y_flow for b in batch], dim=0)



    return data
# ==========================================================
# GNN
# ==========================================================
class SimpleGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden):
        super().__init__(aggr='mean')

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        # 👇 PROYECCIÓN para residual
        self.res_proj = nn.Linear(node_dim, hidden) if node_dim != hidden else nn.Identity()

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(msg)

    def update(self, aggr_out, x):
        out = self.node_mlp(torch.cat([x, aggr_out], dim=-1))

        # residual correcto
        x_res = self.res_proj(x)

        return out + x_res


from torch_geometric.nn import GlobalAttention, MessagePassing

class EgoMotionGNN(nn.Module):
    def __init__(self, node_dim=10, edge_dim=6, hidden=64):
        super().__init__()

        self.gnn1 = SimpleGNNLayer(node_dim, edge_dim, hidden)
        self.gnn2 = SimpleGNNLayer(hidden, edge_dim, hidden)
        self.gnn3 = SimpleGNNLayer(hidden, edge_dim, hidden)

        # Attention pooling sobre nodos del grafo
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        )

        # Head de regresión (ego-motion: tx, ty, tz, wx, wy, wz)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 6)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn1(x, edge_index, edge_attr)
        x = self.gnn2(x, edge_index, edge_attr)
        x = self.gnn3(x, edge_index, edge_attr)

        # pooling con atención (en vez de mean pooling)
        x = self.pool(x, batch)

        return self.head(x)

# ==========================================================
# LOSS
# ==========================================================
# def loss_fn(pred, gt):
#     return F.smooth_l1_loss(pred, gt)

def loss_fn(pred, gt):
    return F.smooth_l1_loss(pred, gt)

def ego_to_flow(pred):
    tx, ty, tz = pred[:,0], pred[:,1], pred[:,2]
    wx, wy, wz = pred[:,3], pred[:,4], pred[:,5]

    # versión simple (suficiente para empezar)
    fx = tx - tz
    fy = ty - tz

    return torch.stack([fx, fy], dim=1)

def get_idx(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])




# ==========================================================
# MAIN
# ==========================================================
def main():

    # --------------------------
    # PATHS
    # --------------------------
    #local
    # graph_dir = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\davis\left\events\nodes/*.npz"
    #server
    graph_dir ="/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/nodes/nodes"
    
    #local
    # gt_csv = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1/gt_aligned.csv"
    #server
    gt_csv = "/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/gt/gt_aligned.csv"
    

    # LOAD FILES
    # graph_files_raw = glob.glob(graph_dir)
    graph_files_raw = glob.glob(os.path.join(graph_dir, "*.npz"))

    # índice -> archivo
    graph_dict = {get_idx(p): p for p in graph_files_raw}

    # LOAD GT
    df = pd.read_csv(gt_csv)
    gt = df[["t_x","t_y","t_z","w_x","w_y","w_z"]].values

    # índices disponibles en graphs
    valid_indices = sorted(graph_dict.keys())

    # recortar para que no se salga del GT
    valid_indices = [i for i in valid_indices if i < len(gt)]

    # ALINEACIÓN CORRECTA
    graph_files = [graph_dict[i] for i in valid_indices]
    gt = gt[valid_indices]
    N = len(graph_files)

    # --------------------------


    # --------------------------
    # SPLIT TRAIN / VAL
    # --------------------------
    split = int(0.8 * N)
    # --------------------------
    # NORMALIZAR SOLO CON TRAIN (CORRECTO)
    # --------------------------
    train_gt = gt[:split]

    mean = train_gt.mean(axis=0)
    std = train_gt.std(axis=0) + 1e-6

    gt_norm = (gt - mean) / std
    # print("GT mean:", gt_norm.mean(0))
    # print("GT std:", gt_norm.std(0))

    # guardar para luego (inferencia)
    np.save("gt_mean.npy", mean)
    np.save("gt_std.npy", std)
    
    
    # # --------------------------
    # # OVERFIT TEST
    # # --------------------------
    # small_N = 20

    # train_dataset = MVSECGraphDataset(
    #     graph_files[:small_N],
    #     gt_norm[:small_N]
    # )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=0
    # )
    # val_loader = DataLoader(
    # train_dataset,
    # batch_size=4,
    # shuffle=False,
    # collate_fn=collate_fn,
    # num_workers=0
    # )

    print("ANTES DATASET")
    train_dataset = MVSECGraphDataset(
        graph_files[:split],
        gt_norm[:split]
    )
    print("DATASET OK")

    val_dataset = MVSECGraphDataset(
        graph_files[split:],
        gt_norm[split:]
    )
    
    print("TEST SINGLE ITEM")
    _ = train_dataset[0]
    print("OK SINGLE ITEM")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,      # 👈 clave
        pin_memory=False
    )
    print("LOADER OK")

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False
    )
    data = next(iter(train_loader))
    print("FIRST BATCH OK")

    
    # data = next(iter(train_loader))
    # print("FIRST BATCH OK")

    # --------------------------
    # MODEL
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EgoMotionGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 3e-4)
    


    # --------------------------
    # TRAINING
    # --------------------------
    EPOCHS = 10

    best_val = float("inf") 

    for epoch in range(EPOCHS):
        model.train()
        print("START EPOCH", epoch)
        train_loss = 0

        for i, data in enumerate(train_loader):
            
            if i == 0:
                print("FIRST BATCH EPOCH", epoch)

            data = data.to(device)

            # # DEBUG
            # if epoch == 0 and i == 0:
            #     print("GT norm:", data.y[0])

            flow_gt = data.y_flow
            print("BEFORE MODEL")
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            print("AFTER MODEL")

            # # DEBUG
            # if epoch == 0 and i == 0:
            #     print("PRED:", pred[0].detach().cpu())

            loss_ego = loss_fn(pred, data.y)

            flow_pred = ego_to_flow(pred)

            # loss_flow = F.smooth_l1_loss(flow_pred, flow_gt)
            # loss = loss_ego + 0.2 * loss_flow

            loss = loss_ego

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ----------------------
        # VALIDATION
        # ----------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):

                if data is None:
                    continue

                data = data.to(device)

                pred = model(data.x, data.edge_index, data.edge_attr, data.batch)

                # SOLO 1 BATCH
                if i == 0:
                    # print("\nPRED[0]:", pred[0].detach().cpu().numpy())
                    # print("GT[0]:  ", data.y[0].detach().cpu().numpy())

                    pred_denorm = pred[0].detach().cpu().numpy() * std + mean
                    gt_denorm = data.y[0].detach().cpu().numpy() * std + mean

                    # print("\nPRED REAL:", pred_denorm)
                    # print("GT REAL:  ", gt_denorm)

                loss_ego = loss_fn(pred, data.y)
                val_loss += loss_ego.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std
            }, "best_model.pth")
            print(">> Guardado nuevo mejor modelo")


if __name__ == "__main__":
    main()
    
    

