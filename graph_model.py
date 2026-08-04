import os, psutil
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import time

from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
# torch.set_num_threads(os.cpu_count())
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


        x = torch.tensor(data_np["feats"], dtype=torch.float32)
        
        #grafo vacio
        if x.shape[0] == 0:
            return None
        
        edge_index = torch.tensor(data_np["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data_np["edge_attr"], dtype=torch.float32)




        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()



        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(self.gt[idx], dtype=torch.float32)[None, :]
        )


        return data

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    # data = Batch.from_data_list(batch)
    if len(batch) == 0:
        return None

    return Batch.from_data_list(batch)
    # data.y_flow = torch.stack([b.y_flow for b in batch], dim=0)



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
        self.bn = nn.BatchNorm1d(hidden)

        self.res_proj = nn.Linear(node_dim, hidden) if node_dim != hidden else nn.Identity()

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(msg)

    def update(self, aggr_out, x):

        out = self.node_mlp(torch.cat([x, aggr_out], dim=-1))

        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        x_res = self.res_proj(x)

        return out + x_res


from torch_geometric.nn import GlobalAttention, MessagePassing

class EgoMotionGNN(nn.Module):
    # def __init__(self, node_dim=10, edge_dim=6, hidden=64):
    ##nodes2
    def __init__(self, node_dim=19, edge_dim=9, hidden=64):
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

        # # Head de regresión (ego-motion: tx, ty, tz, wx, wy, wz)
        # self.head = nn.Sequential(
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     #traslacion + rptacion
        #     nn.Linear(hidden, 6)
        #     #solo traslacion o rotacion
        #     # nn.Linear(hidden, 3)
        # )
        
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 3)
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
    # #local
    # graph_dir = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\davis\left\events\nodes/*.npz"
    # graph_files_raw = glob.glob(graph_dir)
    #server
    #nodes1
    # graph_dir ="/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/nodes/nodes/*.npz"
    #nodes2
    # graph_dir ="/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/nodesPose/*.npz"
    graph_dir ="/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/nodesPose16/nodesPose16/*.npz"
    
    # graph_dir ="/home/rdefrutos/motion_gnn_events/data/MVSEC_indoorflying1/nodesPose/nodesPose/*.npz"
    graph_files_raw = glob.glob(graph_dir)
    # graph_files_raw = glob.glob(os.path.join(graph_dir, "*.npz"))
    
    #local
    # gt_csv = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1/gt_aligned.csv"
    #server
    # gt_csv = "/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/gt/gt_aligned.csv"
    gt_csv = "/home/rdefrutos/motion_gnn_events/data/MVSEC_outdoorday1/gt/gt_aligned_woInterp_PoseRe.csv"
    # gt_csv = "/home/rdefrutos/motion_gnn_events/data/MVSEC_indoorflying1/gt/gt_aligned_woInterp_PoseRe.csv"
    



    # índice -> archivo
    graph_dict = {get_idx(p): p for p in graph_files_raw}

    # LOAD GT
    df = pd.read_csv(gt_csv)
    # gt = df[["t_x","t_y","t_z","w_x","w_y","w_z"]].values
    # gt = df[["t_z"]].values
    # gt = df[["dx","dy","dz","rx","ry","rz"]].values
    # gt = df[["rx","ry","rz"]].values
    
    # #traslacion solo
    # gt = df[["t_x", "t_y", "t_z"]].values
    
    # #rotacion solo
    # gt = df[["w_x", "w_y", "w_z"]].values
    gt = df[["rx","ry","rz"]].values

    # índices disponibles en graphs
    valid_indices = sorted(graph_dict.keys())

    # recortar para que no se salga del GT
    valid_indices = [i for i in valid_indices if i < len(gt)]

    # ALINEACIÓN CORRECTA
    graph_files = [graph_dict[i] for i in valid_indices]
    gt = gt[valid_indices]
    N = len(graph_files)
    empty_graphs = []

    for p in graph_files:

        data = np.load(p)

        n_nodes = data["feats"].shape[0]
        n_edges = data["edge_index"].shape[0]

        if n_nodes == 0:
            empty_graphs.append(
                (get_idx(p), n_nodes, n_edges)
            )

    print(f"Número de grafos vacíos: {len(empty_graphs)}")

    for idx, nodes, edges in empty_graphs[:20]:
        print(
            f"frame={idx} | "
            f"nodos={nodes} | "
            f"edges={edges}"
        )

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
    
    print("GT mean:", np.round(mean, 3))
    print("GT std :", np.round(std, 3))

    gt_norm = (gt - mean) / std
    # print("GT mean:", gt_norm.mean(0))
    # print("GT std:", gt_norm.std(0))

    # guardar para luego (inferencia)
    np.save("gt_mean.npy", mean)
    np.save("gt_std.npy", std)
    
    
    # # # --------------------------
    # # # OVERFIT TEST
    # # # --------------------------
    # small_N = 20

    # train_dataset = MVSECGraphDataset(
    #     graph_files[:small_N],
    #     gt_norm[:small_N]
    # )

    # val_dataset = train_dataset


    train_dataset = MVSECGraphDataset(
        graph_files[:split],
        gt_norm[:split]
    )

    val_dataset = MVSECGraphDataset(
        graph_files[split:],
        gt_norm[split:]
    )
    

    _ = train_dataset[0]


    train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,   # ya correcto
    )


    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=16,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     num_workers=2,
    #     pin_memory=False
    # )
    val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,   # cámbialo a 0 también
    )
    
    
    data = next(iter(train_loader))
    # print("Número de nodos:", data.x.shape[0])
    # print("Número de grafos:", data.num_graphs)
    # print("Nodos por grafo:")

    for g in range(data.num_graphs):
        n = (data.batch == g).sum().item()
        print(f"Grafo {g}: {n} nodos")


    
    # data = next(iter(train_loader))
    # print("FIRST BATCH OK")

    # --------------------------
    # MODEL
    # --------------------------
    
    #server
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    #local
    # device = torch.device("cpu")
    model = EgoMotionGNN().to(device)
    opt = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
        )
    


    # --------------------------
    # TRAINING
    # --------------------------
    EPOCHS = 30

    best_val = float("inf") 

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for i, data in enumerate(train_loader):


            # print("DEVICE:", device)
            # print("X device before:", data.x.device)

            data = data.to(device)
            # print("  nodes:", data.x.shape[0])
            # print("  edges:", data.edge_index.shape[1])

            # print("X device after:", data.x.device)

            # flow_gt = data.y_flow
    
            start = time.time()
        
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
         
            # print("MODEL TIME:", time.time() - start)

            # # DEBUG
            # if epoch == 0 and i == 0:
            #     print("PRED:", pred[0].detach().cpu())
           
            loss_ego = loss_fn(pred, data.y)
         

            # flow_pred = ego_to_flow(pred)

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

            component_errors = []
            component_errors_real = []   # <-- AÑADE ESTO

            for i, data in enumerate(val_loader):

                if data is None:
                    continue

                data = data.to(device)

                pred = model(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                    data.batch
                )
                
                pred_denorm = pred.cpu().numpy() * std + mean
                gt_denorm = data.y.cpu().numpy() * std + mean
                
                # print("pred.shape =", pred.shape)
                # print("data.y.shape =", data.y.shape)
                # print("pred_denorm.shape =", pred_denorm.shape)
                # print("gt_denorm.shape =", gt_denorm.shape)
                # print("num_graphs =", data.num_graphs)
                error_real = np.abs(pred_denorm - gt_denorm)

                component_errors_real.append(error_real)

                # =========================
                # ERROR POR COMPONENTE
                # =========================
                error = torch.abs(pred - data.y)

                component_errors.append(
                    error.cpu().numpy()
                )


                loss_ego = loss_fn(pred, data.y)
                val_loss += loss_ego.item()


        val_loss /= len(val_loader)
        component_errors_real = np.concatenate(
            component_errors_real,
            axis=0
        )

        mae_real = component_errors_real.mean(axis=0)

        print(
            # "MAE real:",
            f"tx={mae_real[0]:.4f}",
            f"ty={mae_real[1]:.4f}",
            f"tz={mae_real[2]:.4f}",
            # f"wx={mae_real[3]:.4f}",
            # f"wy={mae_real[4]:.4f}",
            # f"wz={mae_real[5]:.4f}",
        )
                
                # =========================
        # MAE POR COMPONENTE
        # =========================
        # component_errors = np.concatenate(component_errors, axis=0)

        # mae_components = component_errors.mean(axis=0)

        # print(
        #     "MAE:",
        #     f"Tx={mae_components[0]:.3f}",
        #     f"Ty={mae_components[1]:.3f}",
        #     f"Tz={mae_components[2]:.3f}",
        #     f"wx={mae_components[3]:.3f}",
        #     f"wy={mae_components[4]:.3f}",
        #     f"wz={mae_components[5]:.3f}",
        # )
       

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # component_errors = np.concatenate(component_errors, axis=0)

        # mae_components = component_errors.mean(axis=0)

        # labels = ["c1", "c2", "c3"]

        # print(
        #     "MAE:",
        #     *[f"{name}={err:.3f}" for name, err in zip(labels, mae_components)]
        # )
        
        
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
    
    

