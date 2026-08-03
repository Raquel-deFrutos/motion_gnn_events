
import argparse
from pathlib import Path
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# ==========================================================
# ARGUMENTS AND FOLDERS
# ==========================================================
parser = argparse.ArgumentParser(
    description="Preprocess MVSEC sequences"
)
parser.add_argument("--gt", type=Path, required=True,
                    help="Path to *_gt.hdf5")
parser.add_argument("--data", type=Path, required=True,
                    help="Path to *_data.hdf5")
parser.add_argument(
    "--freq", type=str, choices=["depth", "image", "flow", "pose"], default="pose",
    help="Frequency to split events: 'depth' (~20Hz), 'flow' (~20Hz),  or 'image' (~45Hz)")
# parser.add_argument("--voxel_bins", type=int, default=5)

args = parser.parse_args()
freq = args.freq

GT_PATH = args.gt
DATA_PATH = args.data
SEQ_NAME = GT_PATH.stem.replace("_gt", "")  
OUTPUT_DIR = GT_PATH.parent / SEQ_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# NUM_BINS = args.voxel_bins


TIMESTAMPS_DEPTH_TXT = OUTPUT_DIR / "timestamps_depth.txt"
TIMESTAMPS_POSE_TXT = OUTPUT_DIR / "timestamps_pose.txt"
TIMESTAMPS_FLOW_TXT  = OUTPUT_DIR / "timestamps_flow.txt"
TIMESTAMPS_IMAGES_TXT  = OUTPUT_DIR / "timestamps_images.txt"
EVENTS_OUT_DIR       = OUTPUT_DIR / "davis" / "left" / "events"
EVENTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "depth_rectified").mkdir(exist_ok=True)
(OUTPUT_DIR / "optical_flow").mkdir(exist_ok=True)



if freq == "depth":
    TIMESTAMPS_TXT = OUTPUT_DIR / "timestamps_depth.txt"
elif freq == "image":
    TIMESTAMPS_TXT = OUTPUT_DIR / "timestamps_images.txt"
elif freq == "flow":
    TIMESTAMPS_TXT = OUTPUT_DIR / "timestamps_flow.txt"
    
elif freq == "pose":
    TIMESTAMPS_TXT = OUTPUT_DIR / "timestamps_pose.txt"



EVENT_DTYPE = np.dtype([
    ("x", np.int16),
    ("y", np.int16),
    ("p", np.int8),
    ("ts", np.float64),
])


# # ==========================================================
# # GT PROCESSING
# # ==========================================================
def save_gt_timestamps():
    with h5py.File(GT_PATH, "r") as f:
        flow_ts  = f["davis/left/flow_dist_ts"][:]
        depth_ts = f["davis/left/depth_image_raw_ts"][:]
        
    with h5py.File(DATA_PATH, "r") as f:
        image_ts = f["davis/left/image_raw_ts"][:]

    np.savetxt(TIMESTAMPS_DEPTH_TXT, depth_ts, fmt="%.18e")
    np.savetxt(TIMESTAMPS_FLOW_TXT, flow_ts,  fmt="%.18e")
    np.savetxt(TIMESTAMPS_IMAGES_TXT, image_ts,  fmt="%.18e")
    print(f"Timestamps saved")




# ==========================================================
# EVENTS PROCESSING
# ==========================================================
def load_timestamps_txt(path: Path) -> np.ndarray:
    ts = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                ts.append(float(line))
    ts = np.asarray(ts, dtype=np.float64)
    if not np.all(np.diff(ts) >= 0):
        ts = np.sort(ts, kind="mergesort")
    return ts


def write_events_h5(path: Path, x, y, p, t): 
    n = len(t)
    data = np.empty(n, dtype=EVENT_DTYPE)
    
    if n > 0:
        data["x"] = np.clip(x, -32768, 32767).astype(np.int16) 
        data["y"] = np.clip(y, -32768, 32767).astype(np.int16) 
        data["p"] = p.astype(np.int8)
        data["ts"] = t.astype(np.float64)

    with h5py.File(path, "w") as h5: 
        h5.create_dataset( 
            "myDataset", 
            data=data, 
            compression="gzip", 
            compression_opts=4
        )     
            

def split_events_by_timestamps():

    if not TIMESTAMPS_TXT.exists():
        raise FileNotFoundError(f"{TIMESTAMPS_TXT} not found. Run save_gt_timestamps() first.")
    
    img_ts = load_timestamps_txt(TIMESTAMPS_TXT)
    n_frames = len(img_ts)

    with h5py.File(DATA_PATH, "r") as f:
        if "davis/left/events" in f:
            ev = f["davis/left/events"][:]
        elif "myDataset" in f:
            ev = f["myDataset"][:]
        else:
            raise ValueError("No se encontró dataset de eventos")
        
        if ev.dtype.names is not None:
            # estructurado (tu formato o algunos MVSEC)
            x = ev["x"]
            y = ev["y"]
            t = ev["ts"]
            p = ev["p"]
        else:
            # matriz (MVSEC típico)
            x = ev[:, 0]
            y = ev[:, 1]
            t = ev[:, 2]
            p = ev[:, 3]

        # asegurar orden temporal (sin coste innecesario)
        if not np.all(np.diff(t) >= 0):
            order = np.argsort(t)
            x, y, t, p = x[order], y[order], t[order], p[order]

        t = t.astype(np.float64)

        # índices de corte
        indices = np.searchsorted(t, img_ts, side="right")


        # eventos entre timestamps
        for i in range(n_frames - 1):
            start = indices[i]
            end = indices[i + 1]

            write_events_h5(
                EVENTS_OUT_DIR / f"{i:06d}.h5",
                x[start:end],
                y[start:end],
                p[start:end],
                t[start:end]
            )


    print("Events split and saved")


# ==========================================================
# VOXEL GRID GENERATION
# ==========================================================
def events_to_voxel_grid(events, num_bins, height, width):
    if len(events) == 0:
        return np.zeros((num_bins, height, width), dtype=np.float32)
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    ts = events[:, 3].astype(np.float64)
    t0, t1 = ts.min(), ts.max()
    deltaT = max(t1 - t0, 1e-6)

    ts_norm = (num_bins - 1) * (ts - t0) / deltaT

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    pols = events[:, 2].astype(np.int8)


    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    tis = ts_norm.astype(np.int32)
    dts = ts_norm - tis

    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    for offset, vals in zip([0, 1], [vals_left, vals_right]):
        mask = (tis + offset) < num_bins
        idx = (
            xs[mask]
            + ys[mask] * width
            + (tis[mask] + offset) * width * height
        )
        np.add.at(voxel_grid, idx, vals[mask])

    return voxel_grid.reshape(num_bins, height, width)


    

# ==========================================================
# NODES
# ==========================================================
def build_nodes(events, H, W, cell_size=8):

    """
    events: (N,4) -> x,y,p,t
    returns:
        feats: (M, F)
        coords: (M, 3)  -> x,y,t
    """
    if len(events) == 0:
        return np.zeros((0, 18), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    xs = events[:, 0].astype(np.float32)
    ys = events[:, 1].astype(np.float32)
    ps = events[:, 2].astype(np.float32)
    ts = events[:, 3].astype(np.float64)
    

    gx = (xs.astype(np.int32) // cell_size)
    gy = (ys.astype(np.int32) // cell_size)

    grid_dict = defaultdict(list)

    for i in range(len(events)):
        grid_dict[(gx[i], gy[i])].append(i)

    feats = []
    coords = []

    t_global_min = ts.min()
    t_global_max = ts.max()
    t_range = (t_global_max - t_global_min) + 1e-6

    for (cx, cy), idxs in grid_dict.items():
        idxs = np.array(idxs)
        
        # x0 = cx * cell_size
        # x1 = min((cx + 1) * cell_size, W)

        # y0 = cy * cell_size
        # y1 = min((cy + 1) * cell_size, H)

        # depth_values = depth[y0:y1, x0:x1].reshape(-1)

        # depth_values = depth_values[np.isfinite(depth_values)]
        # depth_values = depth_values[depth_values > 0]

        # if len(depth_values) == 0:
        #     depth_mean = 0.0
        #     depth_std = 0.0
        # else:
        #     depth_mean = np.mean(depth_values)
        #     depth_std = np.std(depth_values)

        # depth_mean = np.log1p(depth_mean)
        # depth_std = np.log1p(depth_std)

        x_mean = (xs[idxs].mean() / W) * 2 - 1
        y_mean = (ys[idxs].mean() / H) * 2 - 1
        x_std = xs[idxs].std() / W
        y_std = ys[idxs].std() / H

        t_mean = ((ts[idxs].mean() - t_global_min) / t_range) * 2 - 1
        t_std  = ts[idxs].std() / t_range

        num_events = len(idxs)
        num_events_norm = np.log1p(num_events)


        pos_ratio = np.sum(ps[idxs] > 0) / (num_events + 1e-6)
        # neg_ratio = np.sum(ps[idxs] < 0) / (num_events + 1e-6)
        

        xs_c = xs[idxs]
        ys_c = ys[idxs]
        ts_c = ts[idxs]

        order = np.argsort(ts_c)

        xs_c = xs_c[order]
        ys_c = ys_c[order]
        ts_c = ts_c[order]
        
        if len(ts_c) > 2:
            A = np.vstack([ts_c, np.ones_like(ts_c)]).T

            vx, _ = np.linalg.lstsq(A, xs_c, rcond=None)[0]
            vy, _ = np.linalg.lstsq(A, ys_c, rcond=None)[0]

            dt_local = ts_c[-1] - ts_c[0] + 1e-6
            # normalización
            vx = np.tanh(vx * dt_local / W)
            vy = np.tanh(vy * dt_local / H)
        else:
            vx = 0.0
            vy = 0.0

        # dt = (ts_c[-1] - ts_c[0]) + 1e-6

        xs_norm = xs[idxs] / W
        ys_norm = ys[idxs] / H
        ts_norm = (ts[idxs] - t_global_min) / t_range

        if num_events > 1:
            cov_xy = np.cov(xs_norm, ys_norm)[0, 1]
            cov_xt = np.cov(xs_norm, ts_norm)[0, 1]
            cov_yt = np.cov(ys_norm, ts_norm)[0, 1]
        else:
            cov_xy = 0.0
            cov_xt = 0.0
            cov_yt = 0.0

        # --- histograma temporal (3 bins) ---
        t0, t1 = ts[idxs].min(), ts[idxs].max()
        dt = (t1 - t0) + 1e-6

        bins = ((ts[idxs] - t0) / dt * 3).astype(int)
        bins = np.clip(bins, 0, 2)

        hist = np.bincount(bins, minlength=3) / (num_events + 1e-6)
        dt_cell = (ts_c[-1] - ts_c[0]) / t_range

        pol_mean = ps[idxs].mean()
        event_rate = num_events / (ts_c[-1] - ts_c[0] + 1e-6)
        event_rate = np.log1p(event_rate)

        node_feat = [
            num_events_norm,
            t_std,
            x_std,
            y_std,
            cov_xy,
            cov_xt,
            cov_yt,
            hist[0],
            hist[1],
            hist[2],
            dt_cell,
            event_rate,
            pos_ratio,
            pol_mean,
            vx,
            vy,
            # depth_mean,
            # depth_std
        ]

        feats.append(node_feat)

        # --- coords separadas ---
        coords.append([x_mean, y_mean, t_mean])

    return np.array(feats, dtype=np.float32), np.array(coords, dtype=np.float32)



def build_graph(coords, k=8, alpha_t=2.0):
    """
    coords: (M, 3) -> x, y, t (normalizados)
    returns:
        edge_index: (E, 2)
        edge_attr: (E, 4) -> dx, dy, dt, dist_xy
    """
    coords_knn = coords.copy()
    coords_knn[:, 2] *= alpha_t



    if len(coords) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 9), dtype=np.float32)
        )

    # kNN en espacio-tiempo

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, len(coords)),
        algorithm='kd_tree'
    ).fit(coords_knn)
    distances, indices = nbrs.kneighbors(coords_knn)

    edge_index = []
    edge_attr = []

    for i in range(len(coords)):
        for j in indices[i][1:]:
            if abs(coords[j, 2] - coords[i, 2]) > 0.3:
                continue

            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            dt = coords[j, 2] - coords[i, 2]   # sin alpha
            
            angle = np.arctan2(dy, dx)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            dist_xy = np.sqrt(dx*dx + dy*dy)
            dist_t = abs(dt)
            dist_total = np.sqrt(dx*dx + dy*dy + dt*dt)
            
            
            # 🔥 normalización (clave)
            dist_xy = dist_xy / np.sqrt(2)
            dist_total = dist_total / np.sqrt(3)
            
            velocity = np.tanh(dist_xy / (dist_t + 0.05))

            # i -> j
            edge_index.append([i, j])
            # edge_attr.append([dx, dy, dt, dist_xy, dist_t, dist_total])
            edge_attr.append([
                dx,
                dy,
                dt,
                dist_xy,
                dist_t,
                dist_total,
                cos_angle,
                sin_angle,
                velocity
            ])


            # j -> i
            edge_index.append([j, i])
            # edge_attr.append([-dx, -dy, -dt, dist_xy, dist_t, dist_total])

            edge_attr.append([
                -dx,
                -dy,
                -dt,
                dist_xy,
                dist_t,
                dist_total,
                -cos_angle,
                -sin_angle,
                velocity
            ])
    return (
        np.array(edge_index, dtype=np.int32),
        np.array(edge_attr, dtype=np.float32)
    )
    

def generate_nodes():
    NODE_FEAT_DIM = 19
    node_dir = EVENTS_OUT_DIR / "nodesPose"
    node_dir.mkdir(parents=True, exist_ok=True)
    # depth_dir = Path(r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\mvsec_outdoor_day_1_20Hz\mvsec_outdoor_day_1_20Hz\outdoor_day_1\depth_rectified")

    # dimensiones
    with h5py.File(DATA_PATH, "r") as f:
        img = f["davis/left/image_raw"]
        H, W = img.shape[1], img.shape[2]
    


    event_files = sorted(EVENTS_OUT_DIR.glob("*.h5"))

    for ev_file in event_files:

        idx = int(ev_file.stem)

        # depth_path = depth_dir / f"depth_metric_{idx:010d}.npy"

        # if depth_path.exists():
        #     depth = np.load(depth_path).astype(np.float32)
        # else:
        #     depth = np.zeros((H, W), dtype=np.float32)

        with h5py.File(ev_file, "r") as h5:
            evs = h5["myDataset"][:]

        if len(evs) == 0:
            np.savez(
                node_dir / f"graph_{idx:010d}.npz",
                feats=np.zeros((0, NODE_FEAT_DIM), dtype=np.float32),
                coords=np.zeros((0, 3), dtype=np.float32),
                edge_index=np.zeros((0, 2), dtype=np.int32),
                edge_attr=np.zeros((0, 9), dtype=np.float32)
            )
            continue

        events = np.zeros((len(evs), 4), dtype=np.float64)

        events[:, 0] = evs["x"].astype(np.float64)
        events[:, 1] = evs["y"].astype(np.float64)
        events[:, 2] = np.where(evs["p"] > 0, 1.0, -1.0)
        events[:, 3] = evs["ts"].astype(np.float64)

        if not np.all(np.diff(events[:, 3]) >= 0):
            events = events[np.argsort(events[:, 3])]

        feats, coords = build_nodes(
            events,
            # depth,
            H,
            W,
            cell_size=8
        )

        feats = np.concatenate([feats, coords], axis=1)

        edge_index, edge_attr = build_graph(coords, k=8)

        np.savez(
            node_dir / f"graph_{idx:010d}.npz",
            feats=feats,
            coords=coords,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
    print("Graphs (nodes + edges + depth) saved")
        



def main():
    # save_gt_timestamps()
    # split_events_by_timestamps()
    generate_nodes()


if __name__ == "__main__":
    main()



