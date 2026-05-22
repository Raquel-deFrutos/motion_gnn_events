import h5py
import numpy as np
from pathlib import Path

# ==========================================================
# PATHS (ajusta si hace falta)
# ==========================================================
GT_PATH = Path(r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1_gt.hdf5")
OUTPUT_DIR = Path(r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1")  

FLOW_DIR = OUTPUT_DIR / "optical_flow"
FLOW_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# SAVE FLOW PER FRAME
# ==========================================================
def save_flow_per_frame():
    with h5py.File(GT_PATH, "r") as f:
        flow = f["davis/left/flow_dist"][:]  
        # shape: (N, 2, H, W)

    N = flow.shape[0]

    for i in range(N):
        flow_i = flow[i].astype(np.float32)

        np.save(
            FLOW_DIR / f"flow_{i:010d}.npy",
            flow_i
        )

    print(f"[OK] Saved {N} flow frames in: {FLOW_DIR}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    save_flow_per_frame()

