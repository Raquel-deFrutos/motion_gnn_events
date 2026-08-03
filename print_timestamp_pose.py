import h5py
import numpy as np


gt_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\indoor_flying1_gt.hdf5"
DATA_PATH = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\indoor_flying1_data.hdf5"
output_txt = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\indoor_flying1\timestamps_pose.txt"

with h5py.File(DATA_PATH, "r") as f:
    img = f["davis/left/image_raw"]
    H, W = img.shape[1], img.shape[2]
    
print(H, W)

# with h5py.File(gt_path, "r") as f:
#     pose_ts = f["davis/left/pose_ts"][:]

# # Guardar con alta precisión
# np.savetxt(output_txt, pose_ts, fmt="%.18e")

# print(f"Guardados {len(pose_ts)} timestamps en:")
# print(output_txt)