import h5py
import numpy as np

gt_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1_gt.hdf5"
output_txt = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\timestamps_pose.txt"

with h5py.File(gt_path, "r") as f:
    pose_ts = f["davis/left/pose_ts"][:]

# Guardar con alta precisión
np.savetxt(output_txt, pose_ts, fmt="%.18e")

print(f"Guardados {len(pose_ts)} timestamps en:")
print(output_txt)