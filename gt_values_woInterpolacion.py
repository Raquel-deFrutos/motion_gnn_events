import numpy as np
import h5py
import pandas as pd
from scipy.spatial.transform import Rotation as R


# =========================
# UTILS
# =========================
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def left_jacobian_SO3_inv(phi):
    theta = np.linalg.norm(phi)
    I = np.eye(3)
    if theta < 1e-8:
        return I - 0.5 * skew(phi)
    a = phi / theta
    a_hat = skew(a)
    coeff1 = (theta / 2) / np.tan(theta / 2)
    coeff2 = 1 - coeff1
    return coeff1 * I + coeff2 * np.outer(a, a) - (theta / 2) * a_hat

def ts_to_seconds(ts_raw):
    ts_raw = np.asarray(ts_raw, dtype=np.float64)
    d = np.median(np.diff(ts_raw))

    if d > 1e6:
        scale = 1e-9
    elif d > 1e3:
        scale = 1e-6
    elif d > 1:
        scale = 1e-3
    else:
        scale = 1.0

    ts = ts_raw * scale
    return ts - ts[0]

def interp_safe(tq, tx, y):
    tq = np.clip(tq, tx[0], tx[-1])
    return np.interp(tq, tx, y)


# =========================
# MAIN FUNCTION
# =========================
def generate_gt_aligned(gt_path, timestamps_path, output_csv):

    print("Cargando GT...")
    with h5py.File(gt_path, "r") as f:
        T = f["davis/left/pose"][:]        # (N,4,4)
        ts_raw = f["davis/left/pose_ts"][:]

    ts = ts_to_seconds(ts_raw)

    # # =========================
    # # COMPUTE TWIST
    # # =========================
    # print("Calculando velocidades (twist)...")
    # V, W, t_mid = [], [], []

    # for i in range(len(T) - 1):
    #     dt = ts[i+1] - ts[i]
    #     if dt <= 0:
    #         continue

    #     T0, T1 = T[i], T[i+1]

    #     R0 = T0[:3, :3]
    #     R1 = T1[:3, :3]
    #     p0 = T0[:3, 3]
    #     p1 = T1[:3, 3]

    #     R_rel = R0.T @ R1
    #     phi = R.from_matrix(R_rel).as_rotvec()
    #     w = phi / dt

    #     p_rel = R0.T @ (p1 - p0)
    #     # v = left_jacobian_SO3_inv(phi) @ (p_rel / dt)
    #     v = p_rel / dt

    #     V.append(v)
    #     W.append(w)
    #     t_mid.append(0.5 * (ts[i] + ts[i+1]))

    # V = np.array(V)
    # W = np.array(W)
    # t_mid = np.array(t_mid)
    
    #     # =========================
    # # COMPUTE DELTA POSE
    # # =========================
    print("Calculando pose relativa...")

    dP = []
    dR = []

    for i in range(len(T) - 1):

        T0, T1 = T[i], T[i+1]

        R0 = T0[:3, :3]
        R1 = T1[:3, :3]

        p0 = T0[:3, 3]
        p1 = T1[:3, 3]

        # traslación relativa en el sistema de referencia de la cámara

        p_rel = R0.T @ (p1 - p0)

        # rotación relativa en axis-angle

        R_rel = R0.T @ R1
        rotvec = R.from_matrix(R_rel).as_rotvec()

        dP.append(p_rel)
        dR.append(rotvec)

    dP = np.array(dP)
    dR = np.array(dR)




    # =========================
    # SAVE CSV
    # =========================
    print("Guardando CSV...")

    df = pd.DataFrame({
        "frame_id": np.arange(len(dP)),
        "t_raw": ts_raw[:-1],
        "t_s": ts[:-1],
        "dx": dP[:, 0],
        "dy": dP[:, 1],
        "dz": dP[:, 2],
        "rx": dR[:, 0],
        "ry": dR[:, 1],
        "rz": dR[:, 2]
    })

    
    df.to_csv(output_csv, index=False)

    print("GT guardado correctamente en:", output_csv)

    print("\nSTD pose relativa:")
    print(df[["dx", "dy", "dz", "rx", "ry", "rz"]].std())


# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":

    # gt_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1_gt.hdf5"
    # timestamps_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\timestamps_depth.txt"
    # output_csv = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\gt_aligned_woInterp_PoseRe.csv"
    
    gt_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\indoor_flying1_gt.hdf5"
    timestamps_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\timestamps_depth.txt"
    output_csv = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\indoor_flying1\gt_aligned_woInterp_PoseRe.csv"

    generate_gt_aligned(gt_path, timestamps_path, output_csv)