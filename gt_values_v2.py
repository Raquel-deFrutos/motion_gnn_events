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


    # =========================
    # COMPUTE RELATIVE POSE
    # =========================
    print("Calculando pose relativa...")

    dP = []
    dR = []
    t_pose = []

    for i in range(len(T)-1):

        T0 = T[i]
        T1 = T[i+1]

        # transformación relativa cámara
        T_rel = np.linalg.inv(T0) @ T1

        R_rel = T_rel[:3,:3]
        t_rel = T_rel[:3,3]

        # rotación axis-angle
        rotvec = R.from_matrix(R_rel).as_rotvec()

        dP.append(t_rel)
        dR.append(rotvec)

        t_pose.append(ts[i])

    dP = np.array(dP)
    dR = np.array(dR)
    t_pose = np.array(t_pose)
    
    #     # =========================
    # # COMPUTE DELTA POSE
    # # =========================
    # print("Calculando delta pose...")

    # dP = []   # traslación relativa
    # dR = []   # rotación (axis-angle)
    # t_mid = []

    # for i in range(len(T) - 1):
    #     dt = ts[i+1] - ts[i]
    #     if dt <= 0:
    #         continue

    #     T0, T1 = T[i], T[i+1]

    #     R0 = T0[:3, :3]
    #     R1 = T1[:3, :3]
    #     p0 = T0[:3, 3]
    #     p1 = T1[:3, 3]

    #     # 👉 transformación relativa correcta
    #     R_rel = R0.T @ R1
    #     t_rel = R0.T @ (p1 - p0)

    #     # rotación en axis-angle
    #     rotvec = R.from_matrix(R_rel).as_rotvec()

    #     dP.append(t_rel)
    #     dR.append(rotvec)
    #     t_mid.append(0.5 * (ts[i] + ts[i+1]))

    # dP = np.array(dP)
    # dR = np.array(dR)
    # t_mid = np.array(t_mid)

    # =========================
    # LOAD FRAME TIMESTAMPS
    # =========================
    print("Cargando timestamps de frames...")
    t_frames_raw = np.loadtxt(timestamps_path)
    t_frames = ts_to_seconds(t_frames_raw)

    # =========================
    # INTERPOLATION
    # =========================
    idx = np.searchsorted(t_pose, t_frames[:-1])

    idx = np.clip(idx, 0, len(dP)-1)

    dx = dP[idx,0]
    dy = dP[idx,1]
    dz = dP[idx,2]

    rx = dR[idx,0]
    ry = dR[idx,1]
    rz = dR[idx,2]

    # dx = interp_safe(t_frames, t_mid, dP[:, 0])
    # dy = interp_safe(t_frames, t_mid, dP[:, 1])
    # dz = interp_safe(t_frames, t_mid, dP[:, 2])

    # rx = interp_safe(t_frames, t_mid, dR[:, 0])
    # ry = interp_safe(t_frames, t_mid, dR[:, 1])
    # rz = interp_safe(t_frames, t_mid, dR[:, 2])
    


    # =========================
    # SAVE CSV
    # =========================
    print("Guardando CSV...")

    df = pd.DataFrame({
        "frame_id": np.arange(len(t_frames)-1),
        "t_raw": t_frames_raw[:-1],
        "t_s": t_frames[:-1],

        "dx": dx,
        "dy": dy,
        "dz": dz,

        "rx": rx,
        "ry": ry,
        "rz": rz
    })
    
    # df = pd.DataFrame({
    # "frame_id": np.arange(len(t_frames) - 1),
    # "t_raw": t_frames_raw[:-1],
    # "t_s": t_frames[:-1],
    # "dx": dx[:-1],
    # "dy": dy[:-1],
    # "dz": dz[:-1],
    # "rx": rx[:-1],
    # "ry": ry[:-1],
    # "rz": rz[:-1]
    # })
    
    df.to_csv(output_csv, index=False)

    print("GT guardado correctamente en:", output_csv)

    # Debug rápido
    print("\nSTD velocidades:")
    print(df[["t_x","t_y","t_z","w_x","w_y","w_z"]].std())


# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":

    gt_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1_gt.hdf5"
    timestamps_path = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\timestamps_depth.txt"
    output_csv = r"C:\Users\Raquel\Documents\Doct\Algoritmo\MVSEC\outdoor_day1\gt_aligned_v2.csv"

    generate_gt_aligned(gt_path, timestamps_path, output_csv)