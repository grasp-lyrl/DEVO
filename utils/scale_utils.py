import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def fit_spline_to_poses(poses, timestamps, smoothing_spline=0.0):
    positions = poses[:, :3]
    quaternions = poses[:, 3:7]
    n = len(timestamps)
    k_pos = min(3, max(1, n - 1))
    spline_x = UnivariateSpline(
        timestamps, positions[:, 0], k=k_pos, s=smoothing_spline
    )
    spline_y = UnivariateSpline(
        timestamps, positions[:, 1], k=k_pos, s=smoothing_spline
    )
    spline_z = UnivariateSpline(
        timestamps, positions[:, 2], k=k_pos, s=smoothing_spline
    )
    k_q = min(3, max(1, n - 1))
    spline_qx = UnivariateSpline(timestamps, quaternions[:, 0], k=k_q, s=0)
    spline_qy = UnivariateSpline(timestamps, quaternions[:, 1], k=k_q, s=0)
    spline_qz = UnivariateSpline(timestamps, quaternions[:, 2], k=k_q, s=0)
    spline_qw = UnivariateSpline(timestamps, quaternions[:, 3], k=k_q, s=0)

    return {
        "pos_x": spline_x,
        "pos_y": spline_y,
        "pos_z": spline_z,
        "quat_x": spline_qx,
        "quat_y": spline_qy,
        "quat_z": spline_qz,
        "quat_w": spline_qw,
    }

def estimate_scale_bias(spline_accels, imu_accels, with_bias=True):
    if not with_bias:
        spline_norms = np.linalg.norm(spline_accels, axis=1)
        imu_norms = np.linalg.norm(imu_accels, axis=1)
        scale = np.dot(spline_norms, imu_norms) / np.dot(spline_norms, spline_norms)
        bias = np.zeros(3)
        return float(scale), bias

    def residuals(params):
        log_s = params[0]
        scale = np.exp(log_s)
        bias = params[1:4]
        pred = scale * spline_accels + bias
        return (pred - imu_accels).ravel()

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    res = least_squares(residuals, x0)
    scale = float(np.exp(res.x[0]))
    bias = res.x[1:4].astype(float)
    return scale, bias

def rotate_imu_to_camera(imu_data, T_imu_cam):
    R_imu_cam = np.asarray(T_imu_cam)[:3, :3]
    R_cam_imu = R_imu_cam.T
    wx, wy, wz = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]
    ax, ay, az = imu_data[:, 3], imu_data[:, 4], imu_data[:, 5]
    wx_cam = R_cam_imu[0, 0] * wx + R_cam_imu[0, 1] * wy + R_cam_imu[0, 2] * wz
    wy_cam = R_cam_imu[1, 0] * wx + R_cam_imu[1, 1] * wy + R_cam_imu[1, 2] * wz
    wz_cam = R_cam_imu[2, 0] * wx + R_cam_imu[2, 1] * wy + R_cam_imu[2, 2] * wz
    ax_cam = R_cam_imu[0, 0] * ax + R_cam_imu[0, 1] * ay + R_cam_imu[0, 2] * az
    ay_cam = R_cam_imu[1, 0] * ax + R_cam_imu[1, 1] * ay + R_cam_imu[1, 2] * az
    az_cam = R_cam_imu[2, 0] * ax + R_cam_imu[2, 1] * ay + R_cam_imu[2, 2] * az
    imu_cam = np.column_stack((wx_cam, wy_cam, wz_cam, ax_cam, ay_cam, az_cam))
    return imu_cam

def run_scale_estimation(poses, pose_ts, imu_data, imu_ts, T_cam_imu):
    """
    Scale trajectory using IMU data.
    Args:
        poses: (N, 7) array of estimated poses (x,y,z,qx,qy,qz,qw)
        pose_ts: (N,) array of pose timestamps in microseconds
        imu_data: (M, 6) array of IMU data (wx,wy,wz,ax,ay,az)
        imu_ts: (M,) array of IMU timestamps in microseconds
        T_cam_imu: (4,4) transformation matrix from IMU to camera frame
    Returns:
        scaled_poses: (N, 3) array of scaled positions
        scales: (N,) array of scale factors
        biases: (N, 3) array of estimated accelerometer biases
    """
    poses = np.asarray(poses)
    pose_ts *= 1e-6 # convert to seconds
    imu_ts *= 1e-6 # convert to seconds
    imu_data = np.asarray(imu_data)

    # rotate IMU measurements to camera frame
    T_imu_cam = np.linalg.inv(T_cam_imu)
    imu_cam = rotate_imu_to_camera(imu_data, T_imu_cam)
    acc_cam = imu_cam[:, 3:6]

    # gather IMU measurements in stationary period
    stationary_mask = imu_ts <= (pose_ts[0] - 2.0) # 2 seconds is arbitrary
    if np.sum(stationary_mask) > 0:
        gravity_cam0 = acc_cam[stationary_mask].mean(axis=0)
        print("estimated gravity vector in camera frame:", gravity_cam0)
    else:
        gravity_cam0 = np.array([0.0, -9.81, 0.0])
        print(
            "no stationary IMU data found; using default gravity vector:", gravity_cam0
        )

    # align pose and imu timestamps
    print("imu ts from ", imu_ts[0], "to", imu_ts[-1])
    print("pose ts from ", pose_ts[0], "to", pose_ts[-1])
    first_imu_ts = imu_ts[0]
    removed = 0
    while len(pose_ts) > 0 and pose_ts[0] < first_imu_ts:
        pose_ts = pose_ts[1:]
        poses = poses[1:]
        removed += 1
    if removed > 0:
        print(f"removed {removed} poses/timestamps to align with IMU start time.")
    if len(poses) == 0:
        raise ValueError("no poses left after alignment!")
    
    num_poses = len(poses)

    # incremental scale estimation
    min_samples = 100
    window_size = 100
    use_window = False
    if not use_window:
        window_size = num_poses
    
    scales = np.ones(num_poses)
    biases = np.zeros((num_poses, 3))
    scaled_poses = poses.copy()
    unscaled_poses = []
    spline_init = False

    for i in range(num_poses):
        if i < min_samples:
            unscaled_poses.append(poses[i].copy())
            continue

        start_idx = max(0, i + 1 - window_size)
        ts_i = pose_ts[start_idx : i + 1]
        poses_i = poses[start_idx : i + 1]

        # get imu within window
        imu_mask = (imu_ts >= ts_i[0]) & (imu_ts <= ts_i[-1])
        if len(ts_i) < 2 or np.sum(imu_mask) < 1:
            # not enough data, keep unscaled or last known scale
            scaled_poses[i, :3] = poses[i, :3] * (scales[i - 1] if i > 0 else 1.0)
            scales[i] = scales[i - 1] if i > 0 else 1.0
            biases[i] = biases[i - 1] if i > 0 else np.zeros(3)
            continue

        imu_sub_ts = imu_ts[imu_mask]
        imu_sub_acc = acc_cam[imu_mask]

        # fit spline to poses in window
        spl = fit_spline_to_poses(poses_i, ts_i, smoothing_spline=1.0)

        # evaluate spline accelerations at imu timestamps
        spline_accels_cam0 = np.column_stack(
            [
                spl["pos_x"].derivative(n=2)(imu_sub_ts),
                spl["pos_y"].derivative(n=2)(imu_sub_ts),
                spl["pos_z"].derivative(n=2)(imu_sub_ts),
            ]
        )

        # evaluate quaternion splines
        spline_quats = np.column_stack(
            [
                spl["quat_x"](imu_sub_ts),
                spl["quat_y"](imu_sub_ts),
                spl["quat_z"](imu_sub_ts),
                spl["quat_w"](imu_sub_ts),
            ]
        )
        # normalize quaternions
        q_norms = np.linalg.norm(spline_quats, axis=1, keepdims=True)
        spline_quats = spline_quats / np.maximum(q_norms, 1e-12)
        # convert to rotation matrices
        rot_mats = R.from_quat(spline_quats).as_matrix()  # (M,3,3) R_cam0_cami

        # rotate spline accelerations into IMU frame
        # spline_accels_cami = R_cami_cam0 @ spline_accels_cam0
        spline_accels_cami = np.einsum(
            "nij,nj->ni", rot_mats.transpose(0, 2, 1), spline_accels_cam0
        )

        # remove gravity
        # gravity_rot = R_cami_cam0 @ gravity_cam0
        gravity_rot = np.einsum("nij,j->ni", rot_mats.transpose(0, 2, 1), gravity_cam0)
        imu_acc_no_g = imu_sub_acc - gravity_rot

        # estimate scale and bias for this window
        scale_i, bias_i = estimate_scale_bias(
            spline_accels_cami, imu_acc_no_g, with_bias=True
        )

        # exponential smoothing of scale
        if i > 0:
            sf = 0.99
            scale_i = sf * scales[i - 1] + (1.0 - sf) * scale_i

        scales[i] = scale_i
        biases[i] = bias_i

        if not spline_init:
            # scale all previous unscaled poses
            unscaled_poses = np.array(unscaled_poses)
            if unscaled_poses.size > 0:
                unscaled_poses[:, :3] = unscaled_poses[:, :3] * scale_i
                scaled_poses[: len(unscaled_poses), :3] = unscaled_poses[:, :3]
            spline_init = True

        scaled_poses[i, :3] = poses[i, :3] * scale_i

        print(
            f"pose {i}: estimated scale={scale_i:.4f}, bias={bias_i}, num_imu={np.sum(imu_mask)}"
        )

    return scaled_poses, scales, biases
