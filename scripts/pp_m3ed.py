import os
import cv2
import h5py
import argparse
import numpy as np


def compute_rmap_m3ed_radtan(indir, fx, fy, cx, cy, k1, k2, p1, p2, side, H=720, W=1280):

    K_evs = np.zeros((3, 3))
    K_evs[0, 0] = fx
    K_evs[0, 2] = cx
    K_evs[1, 1] = fy
    K_evs[1, 2] = cy
    K_evs[2, 2] = 1
    dist_coeffs_evs = np.asarray([k1, k2, p1, p2])

    K_new_evs, _ = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))

    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float64")
    points = cv2.undistortPoints(coords.T[:, None, :], K_evs, dist_coeffs_evs, P=K_new_evs)
    rectify_map = points.reshape((H, W, 2))

    # Create rectify map for events
    h5outfile = os.path.join(indir, f"rectify_map_{side}.h5")
    ef_out = h5py.File(h5outfile, 'w')
    ef_out.clear()
    ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
    ef_out["rectify_map"][:] = rectify_map
    ef_out.close()

    return rectify_map, K_new_evs


def process_seq_m3ed(indir, side="left"):
    dirname = os.path.basename(indir)
    fnameh5 = os.path.join(indir, f"{dirname}_data.h5")

    print(f"\n\n M3ED: Computing rect-map for {fnameh5}, and undistorting images.")
    
    datain = h5py.File(fnameh5, 'r')
    H, W = 720, 1280
    fx, fy, cx, cy = datain["prophesee/left/calib/intrinsics"][:]
    k1, k2, p1, p2 = datain["prophesee/left/calib/distortion_coeffs"][:]

    print(f"fx, fy, cx, cy: {fx, fy, cx, cy}")
    print(f"k1, k2, p1, p2: {k1, k2, p1, p2}")

    _, Knew = compute_rmap_m3ed_radtan(indir, fx, fy, cx, cy, k1, k2, p1, p2, side, H=H, W=W)
    f = open(os.path.join(indir, f"calib_undist_{side}.txt"), 'w')
    f.write(f"{Knew[0,0]} {Knew[1,1]} {Knew[0,2]} {Knew[1,2]}")
    f.close()

    print(f"NEW intrinsics: {Knew[0,0], Knew[1,1], Knew[0,2], Knew[1,2]}")

    datain.close()
    print(f"Finshied processing M3ED {indir}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP M3ED data in dir")
    parser.add_argument(
        "--indir", help="Input directory.", default=""
    )
    args = parser.parse_args()

    process_seq_m3ed(args.indir)
