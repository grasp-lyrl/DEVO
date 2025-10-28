import os
from pathlib import Path
import torch
from devo.config import cfg

from utils.load_utils import load_gt_us, fpv_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference

from devo.plot_utils import save_trajectory_tum_format

from utils.scale_utils import run_scale_estimation
import numpy as np

H, W = 260, 346

T_CAM_IMU = np.array([
    [0.9999711474430529, 0.0013817010649267755, -0.007469617365767657, 0.00018050225881571712],
    [-0.0014085305353606873, 0.9999925720306121, -0.00358774655345255, -0.004316353415695194],
    [0.007464604688444933, 0.0035981642219379494, 0.9999656658561218, -0.027547385763471585],
    [0.0, 0.0, 0.0, 1.0]
])

def load_uzhfpv_imu(filepath):
    """
    ['idx', 'timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']
    """
    data = []
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 8:
            try:
                data.append([
                    float(parts[1]),  # timestamp
                    float(parts[2]),  # wx
                    float(parts[3]),  # wy
                    float(parts[4]),  # wz
                    float(parts[5]),  # ax
                    float(parts[6]),  # ay
                    float(parts[7])   # az
                ])
            except Exception as e:
                print(f"failed to parse line {i}: {e}")
        else:
            print(f"skipping line {i}: got {len(parts)} parts (expected 8)")
    data = np.array(data)
    imu_ts = data[:, 0]
    imu_data = data[:, 1:]
    
    return imu_data, imu_ts

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None, 
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False, viz_flow=False,
             scale_poses_imu=True):
    dataset_name = "fpv_evs"

    if config is None:
        config = cfg
        config.merge_from_file("config/eval_fpv.yaml")
        
    scenes = open(split_file).read().split()

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        has_gt = "_with_gt" in scene
        if not has_gt:            
            datapath_val = os.path.join(datapath, scene)
            outfolder = f"results/{scene}"
            
            for trial in range(trials):
            
                # run the slam system
                traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                          iterator=fpv_evs_iterator(datapath_val, stride=stride, timing=timing, H=H, W=W, tss_gt_us=None),
                                          timing=timing, H=H, W=W, viz_flow=viz_flow)
            
                # save estimated trajectory
                Path(f"{outfolder}").mkdir(exist_ok=True)
                save_trajectory_tum_format((traj_est, tstamps), f"{outfolder}/{scene}_Trial{trial+1:02d}.txt")
            
            # print(f"Not yet implemented, skipping!")
            # continue
        else:

            print(f"Eval on {scene}")
            results_dict_scene[scene] = []

            for trial in range(trials):
                # estimated trajectory
                datapath_val = os.path.join(datapath, scene)
                tss_traj_us, traj_hf = load_gt_us(os.path.join(datapath_val, f"stamped_groundtruth_us_cam.txt"))

                # run the slam system
                traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                            iterator=fpv_evs_iterator(datapath_val, stride=stride, timing=timing, H=H, W=W, tss_gt_us=tss_traj_us),
                                            timing=timing, H=H, W=W, viz_flow=viz_flow)
                
                # run scale estimation with imu
                if scale_poses_imu:
                    # load imu data
                    imu_path = os.path.join(datapath_val, f"imu_offset.txt")
                    imu_data, imu_ts = load_uzhfpv_imu(imu_path)
                    traj_est, _, _ = run_scale_estimation(traj_est.copy(), tstamps.copy(),
                                                          imu_data, imu_ts,
                                                          T_CAM_IMU)

                # do evaluation 
                data = (traj_hf, tss_traj_us, traj_est, tstamps)
                hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
                all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                    plot=plot, save=save, return_figure=return_figure, stride=stride,
                                                                    expname=args.expname)
                
                if viz_flow:
                    viz_flow_inference(outfolder, flowdata)
                
            print(scene, sorted(results_dict_scene[scene]))

            # write output to file with timestamp
            write_raw_results(all_results, outfolder)
            results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/eval_fpv.yaml")
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/fpv/fpv_val.txt")
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--side', type=str, default="left")
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_fpv_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)

    args.save_trajectory = True
    args.plot = True
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, \
                        stride=args.stride, viz_flow=args.viz_flow)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
