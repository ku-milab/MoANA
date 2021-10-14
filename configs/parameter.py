import argparse
import time
from utils import *
import munch
import os

def get_parameters():

    # Argparse setting
    parser = argparse.ArgumentParser()
    args = read_json("/home/jhsohn/Desktop/TIP_I2C/configs/config.json")

    # GPU setting
    parser.add_argument('--n_gpu', type=int, default=args["n_gpu"])
    parser.add_argument('--device', type=str, default=args["devices"])
    parser.add_argument('--parallel', type=bool, default=args["parallel"]) #False
    parser.add_argument('--num_workers', type=int, default=args["num_workers"]) # 8

    ### Project setting
    # parser.add_argument('--time', type=str, default=time.strftime('%Y-%m-%d(%H:%M:%S)', time.localtime(time.time())))
    parser.add_argument('--time', type=str, default=time.strftime('%m-%d(%H:%M:%S)', time.localtime(time.time())))
    parser.add_argument('--project_name', type=str, default=args["project_name"])
    parser.add_argument('--mode', type=str, default=args["mode"], choices=['train', 'test'])

    ### Dataset setting
    parser.add_argument('--dataset_name', type=str, default=args["dataset_name"])
    parser.add_argument('--resize_size', type=int, default=args["resize_size"])
    parser.add_argument('--crop_size', type=int, default=args["crop_size"])
    parser.add_argument('--proxy', type=bool, default=args["proxy"])

    ### Model setting
    parser.add_argument('--model_type', type=str, default=args["model_type"])
    parser.add_argument('--batch_size', type=int, default=args["batch_size"])
    parser.add_argument('--epoch', type=int, default=args["epoch"], help='how many times to update')
    parser.add_argument('--backbone_lr', type=float, default=args["backbone_lr"])
    parser.add_argument('--lr', type=float, default=args["lr"])
    parser.add_argument('--lr_gamma', type=float, default=args["lr_gamma"])
    parser.add_argument('--lr_milestones', type=list, default=args["lr_milestones"])
    parser.add_argument('--momentum', type=float, default=args["momentum"])
    parser.add_argument('--weight_decay', type=float, default=args["weight_decay"])
    parser.add_argument('--local_factor', type=float, default=args["local_factor"])
    parser.add_argument('--global_factor', type=float, default=args["global_factor"])
    parser.add_argument('--local_seed_num', type=float, default=args["local_seed_num"])
    parser.add_argument('--layer_optim', type=bool, default=args["layer_optim"])

    ### CAM setting
    parser.add_argument('--iou_threshold_list', type=list, default=args["iou_threshold_list"])
    parser.add_argument('--cam_curve_interval', type=float, default=args["cam_curve_interval"])
    parser.add_argument('--multi_contour_eval', type=bool, default=args["multi_contour_eval"])
    parser.add_argument('--multi_iou_eval', type=bool, default=args["multi_iou_eval"])

    ### Path setting
    parser.add_argument('--model_save_path', type=str, default=args["model_save_path"])
    parser.add_argument('--dataset_path', type=str, default=args["dataset_path"])
    parser.add_argument('--metadata_path', type=str, default=args["metadata_path"])
    parser.add_argument('--mask_path', type=str, default=args["mask_path"])
    parser.add_argument('--cam_path', type=str, default=args["cam_path"])
    parser.add_argument('--cam_test_path', type=str, default=args["cam_test_path"])

    return parser.parse_args()
