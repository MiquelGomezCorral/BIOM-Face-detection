"""Main file for scripts with arguments and call other functions."""

import argparse
from maikol_utils.other_utils import args_to_dataclass

from src.config import Configuration
from scripts import start_detect_camera

def cmd_detect_camera(args: argparse.Namespace):
    """Call detect_camera with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    start_detect_camera(CONFIG)
    

# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       detect-camera
    # ======================================================================================
    p_camera = subparsers.add_parser("detect-camera", help="Open the camera and detect faces in real-time")
    p_camera.add_argument("-c", "--crop-size", type=int, default=24, help="Crop size for face detection (default: 24)")
    p_camera.add_argument("-s", "--stride", type=int, default=4, help="Stride for face detection (default: 4)")
    p_camera.set_defaults(func=cmd_detect_camera)


    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)