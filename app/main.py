"""Main file for scripts with arguments and call other functions."""

import argparse
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from src.config import Configuration
from scripts import start_detect_camera, train_viola_jones_stages
from scripts.generate_labels import start_generate_filter


def cmd_detect_camera(args: argparse.Namespace):
    """Call detect_camera with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    start_detect_camera(CONFIG)
    
def cmd_train_viola_jones_stages(args: argparse.Namespace):
    """Call train_viola_jones_stages with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("TRAINING VIOLA-JONES CASCADE CLASSIFIER", sep_type="START")
    train_viola_jones_stages(CONFIG)
    print_separator("END TRAINING VIOLA-JONES CASCADE CLASSIFIER", sep_type="START")

def cmd_generate_filter(args: argparse.Namespace):
    """Call generate filter with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("GENERATING FILTERED LABELS", sep_type="START")
    start_generate_filter(CONFIG)
    print_separator("END GENERATING FILTERED LABELS", sep_type="START")


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
    p_camera.add_argument("-s", "--stride", type=int, default=1, help="Stride for face detection (default: 4)")
    p_camera.set_defaults(func=cmd_detect_camera)


    # ======================================================================================
    #                                       train-viola-jones
    # ======================================================================================
    p_train = subparsers.add_parser("train_vj", help="Train a Viola-Jones face detector")
    p_train.add_argument("-cpus", "--max_cpu_cores", type=int, default=32, help="Maximum number of CPU cores to use (default: 16ye)")


    p_train.add_argument("-ff", "--force_features", default=False, action="store_true", help="Force the use of all features (default: False)")
    p_train.add_argument("-rt", "--resume_training", default=True, action="store_false", help="Resume training from existing stages (default: True)")
    p_train.add_argument("-mf", "--max_faces", type=int, default=-1, help="Maximum number of face samples (default: -1 for no limit)")
    p_train.add_argument("-mb", "--max_bg_samples", type=int, default=20000, help="Maximum number of background samples (default: 20000)")
    p_train.add_argument("-sci", "--stop_check_interval", type=int, default=100, help="Interval at which to check for stopping conditions (default: 100)")

    p_train.add_argument("-dw", "--detect_width", type=int, default=320, help="Width of the detection window (default: 320)")


    p_train.add_argument("-sf", "--subsample_factor", type=float, default=0.8, help="Subsampling factor for negative samples (default: 0.8)")
    p_train.add_argument("-n", "--normalize_window", type=int, default=3, help="Window size for feature normalization (default: 3)")
    p_train.add_argument("-mfs", "--max_features_per_stage", type=int, default=500, help="Maximum number of features per stage (default: 200)")
    p_train.add_argument("-stf", "--stage_target_fpr", type=float, default=0.5, help="Target false positive rate for each stage (default: 0.5)")
    p_train.add_argument("-fpr", "--target_fpr", type=float, default=0.005, help="Target false positive rate for training (default: 0.005)")
    p_train.add_argument("-tpr", "--target_tpr", type=float, default=0.985, help="Target true positive rate for training (default: 0.985)")
    p_train.add_argument("-ms", "--max_stages", type=int, default=50, help="Maximum number of stages to train (default: 50)")
    p_train.add_argument("-s", "--stride", type=int, default=1, help="Stride for face detection (default: 4)")

    p_train.add_argument('-vpc', '--use_vpc_faces', default=False, action='store_true', help="Use the VPC faces dataset instead of WIDER (default: False)")


    p_train.set_defaults(func=cmd_train_viola_jones_stages)

    # ======================================================================================
    #                                       generate-filter
    # ======================================================================================
    p_filter = subparsers.add_parser("generate-filter", help="Generate the filtered label view")
    p_filter.set_defaults(func=cmd_generate_filter)
    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)