import os
from maikol_utils.file_utils import make_dirs
from dataclasses import dataclass

@dataclass
class Configuration:
    """Configuration dataclass to hold application settings."""
    # =========================
    seed: int = 42

    # =========================
    DATA_PATH: str = os.path.join("..", "data")
    dataset_classes_path: str = os.path.join(DATA_PATH, "dataset_classes.json")
    wider_path: str = os.path.join(DATA_PATH, "others", "WIDER_train", "images")
    viola_jones_path: str = os.path.join(DATA_PATH, "ViolaJones")

    faces_path: str = os.path.join(viola_jones_path, "face_images")
    no_faces_path: str = os.path.join(viola_jones_path, "no_faces")

    faces_np_path: str = os.path.join(viola_jones_path, "faces.npy")

    # ============================================================================
    MODELS_PATH: str = os.path.join("..", "models")
    best_cnn_model_path: str = os.path.join(MODELS_PATH, "best_cnn_model.ckpt")
    best_ori_model_path: str = os.path.join(MODELS_PATH, "best_ori_model.ckpt")
    
    cv_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades")
    computed_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades_computed")


    # =========================
    max_cpu_cores: int = 16

    force_features: bool = False
    resume_training: bool = True
    crop_size: int = 24
    stride: int = 4
    detect_width: int = 320

    max_bg_samples: int = 20_000
    max_faces: int = 10_000
    stop_check_interval: int = 100
    
    subsample_factor: float = 0.8
    normalize_window: int = 3
    
    max_features_per_stage: int = 200
    max_stages: int = 50
    target_fpr: float = 0.005
    

    def __post_init__(self):
        make_dirs([
            self.faces_path, self.no_faces_path,
            # self.val_path, self.test_path, 
            # self.train_f_path, self.val_f_path, self.test_f_path, 
            self.MODELS_PATH,
            self.computed_haar_cascades
        ])

