import os
from maikol_utils.file_utils import make_dirs
from dataclasses import dataclass

@dataclass
class Configuration:
    """Configuration dataclass to hold application settings."""
    # =========================
    seed: int = 42
    test_size: float = 0.1

    # =========================
    DATA_PATH: str = os.path.join("..", "data")
    dataset_classes_path: str = os.path.join(DATA_PATH, "dataset_classes.json")
    wider_path: str = os.path.join(DATA_PATH, "others", "WIDER_train", "images")
    viola_jones_path: str = os.path.join(DATA_PATH, "ViolaJones")

    faces_original_path: str = os.path.join(viola_jones_path, "face_images")
    faces_all_path: str = os.path.join(faces_original_path, "all")
    faces_passed_path: str = os.path.join(faces_original_path, "cv_passed")
    faces_test_path: str = os.path.join(faces_original_path, "test")
    faces_train_path: str = os.path.join(faces_original_path, "train")
    faces_path_merge: str = os.path.join(viola_jones_path, "face_images_merge")
    faces_vpc_path: str = os.path.join(viola_jones_path, "face_images_vpc")
    faces_cv_passed_path: str = os.path.join(faces_original_path, "cv_passed")
    no_faces_path: str = os.path.join(viola_jones_path, "no_faces")
    no_faces_all_path: str = os.path.join(no_faces_path, "all")
    no_faces_crops_path: str = os.path.join(no_faces_path, "crops")

    faces_np_path: str = os.path.join(viola_jones_path, "faces.npy")

    # ============================================================================
    MODELS_PATH: str = os.path.join("..", "models")
    best_cnn_model_path: str = os.path.join(MODELS_PATH, "best_cnn_model.ckpt")
    best_ori_model_path: str = os.path.join(MODELS_PATH, "best_ori_model.ckpt")
    
    cv_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades")
    computed_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades_computed")


    # =========================
    use_vpc_faces: bool = False


    max_cpu_cores: int = 32

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
    stage_target_fpr: float = 0.5
    target_fpr: float = 0.005
    target_tpr: float = 0.985
    

    def __post_init__(self):
        make_dirs([
            self.faces_original_path, self.no_faces_path,
            self.no_faces_all_path,
            self.faces_train_path, self.faces_test_path,
            self.faces_all_path,
            # self.val_path, self.test_path, 
            # self.train_f_path, self.val_f_path, self.test_f_path, 
            self.MODELS_PATH,
            self.computed_haar_cascades
        ])

