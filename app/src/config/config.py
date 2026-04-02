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
    viola_jones_path: str = os.path.join(DATA_PATH, "ViolaJones")
    faces_path: str = os.path.join(viola_jones_path, "face_images")
    no_faces_path: str = os.path.join(viola_jones_path, "no_faces")

    wider_path: str = os.path.join(DATA_PATH, "others", "WIDER_train", "images")
    # dataset_path: str = os.path.join("data", "NaturalImages")
    # all_path: str = os.path.join(dataset_path, "all")
    # train_path: str = os.path.join(dataset_path, "img_train")
    # train_f_path: str = os.path.join(dataset_path, "img_train_f")
    # val_path: str = os.path.join(dataset_path, "img_val")
    # val_f_path: str = os.path.join(dataset_path, "img_val_f")
    # test_path: str = os.path.join(dataset_path, "img_test")
    # test_f_path: str = os.path.join(dataset_path, "img_test_f")

    faces_np_path: str = os.path.join(viola_jones_path, "faces.npy")


    # ============================================================================
    MODELS_PATH: str = os.path.join("..", "models")
    best_cnn_model_path: str = os.path.join(MODELS_PATH, "best_cnn_model.ckpt")
    best_ori_model_path: str = os.path.join(MODELS_PATH, "best_ori_model.ckpt")
    
    cv_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades")
    computed_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades_computed")

    val_split: float = 0.1
    test_split: float = 0.1

    aug_prob: float = 0.75

    max_files: int = 100
    gray_scale: bool = True
    

    # =========================
    crop_size: int = 24
    stride: int = 4
    subsample_factor: float = 0.8
    normalize_window: int = 3
    
    max_features_per_stage: int = 200
    max_stages: int = 50
    objective_fpr: float = 0.005

    # =========================
    batch_size: int = 128
    patience: int = 10
    num_workers: int = 4
    lr: float = 5e-5
    weight_decay: float = 1e-4
    epochs: int = 100

    def __post_init__(self):
        make_dirs([
            self.faces_path, self.no_faces_path,
            # self.val_path, self.test_path, 
            # self.train_f_path, self.val_f_path, self.test_f_path, 
            self.MODELS_PATH,
            self.computed_haar_cascades
        ])

