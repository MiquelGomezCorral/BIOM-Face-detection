import os
from maikol_utils.file_utils import make_dirs
from dataclasses import dataclass, field

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
    no_faces_train_path: str = os.path.join(no_faces_path, "train")
    no_faces_test_path: str = os.path.join(no_faces_path, "test")
    no_faces_crops_path: str = os.path.join(no_faces_path, "crops")

    faces_np_path: str = os.path.join(viola_jones_path, "faces.npy")

    # ============================================================================
    MODELS_PATH: str = os.path.join("..", "models")
    
    cv_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades")
    computed_haar_cascades: str = os.path.join(MODELS_PATH, "haar_cascades_computed_best_balanced")
    computed_haar_cascades_name: str = "haar_cascade_stage_38_fpr_0.0002.xml"
    computed_haar_cascades_path: str = os.path.join(computed_haar_cascades, computed_haar_cascades_name)


    # =========================
    use_vpc_faces: bool = False


    max_cpu_cores: int = 32

    force_features: bool = False
    resume_training: bool = True
    preserve_fp: bool = True
    crop_size: int = 24
    feature_stride: int = 1
    feature_edge_margin: int = 0
    stride: int = 4
    detect_width: int = 1080
    camera_window_width: int = 1920
    camera_window_height: int = 1080
    halve_size_factor: int = 1

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
    
    use_progresive_fpr: bool = True
    stage_fpr_schedule: list = field(default_factory=lambda: [
        (1, 2, 0.15),
        (3, 5, 0.25),
        (6, 10, 0.40),
        (11, None, 0.50),
    ])
    
    use_augmentation: bool = True
    faces_aug: dict = field(default_factory=lambda: {
        "contrast": 0.4,
        "light": 0.3,
        "blur": 0.2,
        "noise": 0.2,
    })
    bg_aug: dict = field(default_factory=lambda: {
        "noise": 0.8,
        "light": 0.6,
        "blur": 0.4,
        "contrast": 0.1,
    })

    include_square_features: bool = True

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

        
        # self.computed_haar_cascades_path = os.path.join(self.computed_haar_cascades, self.computed_haar_cascades_name)
        self.computed_haar_cascades_path = os.path.join(self.MODELS_PATH, 'haar_cascades_computed_best', self.computed_haar_cascades_name)

        if not self.use_progresive_fpr:
            self.stage_fpr_schedule = [(0, None, 0.50)]

        if not self.use_augmentation:
            self.faces_aug = {}
            self.bg_aug = {}


    def get_stage_fpr(self, stage_num: int) -> float:
        for start, end, fpr in self.stage_fpr_schedule:
            if end is None and stage_num >= start:
                return fpr
            if start <= stage_num <= end:
                return fpr
        return self.stage_target_fpr
