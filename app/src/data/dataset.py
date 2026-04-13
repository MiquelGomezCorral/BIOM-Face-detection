import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from maikol_utils.file_utils import load_json

from src.config import Configuration

def load_gb_images(CONFIG: Configuration):
    to_keep_labels = load_json(CONFIG.dataset_classes_path)

    # Download dataset without faces
    fo.config.dataset_zoo_dir = CONFIG.no_faces_path
    bg_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=to_keep_labels,
        max_samples=CONFIG.max_bg_samples,
        # dataset_name="open-images-bg",  # ADDED: Forces a distinct dataset instance
        drop_existing_dataset=True      # ADDED: Clears old corrupted cache
    )
    bg_dataset = bg_dataset.filter_labels("ground_truth", F("label").is_in(to_keep_labels)) 

    return bg_dataset