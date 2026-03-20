from PIL import Image
import cv2
from tqdm import tqdm
from maikol_utils.file_utils import list_dir_files
from maikol_utils.print_utils import print_separator
import os
import numpy as np

from src.config import Configuration
from .filter import local_normalize_image
    
def create_partition(CONFIG: Configuration, rng):
    files, n = list_dir_files(CONFIG.all_path, recursive=True)
    files = set(files)
        
    test = set(rng.choice(list(files), size=int(n * CONFIG.test_split), replace=False))
    val = set(rng.choice(list(files - test), size=int(n * CONFIG.val_split), replace=False))
    train = files - test - val

    print(f"Train: {len(train)} images")
    print(f"Val: {len(val)} images")
    print(f"Test: {len(test)} images")

    def copy_as_jpg(src: str, dst_dir: str):
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(dst_dir, base + ".jpg")
        with Image.open(src) as img:
            img.convert("RGB").save(dst, "JPEG")

    for split_paths, paths in zip([CONFIG.train_path, CONFIG.val_path, CONFIG.test_path], [train, val, test]):
        print_separator(split_paths)
        for path in tqdm(list(paths), desc=f"Copying {split_paths} images"):
            copy_as_jpg(path, split_paths)


def apply_filter(CONFIG: Configuration):
    paths = [
        (CONFIG.train_path, CONFIG.train_f_path),
        (CONFIG.val_path, CONFIG.val_f_path),
        (CONFIG.test_path, CONFIG.test_f_path),
    ]

    for src_path, dest_path in paths:
        files, n = list_dir_files(src_path)
        for f_path in tqdm(files, desc=os.path.basename(src_path)):
            if CONFIG.gray_scale:
                img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(f_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply filter (per-channel for color images)
            if CONFIG.gray_scale:
                img = local_normalize_image(CONFIG, img)
            else:
                img = np.stack([
                    local_normalize_image(CONFIG, img[:, :, c])
                    for c in range(img.shape[2])
                ], axis=2)

            img_min, img_max = -3, 3#img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
                
            # Save image
            base = os.path.basename(f_path)
            dest_file = os.path.join(dest_path, base)
            if CONFIG.gray_scale:
                cv2.imwrite(dest_file, img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dest_file, img)
    