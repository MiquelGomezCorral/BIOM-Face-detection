import cv2
import numpy as np
from src.config import Configuration

def get_all_image_crops(CONFIG: Configuration, img_path: str = None, img: np.ndarray = None):
    crops = []
    if img is None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    current_scale = 1.0
    iteration = 0
    
    while img.shape[0] > CONFIG.crop_size and img.shape[1] > CONFIG.crop_size:
        crops.extend(get_image_crops(img, CONFIG.stride, CONFIG.crop_size, current_scale))
        
        new_w = int(img.shape[1] * CONFIG.subsample_factor)
        new_h = int(img.shape[0] * CONFIG.subsample_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        current_scale /= CONFIG.subsample_factor 
        iteration += 1
    return crops

def get_image_crops(img: np.ndarray, stride: int, crop_size: int, scale: float):
    h, w = img.shape[:2]
    crops = []
    for i in range(0, h - crop_size + 1, stride):
        for j in range(0, w - crop_size + 1, stride):
            crops.append({
                "x": int(j * scale),
                "y": int(i * scale),
                "w": int(crop_size * scale),
                "h": int(crop_size * scale),
                "img": np.ascontiguousarray(img[
                    i:i + crop_size,
                    j:j + crop_size
                ]),
            })
    return crops


def get_image_crops_from_list(
    crops_info: list,
    img: np.ndarray = None,
    img_path: str = None,
    read_scale_factor: int = 1,
):
    if img is None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if read_scale_factor is None:
        read_scale_factor = 1
    read_scale_factor = max(1, int(read_scale_factor))
    if read_scale_factor > 1:
        new_w = max(1, img.shape[1] // read_scale_factor)
        new_h = max(1, img.shape[0] // read_scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    crops = []
    for crop_info in crops_info:
        x, y, w, h = crop_info["x"], crop_info["y"], crop_info["w"], crop_info["h"]
        crops.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "img": np.ascontiguousarray(img[y:y+h, x:x+w]),
        })

    
    return crops