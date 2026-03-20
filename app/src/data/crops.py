import cv2
import numpy as np
from src.config import Configuration

def get_all_image_crops(CONFIG: Configuration, img_path: str = None, img: np.ndarray = None):
    crops = []
    if img is None:
        if CONFIG.gray_scale:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)

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