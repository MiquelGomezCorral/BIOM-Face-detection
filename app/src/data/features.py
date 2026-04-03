
import os
import numpy as np
from tqdm import tqdm
from typing import List
import multiprocessing as mp
from functools import partial

from src.model import Feature, Rectangle

from src.data import get_integral_image, get_integral_squared_image, get_std_from_integral_images, get_integral_sum, get_image_crops_from_list


# =================================================================
#                   FEATURE DEFINITION AND EXTRACTION
# =================================================================

def generate_all_features(win_w: int = 24, win_h: int = 24) -> List[Feature]:
    features = []
    
    for w in range(1, win_w + 1):
        for h in range(1, win_h + 1):
            for x in range(win_w - w + 1):
                for y in range(win_h - h + 1):
                    
                    # 2-rectangle horizontal (Left/Right)
                    if w % 2 == 0:
                        w_half = w // 2
                        features.append(Feature(
                            feature_id = len(features),
                            rectangles = [
                            Rectangle(x, y, w_half, h, -1.0),
                            Rectangle(x + w_half, y, w_half, h, 1.0)
                        ]))
                        
                    # 2-Rectangle vertical (Top/Bottom)
                    if h % 2 == 0:
                        h_half = h // 2
                        features.append(Feature(
                            feature_id = len(features),
                            rectangles = [
                            Rectangle(x, y, w, h_half, -1.0),
                            Rectangle(x, y + h_half, w, h_half, 1.0)
                        ]))
                        
                    # # 3-Rectangle horizontal (Left/Center/Right)
                    # if w % 3 == 0:
                    #     w_third = w // 3
                    #     features.append(Feature(
                    #         feature_id = len(features),
                    #         rectangles = [
                    #         Rectangle(x, y, w_third, h, -1.0),
                    #         Rectangle(x + w_third, y, w_third, h, 2.0), # Center weight compensates for 2 outside Rectangles
                    #         Rectangle(x + 2 * w_third, y, w_third, h, -1.0)
                    #     ]))
                        
                    # # 3-Rectangle vertical (Top/Center/Bottom)
                    # if h % 3 == 0:
                    #     h_third = h // 3
                    #     features.append(Feature(
                    #         feature_id = len(features),
                    #         rectangles = [
                    #         Rectangle(x, y, w, h_third, -1.0),
                    #         Rectangle(x, y + h_third, w, h_third, 2.0),
                    #         Rectangle(x, y + 2 * h_third, w, h_third, -1.0)
                    #     ]))
                        
                    # # 4-Rectangle (Checkerboard)
                    # if w % 2 == 0 and h % 2 == 0:
                    #     w_half, h_half = w // 2, h // 2
                    #     features.append(Feature(
                    #         feature_id = len(features),
                    #         rectangles = [
                    #         Rectangle(x, y, w_half, h_half, 1.0),
                    #         Rectangle(x + w_half, y, w_half, h_half, -1.0),
                    #         Rectangle(x, y + h_half, w_half, h_half, -1.0),
                    #         Rectangle(x + w_half, y + h_half, w_half, h_half, 1.0)
                    #     ]))
                        
    return features

# =================================================================
#                   MANAGE FEATURES FOR DATASET
# =================================================================
def compute_features_dataset(images_paths, all_features):
    n_workers = int(max(1, (os.cpu_count() or 1) - 1) * 4 / 5)
    chunksize = 8
    # Precompute tensors once
    _R1, _C1, _R2, _C2, _W, _FIDX = precompute_feature_tensors(all_features)
    _N_FEATURES = len(all_features)
    precomputed = (_R1, _C1, _R2, _C2, _W, _FIDX, _N_FEATURES)

    # Bind precomputed tensors to extract_features function
    extract_fn = partial(extract_features, precomputed=precomputed)

    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        results_faces = list(
            tqdm(
                pool.imap(extract_fn, images_paths, chunksize=chunksize),
                total=len(images_paths),
                desc=f"Extracting face features ({n_workers} workers)",
            )
        )

    features = np.stack(results_faces, axis=0).astype(np.float32, copy=False)
    return features

def precompute_feature_tensors(features):
    r1s, c1s, r2s, c2s, ws, fidx = [], [], [], [], [], []
    for i, feat in enumerate(features):
        for rec in feat.rectangles:
            r1s.append(rec.y)
            c1s.append(rec.x)
            r2s.append(rec.y + rec.height - 1)
            c2s.append(rec.x + rec.width - 1)
            ws.append(rec.weight)
            fidx.append(i)
    return (
        np.array(r1s,  dtype=np.int32),
        np.array(c1s,  dtype=np.int32),
        np.array(r2s,  dtype=np.int32),
        np.array(c2s,  dtype=np.int32),
        np.array(ws,   dtype=np.float32),
        np.array(fidx, dtype=np.int32),
    )



def extract_features(img_path: str = None, img: np.ndarray = None, precomputed=None) -> np.ndarray:
    if precomputed is None:
        raise ValueError("precomputed tensors must be provided")
    
    _R1, _C1, _R2, _C2, _W, _FIDX, _N_FEATURES = precomputed
    
    if img is None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    integral_img = get_integral_image(img)
    integral_img_2 = get_integral_squared_image(img)

    # Calculate std
    H, W_size = img.shape
    r1 = np.array([[0]])
    r2 = np.array([[H - 1]])
    c1 = np.array([[0]])
    c2 = np.array([[W_size - 1]])

    _, std_dev = get_std_from_integral_images(integral_img, integral_img_2, r1, r2, c1, c2)
    std_dev = std_dev[0, 0]

    if std_dev <= 0:
        std_dev = 1.0

    res = _compute_features_vectorized(integral_img, _R1, _C1, _R2, _C2, _W, _FIDX, _N_FEATURES) / std_dev
    del integral_img, integral_img_2
    return res




def _compute_features_vectorized(integral_img: np.ndarray, _R1, _C1, _R2, _C2, _W, _FIDX, _N_FEATURES) -> np.ndarray:
    """Compute all Haar features for one integral image — zero Python loops."""
    II = integral_img

    # Integral sum for every rectangle in one shot (vectorized corner lookup)
    vals = II[_R2, _C2].copy()
    mask_r = _R1 > 0
    mask_c = _C1 > 0
    vals[mask_r]           -= II[_R1[mask_r] - 1, _C2[mask_r]]
    vals[mask_c]           -= II[_R2[mask_c], _C1[mask_c] - 1]
    vals[mask_r & mask_c]  += II[_R1[mask_r & mask_c] - 1, _C1[mask_r & mask_c] - 1]

    # Weighted sum, then accumulate into per-feature bins
    return np.bincount(_FIDX, weights=vals * _W, minlength=_N_FEATURES).astype(np.float32)


def extract_features_batch(imgs: list[np.ndarray], precomputed) -> np.ndarray:
    """
    Extract features for a list of crop images.
    Returns shape (n_crops, n_features).
    """
    _R1, _C1, _R2, _C2, _W, _FIDX, _N_FEATURES = precomputed
    
    results = np.empty((len(imgs), _N_FEATURES), dtype=np.float32)
    for i, img in enumerate(imgs):
        integral_img   = get_integral_image(img)
        integral_img_2 = get_integral_squared_image(img)

        H, W_size = img.shape
        _, std_dev = get_std_from_integral_images(
            integral_img, integral_img_2,
            np.array([[0]]), np.array([[H - 1]]),
            np.array([[0]]), np.array([[W_size - 1]]),
        )
        std_dev = float(std_dev[0, 0]) or 1.0

        results[i] = _compute_features_vectorized(integral_img, _R1, _C1, _R2, _C2, _W, _FIDX, _N_FEATURES) / std_dev

    return results
