
import os
import numpy as np
from tqdm import tqdm
from typing import List
import multiprocessing as mp
from functools import partial
import cv2

from src.model import Feature, Rectangle

from src.data import get_integral_image, get_integral_squared_image, get_std_from_integral_images, get_integral_sum, get_image_crops_from_list


# =================================================================
#                   FEATURE DEFINITION AND EXTRACTION
# =================================================================

def generate_all_features(
    win_w: int = 24,
    win_h: int = 24,
    edge_margin: int = 2,
    stride: int = 1,
    include_square_features: bool = True,
) -> List[Feature]:
    features = []

    if edge_margin < 0:
        raise ValueError("edge_margin must be >= 0")
    if stride < 1:
        raise ValueError("position_stride must be >= 1")

    # Ensure there is still valid room after applying margins.
    if (2 * edge_margin) >= win_w or (2 * edge_margin) >= win_h:
        return features
    
    type_constraints = {
        0: (6,  4,  'w%2'),   # horizontal 2rect: mínimo 6x4
        1: (4,  6,  'h%2'),   # vertical 2rect:   mínimo 4x6
        2: (9,  4,  'w%3'),   # horizontal 3rect: mínimo 9x4
        3: (4,  9,  'h%3'),   # vertical 3rect:   mínimo 4x9
        4: (6,  6,  'wh%2'),  # diagonal 4rect:   mínimo 6x6
        5: (6,  6,  'wh%3_square'),  # frame: square with inner square, min 9x9
    }

    def _get_xy_ranges(w: int, h: int):
        x_min = edge_margin
        y_min = edge_margin
        x_max = win_w - w - edge_margin
        y_max = win_h - h - edge_margin
        if x_max < x_min or y_max < y_min:
            return range(0), range(0)

        return (
            range(x_min, x_max + 1, stride),
            range(y_min, y_max + 1, stride),
        )

    # 2-rectangle horizontal (Left/Right)
    min_w, min_h, _ = type_constraints[0]
    for w in range(min_w, win_w + 1):
        if w % 2 != 0:
            continue
        w_half = w // 2
        for h in range(min_h, win_h + 1):
            x_range, y_range = _get_xy_ranges(w, h)
            for x in x_range:
                for y in y_range:
                    features.append(Feature(
                        feature_id=len(features),
                        rectangles=[
                            Rectangle(x, y, w_half, h, -1.0),
                            Rectangle(x + w_half, y, w_half, h, 1.0),
                        ],
                    ))

    # 2-rectangle vertical (Top/Bottom)
    min_w, min_h, _ = type_constraints[1]
    for w in range(min_w, win_w + 1):
        for h in range(min_h, win_h + 1):
            if h % 2 != 0:
                continue
            h_half = h // 2
            x_range, y_range = _get_xy_ranges(w, h)
            for x in x_range:
                for y in y_range:
                    features.append(Feature(
                        feature_id=len(features),
                        rectangles=[
                            Rectangle(x, y, w, h_half, -1.0),
                            Rectangle(x, y + h_half, w, h_half, 1.0),
                        ],
                    ))

    # 3-rectangle horizontal (Left/Center/Right)
    min_w, min_h, _ = type_constraints[2]
    for w in range(min_w, win_w + 1):
        if w % 3 != 0:
            continue
        w_third = w // 3
        for h in range(min_h, win_h + 1):
            x_range, y_range = _get_xy_ranges(w, h)
            for x in x_range:
                for y in y_range:
                    features.append(Feature(
                        feature_id=len(features),
                        rectangles=[
                            Rectangle(x, y, w_third, h, -1.0),
                            Rectangle(x + w_third, y, w_third, h, 2.0),
                            Rectangle(x + 2 * w_third, y, w_third, h, -1.0),
                        ],
                    ))

    # 3-rectangle vertical (Top/Center/Bottom)
    min_w, min_h, _ = type_constraints[3]
    for w in range(min_w, win_w + 1):
        for h in range(min_h, win_h + 1):
            if h % 3 != 0:
                continue
            h_third = h // 3
            x_range, y_range = _get_xy_ranges(w, h)
            for x in x_range:
                for y in y_range:
                    features.append(Feature(
                        feature_id=len(features),
                        rectangles=[
                            Rectangle(x, y, w, h_third, -1.0),
                            Rectangle(x, y + h_third, w, h_third, 2.0),
                            Rectangle(x, y + 2 * h_third, w, h_third, -1.0),
                        ],
                    ))

    # 4-rectangle (Checkerboard)
    min_w, min_h, _ = type_constraints[4]
    for w in range(min_w, win_w + 1):
        if w % 2 != 0:
            continue
        w_half = w // 2
        for h in range(min_h, win_h + 1):
            if h % 2 != 0:
                continue
            h_half = h // 2
            x_range, y_range = _get_xy_ranges(w, h)
            for x in x_range:
                for y in y_range:
                    features.append(Feature(
                        feature_id=len(features),
                        rectangles=[
                            Rectangle(x, y, w_half, h_half, 1.0),
                            Rectangle(x + w_half, y, w_half, h_half, -1.0),
                            Rectangle(x, y + h_half, w_half, h_half, -1.0),
                            Rectangle(x + w_half, y + h_half, w_half, h_half, 1.0),
                        ],
                    ))

    # 5: frame (outer square / inner square)
    if include_square_features:
        min_w, min_h, _ = type_constraints[5]
        for w in range(min_w, min(win_w, win_h) + 1):
            if w % 3 != 0:
                continue
            h = w  # enforce square
            w_third = w // 3
            x_range, y_range = _get_xy_ranges(w, h)
            for x in x_range:
                for y in y_range:
                    features.append(Feature(
                        feature_id=len(features),
                        rectangles=[
                            Rectangle(x,           y,           w,       h,       -1.0),
                            Rectangle(x + w_third, y + w_third, w_third, w_third,  2.0),
                        ],
                    ))
                            
    return features

def _extract_with_aug_and_precomputed(args):
    img_path, augment_fn, precomputed = args
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = augment_fn(img)
    return extract_features(img=img, precomputed=precomputed)

# =================================================================
#                   MANAGE FEATURES FOR DATASET
# =================================================================
def compute_features_dataset(images_paths, all_features, n_workers=8, augment_fn=None):
    chunksize = n_workers
    precomputed = precompute_feature_tensors(all_features)

    if augment_fn is not None:
        tasks = [(p, augment_fn, precomputed) for p in images_paths]
        with mp.get_context("fork").Pool(processes=n_workers) as pool:
            results_faces = list(
                tqdm(
                    pool.imap(_extract_with_aug_and_precomputed, tasks, chunksize=chunksize),
                    total=len(images_paths),
                    desc=f"Extracting face features ({n_workers} workers)",
                )
            )
    else:
        extract_fn = partial(extract_features, precomputed=precomputed)
        with mp.get_context("fork").Pool(processes=n_workers) as pool:
            results_faces = list(
                tqdm(
                    pool.imap(extract_fn, images_paths, chunksize=chunksize),
                    total=len(images_paths),
                    desc=f"Extracting face features ({n_workers} workers)",
                )
            )

    features = np.stack([r for r in results_faces if r is not None], axis=0).astype(np.float32, copy=False)
    return features, precomputed

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
    _R1, _C1, _R2, _C2, _W, _FIDX = (
        np.array(r1s,  dtype=np.int32),
        np.array(c1s,  dtype=np.int32),
        np.array(r2s,  dtype=np.int32),
        np.array(c2s,  dtype=np.int32),
        np.array(ws,   dtype=np.float32),
        np.array(fidx, dtype=np.int32),
    )

    precomputed = (_R1, _C1, _R2, _C2, _W, _FIDX, len(features))
    return precomputed



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


def extract_features_batch(imgs: list[np.ndarray], precomputed, augment_fn=None) -> np.ndarray:
    """
    Extract features for a list of crop images.
    Returns shape (n_crops, n_features).
    """
    _R1, _C1, _R2, _C2, _W, _FIDX, _N_FEATURES = precomputed
    
    results = np.empty((len(imgs), _N_FEATURES), dtype=np.float32)
    for i, img in enumerate(imgs):
        if augment_fn is not None:
            img = augment_fn(img)
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
