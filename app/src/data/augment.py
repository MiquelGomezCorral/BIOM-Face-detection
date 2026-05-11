import cv2
import numpy as np
from functools import partial


def apply_noise(img: np.ndarray, intensity: float = 0.1, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, intensity * 255, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast(img: np.ndarray, factor_range: tuple = (0.5, 1.5), rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    factor = rng.uniform(*factor_range)
    result = img.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, kernel_range: tuple = (1, 3), rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    ksize = rng.integers(*kernel_range, endpoint=True)
    if ksize < 2:
        return img.copy()
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_light(img: np.ndarray, brightness_range: tuple = (-30, 30), rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    brightness = rng.integers(*brightness_range, endpoint=True)
    result = img.astype(np.float32) + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


AUGMENTATION_FUNCS = {
    "noise": apply_noise,
    "contrast": apply_contrast,
    "blur": apply_blur,
    "light": apply_light,
}


def augment_image(img: np.ndarray, aug_config: dict, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    result = img.copy()
    for aug_type, prob in aug_config.items():
        if aug_type not in AUGMENTATION_FUNCS:
            continue
        if rng.random() < prob:
            result = AUGMENTATION_FUNCS[aug_type](result, rng=rng)
    return result


def create_face_augmentor(CONFIG) -> partial:
    return partial(augment_image, aug_config=CONFIG.faces_aug)


def create_bg_augmentor(CONFIG) -> partial:
    return partial(augment_image, aug_config=CONFIG.bg_aug)
