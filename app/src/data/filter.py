import numpy as np
from src.config import Configuration


# ====================================================================
#                        Versiones rápidas
# ====================================================================
def get_integral_image(img: np.ndarray) -> np.ndarray:
    return img.astype(np.int64).cumsum(axis=0).cumsum(axis=1)

def get_integral_squared_image(img: np.ndarray) -> np.ndarray:
    img2 = img.astype(np.int64) ** 2
    return img2.cumsum(axis=0).cumsum(axis=1)

def get_integral_sum(integral: np.ndarray, x1, y1, x2, y2) -> int:
    A = integral[x1 - 1, y1 - 1] if (x1 > 0 and y1 > 0) else 0
    B = integral[x1 - 1, y2] if x1 > 0 else 0
    C = integral[x2, y1 - 1] if y1 > 0 else 0
    D = integral[x2, y2]
    return D - B - C + A

def get_std_from_integral_images(
    integral: np.ndarray, integral_2: np.ndarray, 
    r1: np.ndarray, r2: np.ndarray, 
    c1: np.ndarray, c2: np.ndarray
) -> tuple:
    """
    Calculate mean and standard deviation from integral images.
    
    Uses vectorized operations with integral and squared integral images
    to compute statistics efficiently across the entire image.
    
    Args:
        integral: Integral image
        integral_2: Squared integral image
        r1, r2: Row start and end indices (vectorized)
        c1, c2: Column start and end indices (vectorized)
    
    Returns:
        Tuple of (mean, std_dev) arrays
    """
    n_pixels = (r2 - r1 + 1) * (c2 - c1 + 1)
    
    def _vectorized_region_sum(integ):
        # Vectorized version of get_integral_sum logic
        D = integ[r2, c2]
        B = np.where(r1 > 0, integ[r1 - 1, c2], 0)
        C = np.where(c1 > 0, integ[r2, c1 - 1], 0)
        A = np.where((r1 > 0) & (c1 > 0), integ[r1 - 1, c1 - 1], 0)
        return D - B - C + A
    
    suma   = _vectorized_region_sum(integral).astype(np.float64)
    suma_2 = _vectorized_region_sum(integral_2).astype(np.float64)
    
    mean = suma / n_pixels
    variance = np.maximum((suma_2 - 2 * mean * suma + n_pixels * mean * mean) / n_pixels, 0.0)
    std_dev = np.sqrt(variance)
    
    return mean, std_dev


def local_normalize_image(CONFIG: Configuration, img: np.ndarray):
    """
    Normalize image locally using integral images for efficient computation.
    """
    integral   = get_integral_image(img)
    integral_2 = get_integral_squared_image(img)

    H, W = img.shape
    half = CONFIG.normalize_window // 2

    rows = np.arange(H)
    cols = np.arange(W)

    r1 = np.maximum(0,     rows - half)[:, None]   # (H, 1)
    r2 = np.minimum(H - 1, rows + half)[:, None]
    c1 = np.maximum(0,     cols - half)[None, :]   # (1, W)
    c2 = np.minimum(W - 1, cols + half)[None, :]

    # Calculate mean and std using integral images
    mu, sig = get_std_from_integral_images(integral, integral_2, r1, r2, c1, c2)

    # Normalize: avoid division by zero
    return np.where(sig < 1e-6, 0.0, (img.astype(np.float64) - mu) / sig)



# ====================================================================
#                        Versiones lentas
# ====================================================================

# def local_normalize_image(CONFIG: Configuration, img: np.ndarray):
#     integral = get_integral_image(img) 
#     integral_2 = get_integral_squared_image(img)

#     # x_idx, y_idx = np.indices(img.shape)
#     # vect_normalize = np.vectorize(
#     #     local_normalize_pixel,
#     #     excluded=["integral", "integral_2", "win_size"]
#     # )

#     # return vect_normalize(
#     #     img,
#     #     integral=integral,
#     #     integral_2=integral_2,
#     #     x=x_idx,
#     #     y=y_idx,
#     #     win_size=CONFIG.normalize_window
#     # )
#     H, W = img.shape
#     out_img = np.zeros((H, W), dtype=np.float64)

#     # Wrapper to process an entire row in one thread
#     def process_row(x):
#         for y in range(W):
#             out_img[x, y] = local_normalize_pixel(img[x, y], integral, integral_2, x, y, CONFIG.normalize_window)

#     # Execute rows in parallel
#     with ThreadPoolExecutor() as executor:
#         executor.map(process_row, range(H))

#     return out_img

# def local_normalize_pixel(pixel, integral, integral_2, x, y, win_size):
#     win = win_size // 2
#     max_row = integral.shape[0] - 1
#     max_col = integral.shape[1] - 1

#     # Ajustar límites
#     r1 = max(0, x - win)
#     r2 = min(max_row, x + win)
#     c1 = max(0, y - win)
#     c2 = min(max_col, y + win)

#     pixels = (r2 - r1 + 1) * (c2 - c1 + 1)

#     suma = get_integral_sum(integral, r1, c1, r2, c2)
#     suma_2 = get_integral_sum(integral_2, r1, c1, r2, c2)
#     mu = suma / pixels

#     var = max(0.0, (suma_2 - (2*mu*suma) + pixels*mu*mu)/pixels) 
#     sig = np.sqrt(var)

#     # Prevent division by zero
#     if sig < 1e-6:
#         return 0.0

#     return (pixel - mu)/sig





# def get_integral_image(img: np.ndarray):
#     integral = np.zeros_like(img, dtype=np.uint64) 

#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             integral[i, j] = img[0:i+1, 0:j+1].sum()

#     return integral

# def get_integral_squared_image(img: np.ndarray):
#     img_2 = np.square(img)
#     integral = np.zeros_like(img_2, dtype=np.uint64) 

#     for i in range(img_2.shape[0]):
#         for j in range(img_2.shape[1]):
#             integral[i, j] = img_2[0:i+1, 0:j+1].sum()

#     return integral
