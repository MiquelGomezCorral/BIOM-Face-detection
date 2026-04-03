import numpy as np
import cv2

# from src.data import (
#     get_integral_image, 
#     get_integral_squared_image,
#     get_integral_sum,
#     get_std_from_integral_images,

#     get_all_image_crops
# )

from .cascade_parser import HaarCascade


class CascadeClassifier:
    def __init__(self, CONFIG, cascade: HaarCascade):
        self.CONFIG = CONFIG
        self.cascade = cascade
        self._precompute_stage_arrays()

    # =======================================================================
    # Pre-computation: flatten the cascade into plain numpy arrays once,
    # so the hot path never touches Python objects or attribute lookups.
    # =======================================================================
    def _precompute_stage_arrays(self):
        """
        Convert the HaarCascade object structure into a list of dicts, each
        holding numpy arrays for one stage.  Doing this once at init time
        means the inner loops only touch numpy - no getattr, no Python lists.

        Per stage we store:
          rect_data  : float32 (n_rects, 5)  → [ry, rx, rh, rw, weight]
          clf_slices : int32   (n_clf,  2)   → [rect_start, rect_end)
          clf_thrs   : float64 (n_clf,)
          clf_lv     : float64 (n_clf,)
          clf_rv     : float64 (n_clf,)
          stage_thr  : float scalar
        """
        self.stages_arrays = []

        for stage in self.cascade.stages:
            all_rects = []    # (ry, rx, rh, rw, weight)
            clf_slices = []   # [start, end) into all_rects
            clf_thrs = []
            clf_lv = []
            clf_rv = []

            for clf in stage.weak_classifiers:
                if clf.feature is None:
                    continue
                start = len(all_rects)
                for rec in clf.feature.rectangles:
                    all_rects.append((rec.y, rec.x, rec.height, rec.width, rec.weight))
                end = len(all_rects)
                clf_slices.append((start, end))
                clf_thrs.append(clf.threshold)
                clf_lv.append(clf.left_value)
                clf_rv.append(clf.right_value)

            self.stages_arrays.append({
                "rect_data":  np.array(all_rects,  dtype=np.float32),   # (R, 5)
                "clf_slices": np.array(clf_slices, dtype=np.int32),      # (C, 2)
                "clf_starts": np.array([sl[0] for sl in clf_slices], dtype=np.int64),
                "clf_thrs":   np.array(clf_thrs,   dtype=np.float64),    # (C,)
                "clf_lv":     np.array(clf_lv,     dtype=np.float64),    # (C,)
                "clf_rv":     np.array(clf_rv,     dtype=np.float64),    # (C,)
                "stage_thr":  float(stage.threshold),
            })

    # =======================================================================
    #                          PREDICT (NO MERGE)
    # =======================================================================
    def predict_no_merge(self, img=None, img_path=None, return_candidate_count: bool = False, halve_size: bool = False, return_loaded_image: bool = False):
        assert img is not None or img_path is not None, \
            "Either img or img_path must be provided"

        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if halve_size:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)

        all_faces = []
        total_candidate_crops = 0
        current_scale = 1.0
        crop_size = self.CONFIG.crop_size
        stride = self.CONFIG.stride

        while img.shape[0] > crop_size and img.shape[1] > crop_size:
            faces, n_candidates = self._predict_scale(img, crop_size, stride, current_scale)
            total_candidate_crops += n_candidates
            all_faces.extend(faces)

            new_w = int(img.shape[1] * self.CONFIG.subsample_factor)
            new_h = int(img.shape[0] * self.CONFIG.subsample_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            current_scale /= self.CONFIG.subsample_factor

        if return_candidate_count:
            return all_faces, total_candidate_crops
        if return_loaded_image:
            return all_faces, img
        return all_faces

    # =======================================================================
    #                               PREDICT
    # =======================================================================
    def predict(self, img=None, img_path=None, return_candidate_count: bool = False):
        all_faces, total_candidate_crops = self.predict_no_merge(
            img=img,
            img_path=img_path,
            return_candidate_count=True,
        )

        if not all_faces:
            if return_candidate_count:
                return [], total_candidate_crops
            return []
        
        # ========================== Non-maximum suppression ==========================
        rects = [[f["x"], f["y"], f["w"], f["h"]] for f in all_faces]
        rects_doubled = rects + rects
        grouped, _ = cv2.groupRectangles(rects_doubled, groupThreshold=2, eps=0.3)

        if len(grouped) == 0:
            if return_candidate_count:
                return [], total_candidate_crops
            return []

        grouped_faces = [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for x, y, w, h in grouped
        ]

        if return_candidate_count:
            return grouped_faces, total_candidate_crops
        return grouped_faces

    # =======================================================================
    #                       Process on a scale level
    # =======================================================================
    def _predict_scale(
        self, 
        img: np.ndarray, crop_size: int,
        stride: int, scale: float
    ) -> tuple[list, int]:
        H, W = img.shape[:2]

        # ========= Build padded integral images of the FULL scaled image (once) =========
        # Padding with a zero row/col at the top-left lets us use the clean formula:
        #   region_sum(r1,c1,r2,c2) = ii[r2,c2] - ii[r1,c2] - ii[r2,c1] + ii[r1,c1]
        # where all four indices are in [0, H] x [0, W] – no boundary checks needed.

        base = img.astype(np.int64)
        ii = np.empty((H + 1, W + 1), dtype=np.int64)
        ii[0, :] = 0
        ii[:, 0] = 0
        ii[1:, 1:] = base.cumsum(axis=0).cumsum(axis=1)

        ii2 = np.empty((H + 1, W + 1), dtype=np.int64)
        ii2[0, :] = 0
        ii2[:, 0] = 0
        ii2[1:, 1:] = (base ** 2).cumsum(axis=0).cumsum(axis=1)

        # ========= Generate all window top-left corners =========
        rows = np.arange(0, H - crop_size + 1, stride, dtype=np.int32)
        cols = np.arange(0, W - crop_size + 1, stride, dtype=np.int32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr = rr.ravel()   # (N,)
        cc = cc.ravel()   # (N,)

        N = len(rr)
        if N == 0:
            return [], 0

        # ========= Compute std_dev for every window in one vectorised pass =========
        #
        # For window starting at (r, c) of size S×S:
        #   padded corners: top-left=(r, c), bottom-right=(r+S, c+S)
        #   sum = ii[r+S, c+S] - ii[r, c+S] - ii[r+S, c] + ii[r, c]

        n_pixels = crop_size * crop_size  # scalar

        r2 = rr + crop_size   # (N,) – padded bottom row
        c2 = cc + crop_size   # (N,) – padded right col

        s1 = (ii[r2, c2] - ii[rr, c2] - ii[r2, cc] + ii[rr, cc]).astype(np.float64)
        s2 = (ii2[r2, c2] - ii2[rr, c2] - ii2[r2, cc] + ii2[rr, cc]).astype(np.float64)

        mean     = s1 / n_pixels
        variance = np.maximum(s2 / n_pixels - mean * mean, 0.0)
        std_dev  = np.sqrt(variance)
        std_dev  = np.where(std_dev <= 0.0, 1.0, std_dev)   # (N,)

        inv_area = 1.0 / float(n_pixels)

        # ========= Cascade: process stages, shrinking the active set each time =========
        #
        # active_rr / active_cc / active_std hold only the survivors so far.
        # After each stage we boolean-mask to the passing windows.
        # The early stages (which reject ~99 % of windows) do the heavy work,
        # but later stages automatically operate on far fewer candidates.

        active_rr  = rr
        active_cc  = cc
        active_std = std_dev

        for stage in self.stages_arrays:
            n_active = len(active_rr)
            if n_active == 0:
                break

            rect_data  = stage["rect_data"]    # (R, 5) float32
            clf_slices = stage["clf_slices"]   # (C, 2) int32
            clf_starts = stage["clf_starts"]   # (C,) int64
            clf_thrs   = stage["clf_thrs"]     # (C,)   float64
            clf_lv     = stage["clf_lv"]       # (C,)
            clf_rv     = stage["clf_rv"]       # (C,)
            stage_thr  = stage["stage_thr"]

            R = rect_data.shape[0]

            if n_active * R > 20_000_000:
                stage_sum = np.zeros(n_active, dtype=np.float64)

                for clf_idx in range(len(clf_slices)):
                    r_start, r_end = clf_slices[clf_idx]

                    feature_val = np.zeros(n_active, dtype=np.float64)
                    for r_idx in range(r_start, r_end):
                        ry, rx, rh, rw, weight = rect_data[r_idx]
                        ry = int(ry)
                        rx = int(rx)
                        rh = int(rh)
                        rw = int(rw)

                        fr1 = active_rr + ry
                        fc1 = active_cc + rx
                        fr2 = fr1 + rh
                        fc2 = fc1 + rw

                        rect_sum = (
                            ii[fr2, fc2]
                            - ii[fr1, fc2]
                            - ii[fr2, fc1]
                            + ii[fr1, fc1]
                        ).astype(np.float64)

                        feature_val += rect_sum * weight

                    feature_val *= inv_area
                    norm_thr = clf_thrs[clf_idx] * active_std
                    stage_sum += np.where(
                        feature_val < norm_thr,
                        clf_lv[clf_idx],
                        clf_rv[clf_idx],
                    )
            else:
                ry = rect_data[:, 0].astype(np.int32)
                rx = rect_data[:, 1].astype(np.int32)
                rh = rect_data[:, 2].astype(np.int32)
                rw = rect_data[:, 3].astype(np.int32)
                weights = rect_data[:, 4].astype(np.float64)

                fr1 = active_rr[:, None] + ry[None, :]
                fc1 = active_cc[:, None] + rx[None, :]
                fr2 = fr1 + rh[None, :]
                fc2 = fc1 + rw[None, :]

                rect_sums = (
                    ii[fr2, fc2]
                    - ii[fr1, fc2]
                    - ii[fr2, fc1]
                    + ii[fr1, fc1]
                ).astype(np.float64)

                weighted = rect_sums * weights[None, :]
                feature_vals = np.add.reduceat(weighted, clf_starts, axis=1) * inv_area

                norm_thrs = clf_thrs[None, :] * active_std[:, None]
                votes = np.where(feature_vals < norm_thrs, clf_lv[None, :], clf_rv[None, :])
                stage_sum = votes.sum(axis=1)

            # Reject windows that fail this stage (the cascade gate)
            mask       = stage_sum >= stage_thr
            active_rr  = active_rr[mask]
            active_cc  = active_cc[mask]
            active_std = active_std[mask]

        # ========= Collect surviving windows as face dicts =========
        faces = []
        for r, c in zip(active_rr, active_cc):
            faces.append({
                "x":   int(c * scale),
                "y":   int(r * scale),
                "w":   int(crop_size * scale),
                "h":   int(crop_size * scale),
            })

        return faces, N
    



# class SlowCascadeClassifier:
#     def __init__(self, CONFIG, cascade: HaarCascade):
#         self.CONFIG = CONFIG
#         self.cascade = cascade

#     def predict(self, img=None, img_path=None, return_candidate_count: bool = False):
#         assert img is not None or img_path is not None, "Either img or img_path must be provided"

#         print("Getting image crops...")
#         crops = get_all_image_crops(
#             self.CONFIG, 
#             img=img,
#             img_path=img_path
#         )

#         print("Predicting faces...")
#         faces = [
#             crop
#             for crop in crops
#             if self._predict_crop(crop["img"])
#         ]

#         if return_candidate_count:
#             return faces, len(crops)
#         return faces
    
#     def _predict_crop(self, crop_img):
#         # Calculate integral images
#         integral_img = get_integral_image(crop_img)
#         integral_img_2 = get_integral_squared_image(crop_img)

#         # Calculate std 
#         H, W = crop_img.shape
#         r1 = np.array([[0]])
#         r2 = np.array([[H - 1]])
#         c1 = np.array([[0]])
#         c2 = np.array([[W - 1]])
        
#         _, std_dev = get_std_from_integral_images(integral_img, integral_img_2, r1, r2, c1, c2)
#         std_dev = std_dev[0, 0] 
        
#         if std_dev <= 0:
#             std_dev = 1.0

#         inv_window_area = 1.0 / float(H * W)

#         for stage in self.cascade.stages:
#             if not self._predict_stage(integral_img, stage, std_dev, inv_window_area):
#                 return False
        
#         return True

#     def _predict_stage(self, integral_img, stage, std_dev=1.0, inv_window_area=1.0):
#         total_sum = 0.0
#         for clf in stage.weak_classifiers:
#             if clf.feature is None:
#                 continue

#             feature_val = compute_feature(integral_img, clf.feature) * inv_window_area

#             norm_thrs = clf.threshold * std_dev
#             total_sum += (
#                 clf.left_value 
#                 if feature_val < norm_thrs 
#                 else clf.right_value
#             )

#         return total_sum >= stage.threshold
    
# def compute_feature(integral_img, feature):
#     feature_sum = 0.0
#     for rec in feature.rectangles:
#         # XML rectangle format is x, y, width, height where x is column and y is row.
#         rec_sum = get_integral_sum(
#             integral_img, 
#             rec.y, rec.x,
#             rec.y + rec.height - 1, rec.x + rec.width - 1
#         )
#         feature_sum += rec_sum * rec.weight

#     return feature_sum
    