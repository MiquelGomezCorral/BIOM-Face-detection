"""
fast_cascade_classifier.py
==========================
High-performance, parallelised drop-in replacement for CascadeClassifier.

Architecture
------------

PRIMARY PATH  (Numba installed – ``pip install numba``)
    The whole cascade is pre-flattened into a handful of 1-D NumPy arrays
    at construction time.  A single @njit(parallel=True) kernel evaluates
    every candidate window simultaneously using Numba's ``prange``, spreading
    work across all logical CPU cores through OpenMP/TBB threads.

    Inside the kernel each thread walks through cascade stages independently
    and breaks early the moment a stage rejects its window — the cascade
    rejection property is fully preserved at the per-window level.
    ``cache=True`` saves the compiled binary to __pycache__ so the JIT
    overhead is paid only once, on the very first run.

FALLBACK PATH  (no Numba)
    Scale levels are dispatched to a ``ThreadPoolExecutor``.  NumPy releases
    the GIL during its heavy array operations, so threads run in genuine
    parallel.  The vectorised NumPy pass inside each worker is a clean,
    stage-by-stage masking loop identical in logic to the Numba kernel.

Both paths share the same flat array layout and the same NMS post-processing.

Usage
-----
    from fast_cascade_classifier import CascadeClassifier

    clf = CascadeClassifier(config, cascade)
    clf.warm_up()          # optional: trigger JIT ahead of time
    faces = clf.predict(img=gray_image)
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba import
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    warnings.warn(
        "\n[CascadeClassifier] numba is not installed.\n"
        "  Falling back to threaded NumPy (still multi-core, but ~5-10× slower).\n"
        "  Install it for maximum speed:  pip install numba\n",
        RuntimeWarning,
        stacklevel=2,
    )

from .cascade_parser import HaarCascade


# ===========================================================================
#  Numba JIT kernel  (defined only when numba is importable)
# ===========================================================================
if _NUMBA:

    @njit(parallel=True, cache=True, fastmath=True)
    def _cascade_kernel(
        ii:            np.ndarray,   # (H+1, W+1) int64  – padded integral image
        ii2:           np.ndarray,   # (H+1, W+1) int64  – padded squared-integral
        rows:          np.ndarray,   # (N,)  int32  – window top-left row
        cols:          np.ndarray,   # (N,)  int32  – window top-left col
        crop_size:     int,
        # ---- cascade: stage arrays ----
        stage_thr:     np.ndarray,   # (S,)  float64
        stage_clf_s:   np.ndarray,   # (S,)  int32  – first classifier index
        stage_clf_e:   np.ndarray,   # (S,)  int32  – one-past-last clf index
        # ---- cascade: classifier arrays ----
        clf_thr:       np.ndarray,   # (C,)  float64
        clf_lv:        np.ndarray,   # (C,)  float64  – left (fail) leaf value
        clf_rv:        np.ndarray,   # (C,)  float64  – right (pass) leaf value
        clf_rect_s:    np.ndarray,   # (C,)  int32  – first rect index
        clf_rect_e:    np.ndarray,   # (C,)  int32  – one-past-last rect index
        # ---- cascade: rectangle arrays ----
        rect_ry:       np.ndarray,   # (R,)  int32
        rect_rx:       np.ndarray,   # (R,)  int32
        rect_rh:       np.ndarray,   # (R,)  int32
        rect_rw:       np.ndarray,   # (R,)  int32
        rect_wt:       np.ndarray,   # (R,)  float64
    ) -> np.ndarray:                 # (N,)  uint8   1 = survived all stages
        """
        Evaluate the full Haar cascade over N windows in parallel.

        Each prange iteration is fully independent: it reads from shared
        read-only arrays (ii, ii2, cascade arrays) and writes to its own
        element of `result`.  No synchronisation is needed.
        """
        N        = rows.shape[0]
        n_pix    = float(crop_size * crop_size)
        n_stages = stage_thr.shape[0]
        result   = np.zeros(N, dtype=np.uint8)

        for i in prange(N):                        # ← all CPU cores here
            r  = rows[i]
            c  = cols[i]
            r2 = r + crop_size
            c2 = c + crop_size

            # ------ per-window normalisation std-dev ------
            s1   = float(ii [r2, c2] - ii [r, c2] - ii [r2, c] + ii [r, c])
            s2   = float(ii2[r2, c2] - ii2[r, c2] - ii2[r2, c] + ii2[r, c])
            mean = s1 / n_pix
            var  = s2 / n_pix - mean * mean
            if var < 0.0:
                var = 0.0
            std  = var ** 0.5
            if std < 1e-10:
                std = 1.0

            # ------ cascade: stage-by-stage with early exit ------
            passed = True
            for s in range(n_stages):
                stage_sum = 0.0

                for ci in range(stage_clf_s[s], stage_clf_e[s]):
                    feat = 0.0
                    for ri in range(clf_rect_s[ci], clf_rect_e[ci]):
                        fr1 = r  + rect_ry[ri]
                        fc1 = c  + rect_rx[ri]
                        fr2 = fr1 + rect_rh[ri]
                        fc2 = fc1 + rect_rw[ri]
                        feat += float(
                            ii[fr2, fc2] - ii[fr1, fc2]
                            - ii[fr2, fc1] + ii[fr1, fc1]
                        ) * rect_wt[ri]

                    if feat < clf_thr[ci] * std:
                        stage_sum += clf_lv[ci]
                    else:
                        stage_sum += clf_rv[ci]

                if stage_sum < stage_thr[s]:
                    passed = False
                    break                          # ← cascade early exit

            if passed:
                result[i] = 1

        return result


# ===========================================================================
#  Main classifier
# ===========================================================================

class CascadeClassifier:
    """
    Parallelised Viola-Jones face detector.

    Drop-in replacement for CascadeClassifier – identical public API.
    """

    # -----------------------------------------------------------------------
    def __init__(self, CONFIG, cascade: HaarCascade):
        self.CONFIG     = CONFIG
        self.cascade    = cascade
        self._n_workers = max(1, os.cpu_count() or 2)
        self._build_flat_arrays()

    # -----------------------------------------------------------------------
    #  Flatten the cascade into contiguous 1-D arrays (done once at init)
    # -----------------------------------------------------------------------
    def _build_flat_arrays(self) -> None:
        """
        Walk the HaarCascade object tree and emit six groups of 1-D arrays:
          stage layer  : stage_thr, stage_clf_s, stage_clf_e
          clf   layer  : clf_thr, clf_lv, clf_rv, clf_rect_s, clf_rect_e
          rect  layer  : rect_ry, rect_rx, rect_rh, rect_rw, rect_wt

        The slice convention is [start, end) so that
          for ci in range(stage_clf_s[s], stage_clf_e[s])
        iterates over every classifier in stage s.
        """
        stage_thr_l   = []
        stage_clf_s_l = []
        stage_clf_e_l = []

        clf_thr_l    = []
        clf_lv_l     = []
        clf_rv_l     = []
        clf_rect_s_l = []
        clf_rect_e_l = []

        rect_ry_l = []
        rect_rx_l = []
        rect_rh_l = []
        rect_rw_l = []
        rect_wt_l = []

        clf_cur  = 0   # running classifier index
        rect_cur = 0   # running rectangle index

        for stage in self.cascade.stages:
            stage_thr_l.append(float(stage.threshold))
            stage_clf_s_l.append(clf_cur)

            for clf in stage.weak_classifiers:
                if clf.feature is None:
                    continue
                clf_thr_l.append(float(clf.threshold))
                clf_lv_l.append(float(clf.left_value))
                clf_rv_l.append(float(clf.right_value))
                clf_rect_s_l.append(rect_cur)

                for rec in clf.feature.rectangles:
                    rect_ry_l.append(int(rec.y))
                    rect_rx_l.append(int(rec.x))
                    rect_rh_l.append(int(rec.height))
                    rect_rw_l.append(int(rec.width))
                    rect_wt_l.append(float(rec.weight))
                    rect_cur += 1

                clf_rect_e_l.append(rect_cur)
                clf_cur += 1

            stage_clf_e_l.append(clf_cur)

        # Store as tight, typed, C-contiguous arrays
        f64 = np.float64
        i32 = np.int32

        self._stage_thr   = np.ascontiguousarray(stage_thr_l,   dtype=f64)
        self._stage_clf_s = np.ascontiguousarray(stage_clf_s_l, dtype=i32)
        self._stage_clf_e = np.ascontiguousarray(stage_clf_e_l, dtype=i32)

        self._clf_thr    = np.ascontiguousarray(clf_thr_l,    dtype=f64)
        self._clf_lv     = np.ascontiguousarray(clf_lv_l,     dtype=f64)
        self._clf_rv     = np.ascontiguousarray(clf_rv_l,     dtype=f64)
        self._clf_rect_s = np.ascontiguousarray(clf_rect_s_l, dtype=i32)
        self._clf_rect_e = np.ascontiguousarray(clf_rect_e_l, dtype=i32)

        self._rect_ry = np.ascontiguousarray(rect_ry_l, dtype=i32)
        self._rect_rx = np.ascontiguousarray(rect_rx_l, dtype=i32)
        self._rect_rh = np.ascontiguousarray(rect_rh_l, dtype=i32)
        self._rect_rw = np.ascontiguousarray(rect_rw_l, dtype=i32)
        self._rect_wt = np.ascontiguousarray(rect_wt_l, dtype=f64)

    # -----------------------------------------------------------------------
    #  Public helpers
    # -----------------------------------------------------------------------
    def warm_up(self) -> None:
        """
        Trigger JIT compilation (Numba) or thread-pool warm-up so the first
        real predict() call has no latency spike.
        Uses a tiny all-zero dummy image – detected faces are discarded.
        """
        size  = self.CONFIG.crop_size * 2
        dummy = np.zeros((size, size), dtype=np.uint8)
        self.predict(img=dummy)

    # -----------------------------------------------------------------------
    #  Public API  (identical signature to the original CascadeClassifier)
    # -----------------------------------------------------------------------
    def predict_no_merge(
        self,
        img:                    Optional[np.ndarray] = None,
        img_path:               Optional[str]        = None,
        return_candidate_count: bool                 = False,
        halve_size:             bool                 = False,
        return_loaded_image:    bool                 = False,
    ):
        assert img is not None or img_path is not None, \
            "Either img or img_path must be provided."

        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if halve_size:
            img = cv2.resize(
                img, (img.shape[1] // 2, img.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )

        crop_size = self.CONFIG.crop_size
        stride    = self.CONFIG.stride
        subsample = self.CONFIG.subsample_factor

        if _NUMBA:
            all_faces, total_candidates = self._predict_numba(
                img, crop_size, stride, subsample
            )
        else:
            all_faces, total_candidates = self._predict_threaded(
                img, crop_size, stride, subsample
            )

        if return_candidate_count:
            return all_faces, total_candidates
        if return_loaded_image:
            return all_faces, img
        return all_faces

    def predict(
        self,
        img:                    Optional[np.ndarray] = None,
        img_path:               Optional[str]        = None,
        return_candidate_count: bool                 = False,
    ):
        all_faces, total_candidates = self.predict_no_merge(
            img=img,
            img_path=img_path,
            return_candidate_count=True,
        )

        if not all_faces:
            return ([], total_candidates) if return_candidate_count else []

        # Non-maximum suppression via OpenCV group-rectangles
        rects = [[f["x"], f["y"], f["w"], f["h"]] for f in all_faces]
        grouped, _ = cv2.groupRectangles(rects + rects, groupThreshold=2, eps=0.3)

        if len(grouped) == 0:
            return ([], total_candidates) if return_candidate_count else []

        result = [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for x, y, w, h in grouped
        ]

        return (result, total_candidates) if return_candidate_count else result

    # =======================================================================
    #  Numba path  – sequential over scales, parallel over windows per scale
    # =======================================================================
    def _predict_numba(
        self,
        img:       np.ndarray,
        crop_size: int,
        stride:    int,
        subsample: float,
    ) -> Tuple[List[dict], int]:
        all_faces:   List[dict] = []
        total_cands: int        = 0
        scale = 1.0

        while img.shape[0] >= crop_size and img.shape[1] >= crop_size:
            faces, n = self._scale_numba(img, crop_size, stride, scale)
            all_faces.extend(faces)
            total_cands += n

            new_w = int(img.shape[1] * subsample)
            new_h = int(img.shape[0] * subsample)
            img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale /= subsample

        return all_faces, total_cands

    def _scale_numba(
        self,
        img:       np.ndarray,
        crop_size: int,
        stride:    int,
        scale:     float,
    ) -> Tuple[List[dict], int]:
        H, W = img.shape

        # Padded integral images (zero row+col at index 0)
        base = img.astype(np.int64)
        ii   = np.zeros((H + 1, W + 1), dtype=np.int64)
        ii2  = np.zeros((H + 1, W + 1), dtype=np.int64)
        ii [1:, 1:] = base.cumsum(axis=0).cumsum(axis=1)
        ii2[1:, 1:] = (base * base).cumsum(axis=0).cumsum(axis=1)

        # Window top-left corners
        r_idx = np.arange(0, H - crop_size + 1, stride, dtype=np.int32)
        c_idx = np.arange(0, W - crop_size + 1, stride, dtype=np.int32)
        rr, cc = np.meshgrid(r_idx, c_idx, indexing="ij")
        rows   = np.ascontiguousarray(rr.ravel(), dtype=np.int32)
        cols   = np.ascontiguousarray(cc.ravel(), dtype=np.int32)

        N = len(rows)
        if N == 0:
            return [], 0

        # JIT kernel: all N windows evaluated in parallel
        survived = _cascade_kernel(
            ii, ii2, rows, cols, crop_size,
            self._stage_thr, self._stage_clf_s, self._stage_clf_e,
            self._clf_thr, self._clf_lv, self._clf_rv,
            self._clf_rect_s, self._clf_rect_e,
            self._rect_ry, self._rect_rx, self._rect_rh, self._rect_rw,
            self._rect_wt,
        )

        mask = survived == 1
        faces = [
            {
                "x": int(cols[i] * scale),
                "y": int(rows[i] * scale),
                "w": int(crop_size * scale),
                "h": int(crop_size * scale),
            }
            for i in np.where(mask)[0]
        ]
        return faces, N

    # =======================================================================
    #  Threaded NumPy fallback  – parallel over scales
    # =======================================================================
    def _predict_threaded(
        self,
        img:       np.ndarray,
        crop_size: int,
        stride:    int,
        subsample: float,
    ) -> Tuple[List[dict], int]:
        """
        Pre-compute all scaled images, then fan out each scale to a thread.
        NumPy releases the GIL during its BLAS/array ops → genuine parallelism.
        """
        # Build scale pyramid
        scale_levels = []
        current      = img
        scale        = 1.0

        while current.shape[0] > crop_size and current.shape[1] > crop_size:
            scale_levels.append((current, scale))
            new_w   = int(current.shape[1] * subsample)
            new_h   = int(current.shape[0] * subsample)
            current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale  /= subsample

        all_faces:   List[dict] = []
        total_cands: int        = 0

        with ThreadPoolExecutor(max_workers=self._n_workers) as pool:
            futs = {
                pool.submit(self._scale_numpy, s_img, crop_size, stride, s_val): s_val
                for s_img, s_val in scale_levels
            }
            for fut in as_completed(futs):
                faces, n = fut.result()
                all_faces.extend(faces)
                total_cands += n

        return all_faces, total_cands

    def _scale_numpy(
        self,
        img:       np.ndarray,
        crop_size: int,
        stride:    int,
        scale:     float,
    ) -> Tuple[List[dict], int]:
        """
        Vectorised NumPy cascade for a single scale level.

        Each stage computes feature responses for *all surviving windows* at
        once using array indexing, then discards the rejects with a boolean
        mask.  The surviving set shrinks rapidly after the first few stages,
        making later stages cheap automatically.
        """
        H, W = img.shape

        base = img.astype(np.int64)
        ii   = np.zeros((H + 1, W + 1), dtype=np.int64)
        ii2  = np.zeros((H + 1, W + 1), dtype=np.int64)
        ii [1:, 1:] = base.cumsum(axis=0).cumsum(axis=1)
        ii2[1:, 1:] = (base * base).cumsum(axis=0).cumsum(axis=1)

        r_idx = np.arange(0, H - crop_size + 1, stride, dtype=np.int32)
        c_idx = np.arange(0, W - crop_size + 1, stride, dtype=np.int32)
        rr, cc = np.meshgrid(r_idx, c_idx, indexing="ij")
        active_r = rr.ravel().astype(np.int32)
        active_c = cc.ravel().astype(np.int32)

        N = len(active_r)
        if N == 0:
            return [], 0

        # Compute normalisation std-dev for every initial window
        r2      = active_r + crop_size
        c2      = active_c + crop_size
        n_pix   = float(crop_size * crop_size)
        s1      = (ii [r2, c2] - ii [active_r, c2] - ii [r2, active_c] + ii [active_r, active_c]).astype(np.float64)
        s2      = (ii2[r2, c2] - ii2[active_r, c2] - ii2[r2, active_c] + ii2[active_r, active_c]).astype(np.float64)
        mean    = s1 / n_pix
        var     = np.maximum(s2 / n_pix - mean * mean, 0.0)
        std     = np.sqrt(var)
        std     = np.where(std < 1e-10, 1.0, std)

        active_std = std    # updated in-step alongside active_r / active_c

        n_stages = self._stage_thr.shape[0]

        for s in range(n_stages):
            if active_r.shape[0] == 0:
                break

            c_start = int(self._stage_clf_s[s])
            c_end   = int(self._stage_clf_e[s])
            n_win   = active_r.shape[0]
            stage_sum = np.zeros(n_win, dtype=np.float64)

            for ci in range(c_start, c_end):
                feat = np.zeros(n_win, dtype=np.float64)

                for ri in range(int(self._clf_rect_s[ci]), int(self._clf_rect_e[ci])):
                    fr1 = active_r + int(self._rect_ry[ri])
                    fc1 = active_c + int(self._rect_rx[ri])
                    fr2 = fr1 + int(self._rect_rh[ri])
                    fc2 = fc1 + int(self._rect_rw[ri])
                    feat += (
                        ii[fr2, fc2] - ii[fr1, fc2]
                        - ii[fr2, fc1] + ii[fr1, fc1]
                    ).astype(np.float64) * float(self._rect_wt[ri])

                norm_thr   = float(self._clf_thr[ci]) * active_std
                stage_sum += np.where(
                    feat < norm_thr,
                    float(self._clf_lv[ci]),
                    float(self._clf_rv[ci]),
                )

            mask       = stage_sum >= float(self._stage_thr[s])
            active_r   = active_r  [mask]
            active_c   = active_c  [mask]
            active_std = active_std[mask]

        faces = [
            {
                "x": int(active_c[i] * scale),
                "y": int(active_r[i] * scale),
                "w": int(crop_size * scale),
                "h": int(crop_size * scale),
            }
            for i in range(active_r.shape[0])
        ]
        return faces, N