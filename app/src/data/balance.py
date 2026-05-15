import threading
import queue
import numpy as np
from tqdm import tqdm
from concurrent.futures import CancelledError
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from maikol_utils.print_utils import print_warn

from src.model import CascadeClassifier
from .features import extract_features_batch

import threading
import queue
import numpy as np
from tqdm import tqdm
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait, FIRST_COMPLETED

from maikol_utils.print_utils import print_warn
from src.model import CascadeClassifier
from .features import extract_features_batch


def balance_non_face_samples(
        classifier: CascadeClassifier,
        num_samples,
        bg_samples,
        precomputed,
        crop_size: int,
        n_workers=8,
        stop_check_interval=100,
        max_detections_per_image=1500,
        predictor_halve_size=False,
        predictor_halve_size_factor=1,
        augment_fn=None,
    ):

    stop_event = threading.Event()
    chunk_queue = queue.Queue(maxsize=max(1, n_workers * 4))

    # Shared stats — updated inside workers, read by main thread for display.
    # Using a lock-free approach: only one int written per counter per worker
    # call, and we read them only for display (approximate is fine).
    stats = {
        "candidates":         0,
        "detections":         0,
        "images_processed":   0,
        "images_with_det":    0,
        "generated":          0,
    }
    stats_lock = threading.Lock()

    def _add_stats(**kwargs):
        with stats_lock:
            for k, v in kwargs.items():
                stats[k] += v

    def process_image(filepath):
        if stop_event.is_set():
            return

        fps, candidates = classifier.predict_no_merge(
            img_path=filepath,
            return_candidate_count=True,
            halve_size=predictor_halve_size,
            halve_size_factor=predictor_halve_size_factor,
        )

        n_det = len(fps) if fps else 0
        _add_stats(
            candidates=candidates,
            detections=n_det,
            images_processed=1,
            images_with_det=1 if n_det > 0 else 0,
        )

        if not fps:
            return

        scale = predictor_halve_size_factor if predictor_halve_size else 1
        crop_refs = [
            {"filepath": filepath, "x": fp["x"], "y": fp["y"],
             "w": fp["w"], "h": fp["h"], "scale": scale}
            for fp in fps
        ]

        if len(crop_refs) > max_detections_per_image:
            np.random.shuffle(crop_refs)
            crop_refs = crop_refs[:max_detections_per_image]

        np.random.shuffle(crop_refs)

        for start in range(0, len(crop_refs), stop_check_interval):
            if stop_event.is_set():
                break
            chunk_refs = crop_refs[start:min(start + stop_check_interval, len(crop_refs))]
            crops = _load_crops_from_refs(chunk_refs, crop_size)
            if not crops:
                continue

            chunk_features = list(
                extract_features_batch(
                    [c["img"] for c in crops],
                    precomputed=precomputed,
                    augment_fn=augment_fn,
                )
            )
            if not chunk_features:
                continue

            _add_stats(generated=len(chunk_features))

            # Non-blocking put — retry until stop or success
            while not stop_event.is_set():
                try:
                    chunk_queue.put((filepath, chunk_features), timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _load_crops_from_refs(crop_refs, crop_size):
        if not crop_refs:
            return []
        import cv2

        by_path = {}
        for ref in crop_refs:
            path = ref["filepath"]
            if path in by_path:
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            scale = ref["scale"]          # Bug 2 fix: per-ref, not [0]
            if scale > 1:
                new_w = max(1, img.shape[1] // scale)
                new_h = max(1, img.shape[0] // scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            by_path[path] = img

        crops = []
        for ref in crop_refs:
            img = by_path.get(ref["filepath"])
            if img is None:
                continue
            x  = max(0, ref["x"])
            y  = max(0, ref["y"])
            x2 = min(x + ref["w"], img.shape[1])
            y2 = min(y + ref["h"], img.shape[0])
            if x2 <= x or y2 <= y:
                continue
            crop_img = img[y:y2, x:x2]
            # Bug 1 fix: always resize to the fixed window size
            if crop_img.shape[0] != crop_size or crop_img.shape[1] != crop_size:
                crop_img = cv2.resize(
                    crop_img, (crop_size, crop_size), interpolation=cv2.INTER_AREA
                )
            crops.append({
                "x": x, "y": y, "w": ref["w"], "h": ref["h"],
                "img": np.ascontiguousarray(crop_img),
            })
        return crops

    # ── main collection loop ────────────────────────────────────────────────
    np.random.shuffle(bg_samples)
    print(
        f" - Generating {num_samples} hard negative samples "
        f"({n_workers} workers, stop-check every {stop_check_interval} crops, "
        f"max {max_detections_per_image} detections/image)..."
    )

    all_hard_bg = []
    contributing_images = set()

    def _postfix():
        with stats_lock:
            return dict(
                imgs=stats["images_processed"],
                cand=stats["candidates"],
                det=stats["detections"],
                gen=stats["generated"],
                accepted=len(all_hard_bg),
                contrib=len(contributing_images),
            )

    with tqdm(total=num_samples, desc="Hard negatives", unit="crops") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            fp_iter    = iter(bg_samples)
            futures    = {}
            buffer_size = n_workers * 2

            def submit_next():
                fp = next(fp_iter, None)
                if fp is None or stop_event.is_set():
                    return
                f = executor.submit(process_image, fp)
                futures[f] = fp

            for _ in range(buffer_size):
                submit_next()

            while futures or not chunk_queue.empty():
                # ── drain queue first ──────────────────────────────────────
                while True:
                    try:
                        source_fp, chunk_features = chunk_queue.get_nowait()
                    except queue.Empty:
                        break

                    remaining = num_samples - len(all_hard_bg)
                    if remaining > 0:
                        accepted = chunk_features[:remaining]
                        all_hard_bg.extend(accepted)
                        pbar.update(len(accepted))
                        if accepted:
                            contributing_images.add(source_fp)

                    pbar.set_postfix(_postfix())

                    if len(all_hard_bg) >= num_samples:
                        stop_event.set()

                if stop_event.is_set():
                    for f in list(futures):
                        f.cancel()

                if not futures:
                    if chunk_queue.empty():
                        break
                    continue

                done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    if not future.cancelled():
                        try:
                            future.result()   # stats already updated inside worker
                        except Exception:
                            pass

                pbar.set_postfix(_postfix())

                if len(all_hard_bg) >= num_samples:
                    stop_event.set()
                    for f in list(futures):
                        f.cancel()
                    break

                if not stop_event.is_set():
                    submit_next()

                if stop_event.is_set():
                    break

    with stats_lock:
        s = dict(stats)

    if len(all_hard_bg) < num_samples:
        print_warn(
            f"Only found {len(all_hard_bg)} crops (requested {num_samples}) "
            f"after scanning {s['images_processed']} images."
        )
    else:
        print(
            f" - Collected {len(all_hard_bg)} crops from "
            f"{s['images_processed']} images (target was {num_samples})"
        )

    print(
        f" - Mining stats: "
        f"candidates_tried={s['candidates']}, "
        f"detections_selected={s['detections']}, "
        f"feature_crops_generated={s['generated']}, "
        f"accepted_into_pool={len(all_hard_bg)}, "
        f"images_with_detections={s['images_with_det']}, "
        f"images_contributing={len(contributing_images)}"
    )

    return np.array(all_hard_bg, dtype=np.float32)