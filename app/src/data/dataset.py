import threading
import dataclasses
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from maikol_utils.print_utils import print_warn

from src.model import build_haar_cascade_from_stages, CascadeClassifier
from .crops import get_image_crops_from_list
from .features import extract_features_batch

def balance_non_face_samples(
        CONFIG,
        all_features,
        stages,
        num_samples,
        bg_samples,
        max_iterations=10000, 
        n_workers=8
    ):
    cascade = build_haar_cascade_from_stages(
        stages_output=stages,
        all_features=all_features,
        width=CONFIG.crop_size,
        height=CONFIG.crop_size,
        cascade_type="trained_adaboost_stages",
        feature_type="HAAR",
    )
    classifier = CascadeClassifier(dataclasses.replace(CONFIG, stride=4), cascade)

    stop_event = threading.Event()

    def process_image(filepath, max_crops):
        if stop_event.is_set():
            return []
        fps = classifier.predict_no_merge(img_path=filepath)
        if not fps:
            return []
        crops = get_image_crops_from_list(fps, img_path=filepath)
        if not crops:
            return []
        # Cap how many crops we extract to avoid overshoot
        crops = crops[:max_crops]
        return list(extract_features_batch([c["img"] for c in crops]))

    sample_size = min(max_iterations, len(bg_samples))
    filepaths = np.random.choice(bg_samples, size=sample_size, replace=False)

    print(f" - Generating {num_samples} hard negative samples ({n_workers} workers)...")
    all_hard_bg = []
    images_processed = 0

    with tqdm(total=num_samples, desc="Hard negatives", unit="crops") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit lazily — only keep a small buffer in flight at a time
            fp_iter = iter(filepaths)
            buffer_size = n_workers * 2
            futures = {}

            def submit_next():
                fp = next(fp_iter, None)
                if fp is None or stop_event.is_set():
                    return
                remaining = num_samples - len(all_hard_bg)
                f = executor.submit(process_image, fp, remaining)
                futures[f] = fp

            # Fill initial buffer
            for _ in range(buffer_size):
                submit_next()

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    crops = future.result()
                    images_processed += 1

                    if crops:
                        all_hard_bg.extend(crops)
                        pbar.update(len(crops))

                    pbar.set_postfix(imgs=images_processed, crops=len(all_hard_bg))

                    if len(all_hard_bg) >= num_samples:
                        stop_event.set()
                        break

                    submit_next()  # refill buffer one at a time

                if stop_event.is_set():
                    break

    if len(all_hard_bg) < num_samples:
        print_warn(
            f"Only found {len(all_hard_bg)} crops (requested {num_samples}) "
            f"after processing {images_processed} images."
        )
    else:
        print(f" - Collected {len(all_hard_bg)} crops from {images_processed} images (target was {num_samples})")

    return np.array(all_hard_bg, dtype=np.float32)

