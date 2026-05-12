import threading
import queue
import numpy as np
from tqdm import tqdm
from concurrent.futures import CancelledError
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from maikol_utils.print_utils import print_warn

from src.model import CascadeClassifier
from .crops import get_image_crops_from_list
from .features import extract_features_batch

def balance_non_face_samples(
        classifier: CascadeClassifier,
        num_samples,
        bg_samples,
        precomputed,
        n_workers=8,
        stop_check_interval=100,
    predictor_halve_size=True,
    predictor_halve_size_factor=4,
    augment_fn=None,
    ):
    
    stop_event = threading.Event()

    chunk_queue = queue.Queue(maxsize=max(1, n_workers * 4))

    def process_image(filepath):
        if stop_event.is_set():
            return {
                "filepath": filepath,
                "candidates": 0,
                "detections": 0,
                "generated": 0,
            }
        fps, candidates = classifier.predict_no_merge(
            img_path=filepath,
            return_candidate_count=True,
            halve_size=predictor_halve_size,
            halve_size_factor=predictor_halve_size_factor,
        )
        if not fps:
            return {
                "filepath": filepath,
                "candidates": candidates,
                "detections": 0,
                "generated": 0,
            }
        crops = get_image_crops_from_list(
            fps,
            img_path=filepath,
            read_scale_factor=(predictor_halve_size_factor if predictor_halve_size else 1),
        )
        if not crops:
            return {
                "filepath": filepath,
                "candidates": candidates,
                "detections": len(fps),
                "generated": 0,
            }
        # Process crops in chunks and only check stop_event between chunks.
        # This keeps throughput high and allows controlled overshoot.
        crops = np.random.permutation(crops) 
        produced = 0
        for start in range(0, len(crops), stop_check_interval):
            if stop_event.is_set():
                break
            end = min(start + stop_check_interval, len(crops))
            chunk = crops[start:end]
            chunk_features = list(
                extract_features_batch([c["img"] for c in chunk], precomputed=precomputed, augment_fn=augment_fn)
            )
            if chunk_features:
                # Stream chunk results so the main thread can update tqdm often.
                chunk_queue.put((filepath, chunk_features))
                produced += len(chunk_features)
        return {
            "filepath": filepath,
            "candidates": candidates,
            "detections": len(fps),
            "generated": produced,
        }

    np.random.shuffle(bg_samples)

    print(
        f" - Generating {num_samples} hard negative samples "
        f"({n_workers} workers, stop-check every {stop_check_interval} crops)..."
    )
    all_hard_bg = []
    images_processed = 0
    generated_total = 0
    total_candidates = 0
    total_detections = 0
    images_with_detections = 0
    contributing_images = set()

    with tqdm(total=num_samples, desc="Hard negatives", unit="crops") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit lazily — only keep a small buffer in flight at a time
            fp_iter = iter(bg_samples)
            buffer_size = n_workers * 2
            futures = {}

            def submit_next():
                fp = next(fp_iter, None)
                if fp is None or stop_event.is_set():
                    return
                f = executor.submit(process_image, fp)
                futures[f] = fp

            # Fill initial buffer
            for _ in range(buffer_size):
                submit_next()

            while futures or not chunk_queue.empty():
                # Drain streamed chunks first so progress updates are responsive.
                drained_any = False
                while True:
                    try:
                        source_filepath, chunk_features = chunk_queue.get_nowait()
                    except queue.Empty:
                        break

                    drained_any = True
                    generated_total += len(chunk_features)

                    remaining = num_samples - len(all_hard_bg)
                    if remaining > 0:
                        accepted = chunk_features[:remaining]
                        all_hard_bg.extend(accepted)
                        pbar.update(len(accepted))
                        if accepted:
                            contributing_images.add(source_filepath)

                    pbar.set_postfix(
                        imgs=images_processed,
                        imgs_contrib=len(contributing_images),
                        cand=total_candidates,
                        det=total_detections,
                        accepted=len(all_hard_bg),
                        generated=generated_total,
                    )

                    if len(all_hard_bg) >= num_samples:
                        stop_event.set()

                if stop_event.is_set() and futures:
                    for pending in list(futures):
                        pending.cancel()

                if not futures:
                    if not drained_any:
                        break
                    continue

                done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    images_processed += 1
                    if not future.cancelled():
                        try:
                            result = future.result()
                            total_candidates += result["candidates"]
                            total_detections += result["detections"]
                            if result["detections"] > 0:
                                images_with_detections += 1
                        except CancelledError:
                            # Expected when we stop early and cancel pending work.
                            pass

                    pbar.set_postfix(
                        imgs=images_processed,
                        imgs_contrib=len(contributing_images),
                        cand=total_candidates,
                        det=total_detections,
                        accepted=len(all_hard_bg),
                        generated=generated_total,
                    )

                    if len(all_hard_bg) >= num_samples:
                        stop_event.set()
                        # Cancel queued work that has not started yet.
                        for pending in list(futures):
                            pending.cancel()
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

    print(
        " - Mining stats: "
        f"candidates_tried={total_candidates}, "
        f"detections_selected={total_detections}, "
        f"feature_crops_generated={generated_total}, "
        f"accepted_into_pool={len(all_hard_bg)}, "
        f"images_with_detections={images_with_detections}, "
        f"images_contributing={len(contributing_images)}"
    )

    return np.array(all_hard_bg, dtype=np.float32)

