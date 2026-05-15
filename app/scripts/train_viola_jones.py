import os
import numpy as np

from maikol_utils.print_utils import print_separator, print_color, print_warn
from maikol_utils.file_utils import list_dir_files

import dataclasses
from src.config import Configuration
from src.data import balance_non_face_samples, precompute_feature_tensors, generate_all_features, compute_features_dataset, load_gb_images, create_face_augmentor, create_bg_augmentor, create_face_augmentor, create_bg_augmentor
from src.model import (
    save_stages,
    save_stage_checkpoint,
    CascadeSerializer,
    build_haar_cascade_from_stages,
    CascadeClassifier,
    AdaBoostStumpClassifier,
    resume_training_from_checkpoint,
    test_cascade_fpr
)


def train_viola_jones_stages(CONFIG: Configuration):
    np.random.seed(CONFIG.seed)
    # ===============================================================
    #                   PREPARE DATA
    # ===============================================================
    print_separator("Preparing data", sep_type="LONG")

    all_features = generate_all_features(
        win_w = CONFIG.crop_size, 
        win_h = CONFIG.crop_size,
        edge_margin = CONFIG.feature_edge_margin,
        stride = CONFIG.feature_stride,
        include_square_features = CONFIG.include_square_features,
    )
    print(f" - Generated {len(all_features)} features.")

    # =============== FACES DATASET =============== 
    all_faces, n = list_dir_files(
        CONFIG.faces_vpc_path if CONFIG.use_vpc_faces else CONFIG.faces_train_path,
        recursive=True
    )
    if len(all_faces) > CONFIG.max_faces and CONFIG.max_faces > 0:
        all_faces = np.random.choice(all_faces, size=CONFIG.max_faces, replace=False)
    print(f" - Found {n} files in {CONFIG.faces_vpc_path if CONFIG.use_vpc_faces else CONFIG.faces_train_path }\n")


    # =============== NO-FACES DATASET =============== 
    bg_dataset = load_gb_images(CONFIG) 
    
    # ===============================================================
    #                  PRECOMPUTE FACE FEATURES
    # ===============================================================

    face_augmentor = create_face_augmentor(CONFIG)
    if os.path.exists(CONFIG.faces_np_path) and not CONFIG.force_features:
        print(f" - Loading precomputed face features from {CONFIG.faces_np_path}...")
        X_train_faces = np.load(CONFIG.faces_np_path)
        precomputed = precompute_feature_tensors(all_features)
        print(f" - Loaded face features for {X_train_faces.shape[0]} images.")
        print(f" - X_train_faces dtype={X_train_faces.dtype}, shape={X_train_faces.shape}")
    else:
        X_train_faces, precomputed = compute_features_dataset(all_faces, all_features, n_workers=CONFIG.max_cpu_cores, augment_fn=face_augmentor)
        np.save(CONFIG.faces_np_path, X_train_faces)
        print(f" - Computed face features for {X_train_faces.shape[0]} images.")
        print(f" - X_train_faces dtype={X_train_faces.dtype}, shape={X_train_faces.shape}")


    stages, fpr_macro = generate_all_stages(
        CONFIG,
        X_train_faces=X_train_faces,
        bg_samples=bg_dataset,
        all_features=all_features,
        precomputed=precomputed
    )
    print_color(f" - Finished training {len(stages)} stages with final macro FPR: {fpr_macro:.6f}", color="green")

    # ===============================================================
    #                  PRECOMPUTE FACE FEATURES
    # ===============================================================
    haar_cascade = build_haar_cascade_from_stages(
        stages_output=stages,
        all_features=all_features,
        width=CONFIG.crop_size,
        height=CONFIG.crop_size,
        cascade_type="trained_adaboost_stages",
        feature_type="HAAR",
    )

    print_separator(f"FINAL CASCADE", sep_type="LONG")
    print(f" - Stages: {len(haar_cascade.stages)}")
    print(f" - Features used: {len(haar_cascade.features)}")
    print(f" - Window size: {haar_cascade.width}x{haar_cascade.height}")
    print(haar_cascade)


    cascade_path = os.path.join(CONFIG.computed_haar_cascades, f"stages_vj-{fpr_macro:.4f}.xml")
    CascadeSerializer.save(haar_cascade, cascade_path)

    # loaded_cascade = CascadeSerializer.load(cascade_path)


def generate_all_stages(CONFIG: Configuration, X_train_faces, bg_samples, all_features, precomputed):

    (
        start_stage, stages, 
        fpr_macro_thr, 
        prev_n_faces, n_bg_pre, 
        prev_fp, n_features
    ) = resume_training_from_checkpoint(
        CONFIG, X_train_faces
    )
    
    bg_augmentor = create_bg_augmentor(CONFIG)
    cascade = build_haar_cascade_from_stages(
        stages_output=stages,
        all_features=all_features,
        width=CONFIG.crop_size,
        height=CONFIG.crop_size,
        cascade_type="trained_adaboost_stages",
        feature_type="HAAR",
    )
    classifier = CascadeClassifier(dataclasses.replace(CONFIG, stride=CONFIG.stride), cascade)
    for stage_num in range(start_stage, CONFIG.max_stages):
        print_separator(f"Training stage {stage_num + 1}/{CONFIG.max_stages}", sep_type="LONG")
        print_separator("Generating hard negative samples")


        X_train_bg = balance_non_face_samples(
            classifier=classifier,
            num_samples=prev_n_faces - len(prev_fp) if CONFIG.preserve_fp else prev_n_faces, 
            bg_samples=bg_samples, 
            precomputed=precomputed,
            n_workers=CONFIG.max_cpu_cores,
            crop_size=CONFIG.crop_size,
            max_detections_per_image=1500,     
            stop_check_interval=CONFIG.stop_check_interval,
            augment_fn=None,
        )

        if len(X_train_bg) == 0:
            print_color("No hard negative samples found. Stopping training.", color="green")
            break

        n_bg_pre = len(X_train_bg)

        if CONFIG.preserve_fp:
            X_train = np.vstack((X_train_faces, X_train_bg, prev_fp))
            y_train = np.hstack((np.ones(len(X_train_faces)), np.zeros(len(X_train_bg)), np.zeros(len(prev_fp))))
        else:
            X_train = np.vstack((X_train_faces, X_train_bg))
            y_train = np.hstack((np.ones(len(X_train_faces)), np.zeros(len(X_train_bg))))

        del X_train_bg 

        print_separator("Training")
        stage_fpr = CONFIG.get_stage_fpr(stage_num + 1)
        print(f" - Stage {stage_num + 1} target FPR: {stage_fpr:.2%}")
        clf, threshold = train_stage_early_stopping(X_train, y_train, max_cpu_cores=CONFIG.max_cpu_cores, max_features=CONFIG.max_features_per_stage, target_fpr=stage_fpr, target_tpr=CONFIG.target_tpr)
        stages.append((clf, threshold))

        cascade = build_haar_cascade_from_stages(
            stages_output=stages,
            all_features=all_features,
            width=CONFIG.crop_size,
            height=CONFIG.crop_size,
            cascade_type="trained_adaboost_stages",
            feature_type="HAAR",
        )
        classifier = CascadeClassifier(dataclasses.replace(CONFIG, stride=CONFIG.stride), cascade)
        fpr_macro = test_cascade_fpr(CONFIG, classifier)


        # ===========================
        
        verify_cascade_vs_adaboost(clf, threshold, cascade, X_train, y_train, stage_idx=stage_num)

        # ===========================


        # Save current stage's hard negatives
        
        print_separator("Filtering: Hard Negative mining")
        decision_scores = clf.decision_function(X_train)
        passes_stage = decision_scores >= threshold
        X_train = X_train[passes_stage]
        y_train = y_train[passes_stage]
        
        prev_fp = X_train[y_train == 0]

        n_faces = np.sum(y_train == 1)
        n_bg = len(prev_fp)
        # Stage FPR = negatives that survive this stage / negatives that entered this stage.
        # Macro FPR is the cumulative product across stages.
        fpr_micro = n_bg / n_bg_pre if n_bg_pre > 0 else 0.0
        fpr_macro_thr *= fpr_micro

        save_stages(CONFIG, stages, stage_num + 1, fpr_macro, all_features)
        save_stage_checkpoint(
            CONFIG,
            {
                "stages": stages,
                "stage_num": stage_num + 1,
                "fpr_macro_thr": fpr_macro_thr,
                "prev_fp": prev_fp,
                "prev_n_faces": n_faces,
                "n_bg_pre": n_bg,
                "n_features": n_features,
            },
        )

        print(f" - Stage {stage_num + 1} used {len(clf.estimators_)} features.")
        print(f" - After stage {stage_num + 1}, {len(X_train)} / {prev_n_faces*2} samples remain for training.")
        print(f"   - {n_faces} faces")
        print(f"   - {n_bg} non-faces")
        print(f"   - {fpr_micro:.4f} micro false positive rate")
        print(f"   - {fpr_macro_thr:.4f} thr macro false positive rate")
        print_color(f"   - Tested macro FPR on training faces: {fpr_macro:.10f}", color="blue")
        
        n_bg_pre = n_bg
        prev_n_faces = n_faces
        # Stop conditions
        # if n_bg == 0:
        #     print_color("No more negative samples left. Stopping training.", color="green")
        #     break
        

        if fpr_macro <= CONFIG.target_fpr:
            print_color(f"Reached target FPR of {CONFIG.target_fpr:.10f}. Stopping training.", color="green")
            break

    return stages, fpr_macro_thr



def train_stage_early_stopping(X_train, y_train, max_features=200, target_tpr=0.98, target_fpr=0.50, max_cpu_cores=16):
    """
    train_stage_for_tpr trains an AdaBoost classifier and determines a custom threshold to achieve the target true positive rate (TPR) on the training data.

    X_train shape: (num_images, num_features) - precalculated feature values
    y_train shape: (num_images,) - 1 for face, 0 for background


    returns:
    - clf: the trained AdaBoost classifier
    - custom_threshold: the decision function threshold to achieve the target TPR on the training data
    passes_stage = clf.decision_function(X_test) >= custom_threshold
    """
    print(' - Fitting AdaBoost with early stopping...')
    # clf = AdaBoostClassifier(
    #     estimator=DecisionTreeClassifier(max_depth=1),
    #     n_estimators=max_features, # set to a cap for bc no need to check all
    # )
    clf = AdaBoostStumpClassifier(n_estimators=max_features, n_jobs=max_cpu_cores)
    clf.fit(
        X_train,
        y_train,
        target_tpr=target_tpr,
        target_fpr=target_fpr,
        log_fpr=True,
    )

    custom_threshold = clf.custom_threshold_
    if clf.stop_feature_count_ is None:
        print_warn("Max features reached without hitting FPR target.")

    return clf, custom_threshold


def verify_cascade_vs_adaboost(clf, threshold, cascade, X_train, y_train, stage_idx):
    """
    After training a stage, verify that the cascade kernel produces
    identical accept/reject decisions to clf.decision_function on the
    same feature vectors.

    cascade: the HaarCascade object built from stages so far (including this stage)
    stage_idx: 0-based index of the stage just trained (used to isolate it)
    """
    import numpy as np

    X_neg = X_train[y_train == 0]
    X_pos = X_train[y_train == 1]

    # ── 1. AdaBoost scores ──────────────────────────────────────────────────
    scores_neg = clf.decision_function(X_neg)
    scores_pos = clf.decision_function(X_pos)

    adaboost_fpr = float(np.mean(scores_neg >= threshold))
    adaboost_tpr = float(np.mean(scores_pos >= threshold))

    print(f"\n[Stage {stage_idx + 1}] AdaBoost direct:")
    print(f"  threshold      : {threshold:.6f}")
    print(f"  score range neg: [{scores_neg.min():.4f}, {scores_neg.max():.4f}]")
    print(f"  score range pos: [{scores_pos.min():.4f}, {scores_pos.max():.4f}]")
    print(f"  TPR            : {adaboost_tpr:.4f}  (target ~{0.99:.2f})")
    print(f"  FPR            : {adaboost_fpr:.4f}  (target ~{0.50:.2f})")

    # ── 2. Simulate the cascade stage on X_train feature vectors ───────────
    # Re-implement the stage sum using the flat arrays from the classifier.
    # This bypasses image loading entirely — uses the same feature vectors
    # AdaBoost trained on, so any mismatch is purely in cascade construction.
    stage = cascade.stages[stage_idx]
    n_samples = X_neg.shape[0]
    stage_sums = np.zeros(n_samples, dtype=np.float64)

    for wc in stage.weak_classifiers:
        feat_idx = wc.feature_id
        if feat_idx >= X_neg.shape[1]:
            print(f"  [ERROR] feature_id {feat_idx} out of range ({X_neg.shape[1]} features)")
            continue

        feat_vals = X_neg[:, feat_idx].astype(np.float64)

        # NOTE: X_neg is already std-normalised, so clf_thr must NOT be
        # multiplied by std here. The cascade kernel does feat < thr * std,
        # but since feat here = raw_feat / std already, that is equivalent
        # to (raw_feat / std) < thr, i.e. feat < thr — no std factor.
        goes_left = feat_vals < wc.threshold
        stage_sums += np.where(goes_left, wc.left_value, wc.right_value)

    cascade_accepts_neg = stage_sums >= stage.threshold
    cascade_fpr = float(np.mean(cascade_accepts_neg))

    # Same for positives
    stage_sums_pos = np.zeros(len(X_pos), dtype=np.float64)
    for wc in stage.weak_classifiers:
        feat_idx = wc.feature_id
        if feat_idx >= X_pos.shape[1]:
            continue
        feat_vals = X_pos[:, feat_idx].astype(np.float64)
        goes_left = feat_vals < wc.threshold
        stage_sums_pos += np.where(goes_left, wc.left_value, wc.right_value)

    cascade_tpr = float(np.mean(stage_sums_pos >= stage.threshold))

    print(f"\n[Stage {stage_idx + 1}] Cascade simulation on same features:")
    print(f"  stage.threshold: {stage.threshold:.6f}")
    print(f"  stage_sum range neg: [{stage_sums.min():.4f}, {stage_sums.max():.4f}]")
    print(f"  stage_sum range pos: [{stage_sums_pos.min():.4f}, {stage_sums_pos.max():.4f}]")
    print(f"  TPR              : {cascade_tpr:.4f}")
    print(f"  FPR              : {cascade_fpr:.4f}")

    # ── 3. Agreement check ──────────────────────────────────────────────────
    adaboost_accepts_neg = scores_neg >= threshold
    agreement = float(np.mean(adaboost_accepts_neg == cascade_accepts_neg))
    disagreements = int(np.sum(adaboost_accepts_neg != cascade_accepts_neg))

    print(f"\n[Stage {stage_idx + 1}] Agreement:")
    print(f"  sample-level agreement: {agreement:.6f}  ({disagreements} disagreements out of {n_samples})")

    if disagreements > 0:
        # Show a few disagreeing samples to understand the pattern
        diff_idx = np.where(adaboost_accepts_neg != cascade_accepts_neg)[0][:5]
        print(f"  First disagreeing samples:")
        for idx in diff_idx:
            print(f"    sample {idx}: adaboost_score={scores_neg[idx]:.4f} (>={threshold:.4f} = {adaboost_accepts_neg[idx]})"
                  f" | cascade_sum={stage_sums[idx]:.4f} (>={stage.threshold:.4f} = {cascade_accepts_neg[idx]})")

    if agreement < 0.999:
        print(f"\n  [WARNING] AdaBoost and cascade disagree on >{1-agreement:.1%} of samples.")
        print(f"  Likely cause: stage.threshold != clf threshold, or left/right_value sign/scale mismatch.")
        print(f"  Check: clf threshold={threshold:.6f} vs stage.threshold={stage.threshold:.6f}")
        # Check scale of left/right values vs alpha * ±1
        wc0 = stage.weak_classifiers[0]
        alpha0 = float(clf.estimator_weights_[0])
        print(f"  First weak clf: left_value={wc0.left_value:.6f}, right_value={wc0.right_value:.6f}")
        print(f"  Expected from alpha: ±{alpha0:.6f}")
    else:
        print(f"  [OK] Cascade and AdaBoost are in perfect agreement on training data.")