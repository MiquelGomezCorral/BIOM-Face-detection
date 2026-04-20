import os
import numpy as np

from maikol_utils.print_utils import print_separator, print_color, print_warn
from maikol_utils.file_utils import list_dir_files

import dataclasses
from src.config import Configuration
from src.data import balance_non_face_samples, precompute_feature_tensors, generate_all_features, compute_features_dataset, load_gb_images
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
        win_h = CONFIG.crop_size
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

    if os.path.exists(CONFIG.faces_np_path) and not CONFIG.force_features:
        print(f" - Loading precomputed face features from {CONFIG.faces_np_path}...")
        X_train_faces = np.load(CONFIG.faces_np_path)
        precomputed = precompute_feature_tensors(all_features)
        print(f" - Loaded face features for {X_train_faces.shape[0]} images.")
        print(f" - X_train_faces dtype={X_train_faces.dtype}, shape={X_train_faces.shape}")
    else:
        X_train_faces, precomputed = compute_features_dataset(all_faces, all_features, n_workers=CONFIG.max_cpu_cores)
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
            # Only add enough new bg samples to maintain balance with faces
            num_samples=prev_n_faces - len(prev_fp), 
            # num_samples=prev_n_faces, 
            bg_samples=bg_samples, 
            precomputed=precomputed,
            n_workers=CONFIG.max_cpu_cores,
            stop_check_interval=CONFIG.stop_check_interval
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
        clf, threshold = train_stage_early_stopping(X_train, y_train, max_cpu_cores=CONFIG.max_cpu_cores, max_features=CONFIG.max_features_per_stage, target_fpr=CONFIG.stage_target_fpr, target_tpr=CONFIG.target_tpr)
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
