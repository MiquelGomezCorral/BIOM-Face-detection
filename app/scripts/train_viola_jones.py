import os
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from maikol_utils.print_utils import print_separator, print_color, print_warn
from maikol_utils.file_utils import load_json, list_dir_files

from src.data import balance_non_face_samples, precompute_feature_tensors, generate_all_features, compute_features_dataset
from src.model import save_stages, CascadeSerializer, build_haar_cascade_from_stages


def train_viola_jones_stages(CONFIG):
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
    all_faces, n = list_dir_files(CONFIG.faces_path, recursive=True)
    print(f" - Found {n} files in {CONFIG.faces_path}\n")


    # =============== NO-FACES DATASET =============== 
    to_keep_labels = load_json(CONFIG.dataset_classes_path)

    # Download dataset without faces
    fo.config.dataset_zoo_dir = CONFIG.no_faces_path
    bg_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=to_keep_labels,
        max_samples=20000,
        # dataset_name="open-images-bg",  # ADDED: Forces a distinct dataset instance
        drop_existing_dataset=True      # ADDED: Clears old corrupted cache
    )
    bg_dataset = bg_dataset.filter_labels("ground_truth", F("label").is_in(to_keep_labels)) # REVERTED
    # ===============================================================
    #                  PRECOMPUTE FACE FEATURES
    # ===============================================================
    all_faces = np.random.choice(all_faces, size=CONFIG.max_faces, replace=False)
    print(f" - Using {len(all_faces)} files in {CONFIG.faces_path}")

    if os.path.exists(CONFIG.faces_np_path) and not CONFIG.force_features:
        print(f" - Loading precomputed face features from {CONFIG.faces_np_path}...")
        X_train_faces = np.load(CONFIG.faces_np_path)
        precomputed = precompute_feature_tensors(all_features)
        print(f" - Loaded face features for {X_train_faces.shape[0]} images.")
        print(f" - X_train_faces dtype={X_train_faces.dtype}, shape={X_train_faces.shape}")
    else:
        X_train_faces, precomputed = compute_features_dataset(all_faces, all_features)
        np.save(CONFIG.faces_np_path, X_train_faces)
        print(f" - Computed face features for {X_train_faces.shape[0]} images.")
        print(f" - X_train_faces dtype={X_train_faces.dtype}, shape={X_train_faces.shape}")


    stages, fpr_macro = generate_all_stages(
        CONFIG,
        X_train_faces=X_train_faces,
        bg_samples=[sample.filepath for sample in bg_dataset],
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



def generate_all_stages(CONFIG, X_train_faces, bg_samples, all_features, precomputed):
    stages = []
    prev_n_faces = len(X_train_faces) 
    n_bg_pre = len(X_train_faces) 
    n_features = X_train_faces.shape[1]
    prev_fp = np.empty((0, n_features), dtype=np.float32) # Start with no hard negatives
    fpr_macro = 1.0

    for stage_num in range(CONFIG. max_stages):
        print_separator(f"Training stage {stage_num + 1}/{CONFIG. max_stages}", sep_type="LONG")
        print_separator("Generating hard negative samples")

        X_train_bg = balance_non_face_samples(
            CONFIG,
            stages, 
            all_features,
            # Only add enough new bg samples to maintain balance with faces
            num_samples=prev_n_faces - len(prev_fp), 
            bg_samples=bg_samples, 
            precomputed=precomputed
        )

        if len(X_train_bg) == 0:
            print_color("No hard negative samples found. Stopping training.", color="green")
            break

        X_train = np.vstack((X_train_faces, X_train_bg, prev_fp))
        y_train = np.hstack((np.ones(len(X_train_faces)), np.zeros(len(X_train_bg)), np.zeros(len(prev_fp))))
        del X_train_bg 

        print_separator("Training")
        clf, threshold = train_stage_early_stopping(X_train, y_train)
        stages.append((clf, threshold))

        # Save current stage's hard negatives
        
        print_separator("Filtering: Hard Negative mining")
        decision_scores = clf.decision_function(X_train)
        passes_stage = decision_scores >= threshold
        X_train = X_train[passes_stage]
        y_train = y_train[passes_stage]
        
        prev_fp = X_train[y_train == 0]

        n_faces = np.sum(y_train == 1)
        n_bg = len(prev_fp)
        # macro FPR: Fi = Fi-1 * fpr_micro with F0 = 1.0  
        fpr_micro = n_bg / n_bg_pre
        fpr_macro *= fpr_micro

        save_stages(CONFIG, stages, stage_num + 1, fpr_macro)

        print(f" - Stage {stage_num + 1} used {len(clf.estimators_)} features.")
        print(f" - After stage {stage_num + 1}, {len(X_train)} / {prev_n_faces*2} samples remain for training.")
        print(f"   - {n_faces} faces")
        print(f"   - {n_bg} non-faces")
        print(f"   - {fpr_micro:.4f} micro false positive rate")
        print(f"   - {fpr_macro:.4f} macro false positive rate")
        
        n_bg_pre = n_bg
        prev_n_faces = n_faces
        # Stop conditions
        if n_bg == 0:
            print_color("No more negative samples left. Stopping training.", color="green")
            break
        
        if fpr_macro <= CONFIG.target_fpr:
            print_color(f"Reached target FPR of {CONFIG.target_fpr:.4f}. Stopping training.", color="green")
            break

    return stages, fpr_macro



def train_stage_early_stopping(X_train, y_train, max_features=200, target_tpr=0.995, target_fpr=0.50):
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
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=max_features, # set to a cap for bc no need to check all
    )
    clf.fit(X_train, y_train)

    X_faces = X_train[y_train == 1]
    X_bg = X_train[y_train == 0]

    print(' - Refining threshold and selecting features...')
    # .staged_decision_function evaluates the ensemble using 1 feature, then 2 features, etc.
    for i, (face_scores, bg_scores) in enumerate(zip(
        clf.staged_decision_function(X_faces),
        clf.staged_decision_function(X_bg)
    )):
        
        # Force the threshold to meet the 99.5% TPR target
        face_scores_sorted = np.sort(face_scores)
        drop_count = int(len(face_scores) * (1.0 - target_tpr))
        custom_threshold = face_scores_sorted[drop_count]

        # Check the False Positive Rate using this forced threshold
        false_positives = np.sum(bg_scores >= custom_threshold)
        fpr = false_positives / len(X_bg)

        print(f"   - Features: {i+1} | FPR: {fpr:.3f}")

        # 3. Stop early if we hit the FPR target!
        if fpr <= target_fpr:
            print(f" - Stage criteria met! Stopping at {i+1} features.")
            
            # Truncate the sklearn classifier to drop the unused extra features
            clf.estimators_ = clf.estimators_[:i+1]
            clf.estimator_weights_ = clf.estimator_weights_[:i+1]
            clf.estimator_errors_ = clf.estimator_errors_[:i+1]
            clf.classes_ = np.array([0, 1])
            
            return clf, custom_threshold

    print_warn("Max features reached without hitting FPR target.")
    return clf, custom_threshold
