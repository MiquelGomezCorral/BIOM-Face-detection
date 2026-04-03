"""
adaboost_stump.py
=================
Parallelised AdaBoost restricted to decision stumps.

Drop-in replacement for:

    AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=N,
    )

Key differences / improvements over the sklearn version:
  * Feature search is parallelised across **all available CPUs** using
    joblib threads (shared memory - no serialisation overhead for large X).
  * The inner search is pure NumPy (argsort + cumsum) - no per-feature
    sklearn estimator construction.
  * Exposes the same attributes the Viola-Jones pipeline relies on:
      estimators_, estimator_weights_, estimator_errors_, classes_,
      staged_decision_function(), decision_function(), predict().
  * Each estimator in ``estimators_`` carries a ``tree_`` object whose
    layout is identical to a fitted sklearn DecisionTreeClassifier
    (max_depth=1), so _extract_stump_params() works unchanged.
"""

from __future__ import annotations

import multiprocessing
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


# ---------------------------------------------------------------------------
# sklearn tree_ sentinel constants  (same values sklearn uses internally)
# ---------------------------------------------------------------------------
_TREE_LEAF = -1
_TREE_UNDEFINED = -2


# ===========================================================================
# _FakeTree  -  mimics sklearn's DecisionTreeClassifier.tree_ for a stump
# ===========================================================================

class _FakeTree:
    """
    Reproduces the subset of the sklearn ``tree_`` Cython object that
    _extract_stump_params (and anything calling .tree_ on a stump) reads.

    Node layout for a depth-1 tree (3 nodes):
        0 = root        (internal, splits on feature_idx at threshold)
        1 = left child  (leaf, reached when x <= threshold)
        2 = right child (leaf, reached when x  > threshold)
    """

    def __init__(
        self,
        feature_idx: int,
        threshold: float,
        left_predicts_one: bool,   # True  → left leaf predicts class 1
        right_predicts_one: bool,  # False → right leaf predicts class 0
    ) -> None:

        # ---- topology ----
        self.feature = np.array(
            [feature_idx, _TREE_LEAF, _TREE_LEAF], dtype=np.intp
        )
        self.threshold = np.array(
            [threshold, float(_TREE_UNDEFINED), float(_TREE_UNDEFINED)],
            dtype=np.float64,
        )
        self.children_left  = np.array([1, _TREE_LEAF, _TREE_LEAF], dtype=np.intp)
        self.children_right = np.array([2, _TREE_LEAF, _TREE_LEAF], dtype=np.intp)

        # ---- node values  shape (n_nodes=3, n_outputs=1, n_classes=2) ----
        # Encode predicted class as [neg_weight, pos_weight].
        # Using {0.0, 1.0} keeps argmax() unambiguous.
        def _leaf(predicts_one: bool) -> np.ndarray:
            return np.array([[[0.0, 1.0]]] if predicts_one else [[[1.0, 0.0]]],
                            dtype=np.float64)

        self.value = np.concatenate(
            [
                np.array([[[0.5, 0.5]]], dtype=np.float64),  # root - not used
                _leaf(left_predicts_one),
                _leaf(right_predicts_one),
            ],
            axis=0,
        )  # → shape (3, 1, 2)

        # ---- misc sklearn attributes (read by some inspection code) ----
        self.n_node_samples = np.zeros(3, dtype=np.intp)
        self.n_outputs      = 1
        self.n_classes      = np.array([2], dtype=np.intp)
        self.node_count     = 3


# ===========================================================================
# DecisionStump  -  one weak learner
# ===========================================================================

class DecisionStump:
    """
    A binary decision stump that is fully compatible with sklearn's
    ``DecisionTreeClassifier(max_depth=1)`` interface, specifically the
    attributes read by ``build_haar_cascade_from_stages`` and
    ``_extract_stump_params``.

    Parameters
    ----------
    feature_idx : int
        Column index in X that this stump splits on.
    threshold : float
        Decision boundary.
    polarity : {1, -1}
        *  1 → predict class 1 when x  > threshold  (left leaf = 0)
        * -1 → predict class 1 when x <= threshold  (left leaf = 1)
    """

    def __init__(self, feature_idx: int, threshold: float, polarity: int) -> None:
        if polarity not in (1, -1):
            raise ValueError("polarity must be 1 or -1")

        self.feature_idx = feature_idx
        self.threshold   = threshold
        self.polarity    = polarity

        # sklearn-compat attributes
        self.classes_   = np.array([0, 1])
        self.n_classes_ = 2

        # Build the fake tree used by _extract_stump_params
        #   left child  (x <= threshold):  predicts 1 only when polarity == -1
        #   right child (x  > threshold):  predicts 1 only when polarity ==  1
        self.tree_ = _FakeTree(
            feature_idx       = feature_idx,
            threshold         = threshold,
            left_predicts_one = (polarity == -1),
            right_predicts_one= (polarity ==  1),
        )

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels {0, 1} for each row in X."""
        x = X[:, self.feature_idx]
        if self.polarity == 1:
            return (x > self.threshold).astype(np.int32)
        return (x <= self.threshold).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return hard 0/1 probability matrix of shape (n_samples, 2)."""
        pred = self.predict(X)
        out = np.zeros((len(X), 2), dtype=np.float64)
        out[pred == 1, 1] = 1.0
        out[pred == 0, 0] = 1.0
        return out

    def __repr__(self) -> str:
        cmp = ">" if self.polarity == 1 else "<="
        return (
            f"DecisionStump(feature={self.feature_idx}, "
            f"threshold={self.threshold:.6f}, predict_1_when_x {cmp} threshold)"
        )


# ===========================================================================
# Module-level helpers  (must be at module scope for joblib pickling)
# ===========================================================================

def _best_threshold_for_feature(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    pos_total: float,
    neg_total: float,
) -> Tuple[float, int, float]:
    """
    Find the (threshold, polarity, weighted_error) that minimise the weighted
    misclassification error for a single feature vector ``x``.

    Uses a single sort + cumulative-sum scan: O(n log n) per feature.
    """
    sort_idx = np.argsort(x, kind="quicksort")
    x_s = x[sort_idx]
    w_s = w[sort_idx]
    is_pos = (y[sort_idx] == 1).astype(np.float64)

    cum_pos = np.cumsum(w_s * is_pos)        # positives to the left of split
    cum_neg = np.cumsum(w_s * (1.0 - is_pos))  # negatives to the left

    # Valid split points: where consecutive x values differ
    valid = x_s[:-1] != x_s[1:]
    if not np.any(valid):
        # Constant feature - no useful split possible
        err = float(min(pos_total, neg_total))
        return float(x_s[0]) - 1.0, 1, err

    # ---- polarity  1: predict class 1 when x > threshold ----
    #   miss: positives on the left + negatives on the right
    err1 = cum_pos[:-1] + (neg_total - cum_neg[:-1])

    # ---- polarity -1: predict class 1 when x <= threshold ----
    #   miss: negatives on the left + positives on the right
    err2 = cum_neg[:-1] + (pos_total - cum_pos[:-1])

    # Mask out invalid positions
    _INF = np.inf
    err1 = np.where(valid, err1, _INF)
    err2 = np.where(valid, err2, _INF)

    i1 = int(np.argmin(err1))
    i2 = int(np.argmin(err2))

    if err1[i1] <= err2[i2]:
        thresh = (x_s[i1] + x_s[i1 + 1]) * 0.5
        return float(thresh), 1, float(err1[i1])

    thresh = (x_s[i2] + x_s[i2 + 1]) * 0.5
    return float(thresh), -1, float(err2[i2])


def _search_chunk(
    X_chunk: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    global_indices: np.ndarray,
    pos_total: float,
    neg_total: float,
) -> Tuple[int, float, int, float]:
    """
    Search for the best stump inside *one* column-slice of X.

    Called by every parallel worker; returns the local winner as
    ``(global_feature_idx, threshold, polarity, weighted_error)``.
    """
    best_error  = np.inf
    best_feat   = int(global_indices[0])
    best_thresh = 0.0
    best_pol    = 1

    n_chunk_features = X_chunk.shape[1]
    for local_idx in range(n_chunk_features):
        thresh, pol, err = _best_threshold_for_feature(
            X_chunk[:, local_idx], y, w, pos_total, neg_total
        )
        if err < best_error:
            best_error  = err
            best_feat   = int(global_indices[local_idx])
            best_thresh = thresh
            best_pol    = pol

    return best_feat, best_thresh, best_pol, float(best_error)


# ===========================================================================
# AdaBoostStumpClassifier
# ===========================================================================

class AdaBoostStumpClassifier:
    """
    AdaBoost (SAMME, discrete) restricted to decision stumps with
    parallelised feature search.

    This is a **drop-in replacement** for::

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=N,
        )

    Just swap the class and keep every other line of your code unchanged.

    Sklearn-compatible attributes
    -----------------------------
    estimators_         : list[DecisionStump]
    estimator_weights_  : ndarray[float64]   - alpha values
    estimator_errors_   : ndarray[float64]   - weighted errors
    classes_            : ndarray([0, 1])

    Sklearn-compatible methods
    --------------------------
    fit(X, y)
    decision_function(X)
    staged_decision_function(X)   # generator, yields (n_samples,) arrays
    predict(X)
    predict_proba(X)

    Parameters
    ----------
    estimator :
        Accepted for drop-in compatibility; always ignored - stumps are
        always used.
    n_estimators : int
        Maximum number of boosting rounds (weak learners).
    learning_rate : float
        Shrinks each weak learner's contribution.  Default 1.0.
    n_jobs : int
        Worker threads for the parallel feature search.
        -1 (default) → use all logical CPUs.
    """

    def __init__(
        self,
        estimator=None,           # ignored, kept for sklearn API compat
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        n_jobs: int = -1,
    ) -> None:
        self.estimator     = estimator   # stored but never used
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.n_jobs        = (
            int(multiprocessing.cpu_count() * 4/5) if n_jobs == -1 else max(1, n_jobs)
        )

        # ---- sklearn-compatible output attributes (pre-declared) ----
        self.estimators_:        List[DecisionStump] = []
        self.estimator_weights_: np.ndarray          = np.array([], dtype=np.float64)
        self.estimator_errors_:  np.ndarray          = np.array([], dtype=np.float64)
        self.classes_:           np.ndarray          = np.array([0, 1])

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "AdaBoostStumpClassifier":
        """
        Fit the AdaBoost ensemble.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Pre-computed feature matrix (e.g. Haar feature responses).
        y : ndarray, shape (n_samples,)
            Binary labels {0, 1}.
        sample_weight : ndarray, shape (n_samples,), optional
            Initial per-sample weights.  Uniform if None.
        """
        n_samples, n_features = X.shape
        y = np.asarray(y, dtype=np.int32)

        # Initialise sample weights
        if sample_weight is None:
            w = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64).copy()
            w /= w.sum()

        self.classes_   = np.array([0, 1])
        self.estimators_ = []
        weights_list: List[float] = []
        errors_list:  List[float] = []

        print(f"\n - Starting AdaBoost training: {self.n_estimators} stumps target, "
              f"{n_samples} samples, {n_features} features\n")

        for round_idx in tqdm(range(self.n_estimators), desc="Training stumps", unit="stump"):
            pos_total = float(np.dot(w, y == 1))
            neg_total = float(np.dot(w, y == 0))

            # ---- Find the globally best stump across all features ----
            feat_idx, thresh, polarity, error = self._parallel_best_stump(
                X, y, w, pos_total, neg_total
            )

            # Clamp to avoid log(0) / division by zero
            error = float(np.clip(error, 1e-10, 1.0 - 1e-10))

            # Alpha (SAMME, binary: no log(K-1) term since log(1)=0)
            alpha = self.learning_rate * 0.5 * np.log((1.0 - error) / error)

            stump = DecisionStump(feat_idx, thresh, polarity)

            # ---- Exponential weight update ----
            # w_i *= exp(-alpha * y_sign_i * h_sign_i)
            # where  y_sign, h_sign  ∈ {-1, +1}
            pred   = stump.predict(X)
            y_sign = (2 * y    - 1).astype(np.float64)
            h_sign = (2 * pred - 1).astype(np.float64)
            w *= np.exp(-alpha * y_sign * h_sign)
            w /= w.sum()

            self.estimators_.append(stump)
            weights_list.append(alpha)
            errors_list.append(error)

            # stumps_remaining = self.n_estimators - round_idx - 1
            # tqdm.write(
            #     f"  - Stump {round_idx + 1}: feature={feat_idx}, "
            #     f"threshold={thresh:.6f}, error={error:.6f}, alpha={alpha:.6f} "
            #     f"| {stumps_remaining} remaining"
            # )

        print(f" - AdaBoost training complete! {len(self.estimators_)} stumps trained.")

        self.estimator_weights_ = np.array(weights_list, dtype=np.float64)
        self.estimator_errors_  = np.array(errors_list,  dtype=np.float64)
        return self

    # ------------------------------------------------------------------
    # Parallel search internals
    # ------------------------------------------------------------------

    def _parallel_best_stump(
        self,
        X:         np.ndarray,
        y:         np.ndarray,
        w:         np.ndarray,
        pos_total: float,
        neg_total: float,
    ) -> Tuple[int, float, int, float]:
        """
        Evaluate every feature in parallel and return the global winner.

        Uses the 'threads' backend so that the large X matrix is shared
        directly between workers without any serialisation overhead.
        NumPy operations (argsort, cumsum, argmin) release the GIL, so
        true parallelism is achieved for the numerically heavy parts.
        """
        n_features = X.shape[1]
        n_workers  = min(self.n_jobs, n_features)

        # Partition feature column indices into n_workers roughly equal chunks
        chunks = [c for c in np.array_split(np.arange(n_features), n_workers) if len(c)]

        # print(f" - Searching {n_features} features across {len(chunks)} parallel workers")

        results: List[Tuple[int, float, int, float]] = Parallel(
            n_jobs    = n_workers,
            prefer    = "threads",   # shared memory, no pickle cost
            backend   = "threading",
        )(
            delayed(_search_chunk)(X[:, chunk], y, w, chunk, pos_total, neg_total)
            for chunk in chunks
        )

        # r[3] is the weighted error - pick the chunk winner with lowest error
        return min(results, key=lambda r: r[3])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Cumulative AdaBoost score for each sample.
        Positive → class 1 (face), Negative → class 0 (background).

        Returns
        -------
        scores : ndarray, shape (n_samples,)
        """
        scores = np.zeros(X.shape[0], dtype=np.float64)
        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            scores += alpha * (2 * stump.predict(X) - 1)
        return scores

    def staged_decision_function(
        self, X: np.ndarray
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield cumulative scores after each successive weak learner,
        matching sklearn's ``AdaBoostClassifier.staged_decision_function``.

        Each yielded array has shape ``(n_samples,)`` and represents the
        ensemble score using 1, 2, … weak classifiers respectively.

        Yields
        ------
        scores : ndarray, shape (n_samples,)
            A fresh array is yielded each iteration (no aliasing issues
            when two generators are zipped as in the training loop).
        """
        scores = np.zeros(X.shape[0], dtype=np.float64)
        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            scores = scores + alpha * (2 * stump.predict(X) - 1)
            yield scores.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {0, 1}."""
        return (self.decision_function(X) > 0).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return soft class probabilities, shape (n_samples, 2).
        Uses a sigmoid on the raw AdaBoost score.
        """
        scores   = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-2.0 * scores))
        return np.column_stack([1.0 - prob_pos, prob_pos])

    # ------------------------------------------------------------------
    # sklearn property aliases
    # ------------------------------------------------------------------

    @property
    def n_classes_(self) -> int:
        return int(len(self.classes_))

    @property
    def n_estimators_(self) -> int:
        """Number of fitted weak learners (may be less than n_estimators)."""
        return len(self.estimators_)

    def __repr__(self) -> str:
        return (
            f"AdaBoostStumpClassifier("
            f"n_estimators={self.n_estimators}, "
            f"fitted={len(self.estimators_)}, "
            f"learning_rate={self.learning_rate}, "
            f"n_jobs={self.n_jobs})"
        )