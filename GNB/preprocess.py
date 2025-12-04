from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import re

try:
    # Local import for encoder; used only by HeartPreprocessor when handling raw text rows with '?'
    from . import heart_raw2cod  # type: ignore
except Exception:
    try:
        import heart_raw2cod  # fallback when running as script
    except Exception:
        heart_raw2cod = None  # will guard at call-site

try:
    # Local import for cancer encoder when available alongside this module
    from . import cancer_raw2cod  # type: ignore
except Exception:
    try:
        import cancer_raw2cod  # fallback when running as script
    except Exception:
        cancer_raw2cod = None


def expand_heart_features(vals: List[float]) -> np.ndarray:
    age, sex, cp, rest_bp, chol, fbs, ecg, max_hr, angina, oldpeak, slope, vessels, thal = vals
    features: List[float] = []

    # Mirrors the existing project logic for expansion (35 features)
    features.append(age); features.append(0)
    features.append(sex); features.append(0)
    features.extend([1 if int(cp) == i else 0 for i in range(4)]); features.append(0)
    features.append(rest_bp); features.append(0)
    features.append(chol); features.append(0)
    features.append(fbs); features.append(0)
    features.extend([1 if int(ecg) == i else 0 for i in range(3)]); features.append(0)
    features.append(max_hr); features.append(0)
    features.append(angina); features.append(0)
    features.append(oldpeak); features.append(0)
    features.extend([1 if int(slope) == i else 0 for i in range(3)]); features.append(0)
    features.append(vessels); features.append(0)
    thal_map = {3: 0, 6: 1, 7: 2}
    thal_idx = thal_map.get(int(thal), -1)
    features.extend([1 if thal_idx == i else 0 for i in range(3)]); features.append(0)

    if len(features) != 35:
        raise ValueError(f"Expanded heart features length mismatch: expected 35, got {len(features)}")

    return np.array(features, dtype=float)


class BasePreprocessor:
    task: str

    def expected_counts(self) -> set[int]:
        raise NotImplementedError

    def column_hint(self) -> str:
        raise NotImplementedError

    # Interactive single-input preprocessing
    def preprocess_single(self, vals: List[float]) -> np.ndarray:
        return np.array(vals, dtype=float)

    # Batch matrix preprocessing: returns (X, y_true, messages)
    def preprocess_batch_matrix(self, arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        raise NotImplementedError

    # Utilities shared across preprocessors
    @staticmethod
    def _is_binary(col: np.ndarray) -> bool:
        u = np.unique(col)
        return np.all(np.isin(u, [0, 1])) and 1 <= u.size <= 2

    @staticmethod
    def _is_id_like(col: np.ndarray, row_count: int, scaled_context: bool) -> bool:
        # Integer-ish, unique per row, and large magnitude if scaled context
        if col.dtype.kind not in {"i", "u", "f"}:
            return False
        u = np.unique(col)
        if len(u) != row_count:
            return False
        if scaled_context:
            return np.nanmax(col) > 10
        return True


class DiabetesPreprocessor(BasePreprocessor):
    task = "diabetes"

    def expected_counts(self) -> set[int]:
        return {8}

    def column_hint(self) -> str:
        return (
            "[INFO] Batch file should contain 8 feature columns (optionally +1 binary label, or +2 one-hot label at the end). Avoid ID columns."
        )

    def preprocess_single(self, vals: List[float]) -> np.ndarray:
        # Min-max scale to 0..1 per dataset notes
        mins = np.array([0, 0, 0, 0, 0, 0, 0.078, 21], dtype=float)
        maxs = np.array([17, 199, 122, 99, 846, 67.1, 2.42, 81], dtype=float)
        v = np.array(vals, dtype=float)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        x = (v - mins) / denom
        return np.clip(x, 0.0, 1.0)

    def preprocess_batch_matrix(self, arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        msgs: List[str] = []
        X = arr.copy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_true = None
        expected_set = self.expected_counts()
        allowed = expected_set

        rows, cols = X.shape

        # Two-column one-hot label at end [1 0]=Non-Diabetic(0), [0 1]=Diabetic(1)
        if cols >= 10:
            tail = X[:, -2:]
            if self._is_binary(tail[:, 0]) and self._is_binary(tail[:, 1]) and np.all(np.sum(tail, axis=1) == 1):
                y_true = tail[:, 1].astype(int)
                X = X[:, :-2]
                cols = X.shape[1]
                msgs.append("[INFO] Detected 2-column one-hot label at end; separated y_true and removed label columns.")

        # Try label at end or start if that yields a valid count
        if cols >= min(allowed) + 1 and self._is_binary(X[:, -1]) and (cols - 1) in allowed:
            y_true = X[:, -1].astype(int)
            X = X[:, :-1]
            cols = X.shape[1]
        if cols >= min(allowed) + 1 and self._is_binary(X[:, 0]) and (cols - 1) in allowed and y_true is None:
            y_true = X[:, 0].astype(int)
            X = X[:, 1:]
            cols = X.shape[1]

        # Drop ID-like leading column if exactly one extra
        if cols == min(allowed) + 1:
            if self._is_id_like(X[:, 0], rows, scaled_context=True):
                X = X[:, 1:]
                cols = X.shape[1]
                msgs.append("[NOTICE] Dropped first column (assumed ID).")

        # Fallback: drop first/last if still one extra
        if cols == min(allowed) + 1:
            cand = X[:, 1:]
            if cand.shape[1] in allowed:
                X = cand; cols = X.shape[1]
                msgs.append("[WARN] Dropped first extra column to match expected feature count.")
            else:
                cand = X[:, :-1]
                if cand.shape[1] in allowed:
                    X = cand; cols = X.shape[1]
                    msgs.append("[WARN] Dropped last extra column to match expected feature count.")

        if cols not in allowed:
            raise ValueError(
                f"Feature column mismatch. Diabetes expects 8 features; got {cols}. "
                "Ensure file has 8 feature columns (optionally +1 binary label)."
            )

        # Auto-detect scaling
        xmin = float(np.nanmin(X)); xmax = float(np.nanmax(X))
        is_scaled = (xmin >= -1e-6) and (xmax <= 1.05)
        if is_scaled:
            X = np.clip(X, 0.0, 1.0)
            msgs.append("[INFO] Detected inputs already in [0,1] range; no additional scaling applied.")
        else:
            mins = np.array([0, 0, 0, 0, 0, 0, 0.078, 21], dtype=float)
            maxs = np.array([17, 199, 122, 99, 846, 67.1, 2.42, 81], dtype=float)
            denom = (maxs - mins)
            denom[denom == 0] = 1.0
            X = (X - mins) / denom
            X = np.clip(X, 0.0, 1.0)
            msgs.append("[INFO] Detected raw diabetes inputs; applied min-max scaling to [0,1] per dataset doc.")

        return X, y_true, msgs


class CancerPreprocessor(BasePreprocessor):
    task = "cancer"

    def expected_counts(self) -> set[int]:
        return {9}

    def column_hint(self) -> str:
        return (
            "[INFO] Batch file should contain 9 feature columns (optionally +1 label column at the end = 10 total). Remove any ID column."
        )

    def preprocess_single(self, vals: List[float]) -> np.ndarray:
        v = np.array(vals, dtype=float)
        # Scale 1..10 -> 0..1 using (x-1)/9
        return np.clip((v - 1.0) / 9.0, 0.0, 1.0)

    def preprocess_batch_matrix(self, arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        msgs: List[str] = []
        X = arr.copy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_true = None
        expected_set = self.expected_counts()
        allowed = expected_set

        rows, cols = X.shape

        # Label at end/start only if dropping yields valid count
        if cols >= 10 and self._is_binary(X[:, -1]) and (cols - 1) in allowed:
            y_true = X[:, -1].astype(int); X = X[:, :-1]; cols = X.shape[1]
        if cols >= 10 and self._is_binary(X[:, 0]) and (cols - 1) in allowed and y_true is None:
            y_true = X[:, 0].astype(int); X = X[:, 1:]; cols = X.shape[1]

        # Drop ID-like leading column if exactly one extra
        if cols == 10 and self._is_id_like(X[:, 0], rows, scaled_context=True):
            X = X[:, 1:]; cols = X.shape[1]
            msgs.append("[NOTICE] Dropped first column (assumed ID).")

        # Fallback: drop first/last if still one extra
        if cols == 10:
            cand = X[:, 1:]
            if cand.shape[1] in allowed:
                X = cand; cols = X.shape[1]
                msgs.append("[WARN] Dropped first extra column to match expected feature count.")
            else:
                cand = X[:, :-1]
                if cand.shape[1] in allowed:
                    X = cand; cols = X.shape[1]
                    msgs.append("[WARN] Dropped last extra column to match expected feature count.")

        if cols not in allowed:
            raise ValueError(
                f"Feature column mismatch. Cancer expects 9 features; got {cols}. "
                "Ensure file has 9 feature columns (optionally +1 binary label)."
            )

        # Auto-detect scaling
        xmin = float(np.nanmin(X)); xmax = float(np.nanmax(X))
        is_scaled = (xmin >= -1e-6) and (xmax <= 1.05)
        if is_scaled:
            X = np.clip(X, 0.0, 1.0)
            msgs.append("[INFO] Detected inputs already in [0,1] range; no additional scaling applied.")
        else:
            X = np.clip((X - 1.0) / 9.0, 0.0, 1.0)
            msgs.append("[INFO] Detected raw cancer inputs; scaled from 1..10 to 0..1.")

        return X, y_true, msgs

    # New: handle raw cancer text containing commas and '?' using cancer_raw2cod encoder
    def preprocess_batch_text(self, text: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if cancer_raw2cod is None:
            raise RuntimeError("cancer_raw2cod module not available; cannot process raw cancer text with missing values.")
        feats: List[List[float]] = []
        labels: List[int] = []
        any_lines = 0
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            any_lines += 1
            fields = [t.strip() for t in s.split(",")]
            enc = cancer_raw2cod.encode_row(fields)
            toks = enc.split()
            nums = [float(t) for t in toks]
            if len(nums) < 11:
                raise ValueError(f"Encoded row length {len(nums)} < 11; expected 9 features + 2 labels.")
            feats.append(nums[:9])
            # labels two cols: [1,0] => Benign(0), [0,1] => Malignant(1)
            y = 0 if int(nums[9]) == 1 else 1
            labels.append(y)

        if any_lines == 0:
            raise ValueError("No data lines found in provided text.")

        X = np.array(feats, dtype=float)
        y_true = np.array(labels, dtype=int)
        msgs = [
            "[INFO] Detected raw Cancer rows with missing values; applied cancer_raw2cod encoding and extracted binary labels.",
        ]
        return X, y_true, msgs


class HeartPreprocessor(BasePreprocessor):
    task = "heart"

    def expected_counts(self) -> set[int]:
        # Accept 13 raw or 35 expanded
        return {13, 35}

    def column_hint(self) -> str:
        return (
            "[INFO] Heart batch file may contain either: 13 raw features (auto-expanded) or 35 expanded features. "
            "Labels supported: single binary column (0/1) or two-column one-hot at the end [1 0]=Positive, [0 1]=Negative. Avoid ID columns."
        )

    def preprocess_single(self, vals: List[float]) -> np.ndarray:
        return expand_heart_features(vals)

    def preprocess_batch_matrix(self, arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        msgs: List[str] = []
        X = arr.copy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_true = None

        rows, cols = X.shape
        allowed = self.expected_counts()

        # One-hot two-column label at end
        if cols >= 15:
            tail = X[:, -2:]
            if self._is_binary(tail[:, 0]) and self._is_binary(tail[:, 1]) and np.all(np.sum(tail, axis=1) == 1):
                y_true = (tail[:, 0] == 1).astype(int)  # [1,0] => Positive(1); [0,1] => Negative(0)
                X = X[:, :-2]
                cols = X.shape[1]
                msgs.append("[INFO] Detected 2-column one-hot label at end; separated y_true and removed label columns.")

        # Single binary label at end/start if dropping yields 13 or 35
        if cols >= 14 and self._is_binary(X[:, -1]) and (cols - 1) in allowed and y_true is None:
            y_true = X[:, -1].astype(int); X = X[:, :-1]; cols = X.shape[1]
        if cols >= 14 and self._is_binary(X[:, 0]) and (cols - 1) in allowed and y_true is None:
            y_true = X[:, 0].astype(int); X = X[:, 1:]; cols = X.shape[1]

        # ID-like leading column: only drop if leads to allowed count
        if (cols - 1) in allowed and self._is_id_like(X[:, 0], rows, scaled_context=False):
            X = X[:, 1:]; cols = X.shape[1]
            msgs.append("[NOTICE] Dropped first column (assumed ID) to match expected feature count.")

        # If still off by one, try dropping first/last to match allowed
        if (cols - 1) in allowed:
            cand = X[:, 1:]
            if cand.shape[1] in allowed:
                X = cand; cols = X.shape[1]
                msgs.append("[WARN] Dropped first extra column to match expected feature count.")
            else:
                cand = X[:, :-1]
                if cand.shape[1] in allowed:
                    X = cand; cols = X.shape[1]
                    msgs.append("[WARN] Dropped last extra column to match expected feature count.")

        if cols not in allowed:
            raise ValueError(
                f"Feature column mismatch after alignment attempts. Heart accepts 13 raw or 35 expanded features; got {cols}."
            )

        # Expand if 13 raw, else use as-is
        if cols == 13:
            X = np.array([expand_heart_features(row.tolist()) for row in X], dtype=float)
        # No additional scaling for heart here; assumes values prepared as in project

        return X, y_true, msgs

    # New path: handle raw heart files (comma-separated) that may include '?' for missing values.
    # Uses the Python port of raw2cod to produce 35 inputs + 2 outputs, then extracts X and y.
    def preprocess_batch_text(self, text: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if heart_raw2cod is None:
            raise RuntimeError("heart_raw2cod module not available; cannot process raw heart text with missing values.")
        feats: List[List[float]] = []
        labels: List[int] = []
        any_lines = 0
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            any_lines += 1
            # Normalize to comma-separated tokens
            if "," in s:
                fields = [t.strip() for t in s.split(",")]
            else:
                # Split on whitespace and keep as string tokens
                fields = re.split(r"\s+", s)
            # Encode with classifier output (2 columns)
            encoded = heart_raw2cod.encode_row(fields, classifier=True)
            nums = [float(tok) for tok in encoded.split()]
            if len(nums) < 37:
                raise ValueError(f"Encoded row length {len(nums)} < 37; expected 35 features + 2 labels.")
            feats.append(nums[:35])
            # Two-column one-hot: [1,0] => Positive(1); [0,1] => Negative(0)
            y = 1 if int(nums[35]) == 1 else 0
            labels.append(y)

        if any_lines == 0:
            raise ValueError("No data lines found in provided text.")

        X = np.array(feats, dtype=float)
        y_true = np.array(labels, dtype=int)
        msgs = [
            "[INFO] Detected raw Heart rows with missing values; applied heart_raw2cod encoding to 35 features and extracted binary labels.",
        ]
        return X, y_true, msgs


def get_preprocessor(task: str) -> BasePreprocessor:
    if task == "diabetes":
        return DiabetesPreprocessor()
    if task == "cancer":
        return CancerPreprocessor()
    if task == "heart":
        return HeartPreprocessor()
    raise ValueError(f"Unknown task: {task}")
