from __future__ import annotations
import argparse
import io
import csv
import os
from typing import Iterable, List

import numpy as np
from joblib import load

from train_gnb import MyGaussianNB
try:
    # Prefer local module import (same folder)
    import heart_raw2cod  # type: ignore
except Exception:
    heart_raw2cod = None

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required. Install with 'pip install scikit-learn' for metrics."
    ) from e

# Directories and configs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = {
    "diabetes": os.path.join(BASE_DIR, "results", "diabetes"),
    "cancer": os.path.join(BASE_DIR, "results", "cancer"),
    "heart": os.path.join(BASE_DIR, "results", "heart"),
}

MODELS = {
    "diabetes": os.path.join(BASE_DIR, "models", "gnb_diabetes.joblib"),
    "cancer": os.path.join(BASE_DIR, "models", "gnb_cancer.joblib"),
    "heart": os.path.join(BASE_DIR, "models", "gnb_heart.joblib"),
}

for path in RESULTS_DIR.values():
    os.makedirs(path, exist_ok=True)

from preprocess import get_preprocessor

# Feature descriptions per task
FEATURES = {
    "diabetes": [
        "0. Number of times pregnant (0–17; scaled 0–1)",
        "1. Plasma glucose concentration (2hr OGTT) (0–199; scaled 0–1)",
        "2. Diastolic blood pressure (mm Hg) (0–122; scaled 0–1)",
        "3. Triceps skin fold thickness (mm) (0–99; scaled 0–1)",
        "4. 2-Hour serum insulin (mu U/ml) (0–846; scaled 0–1)",
        "5. Body mass index (kg/m^2) (0–67.1; scaled 0–1)",
        "6. Diabetes pedigree function (0.078–2.42; scaled 0–1)",
        "7. Age (years) (21–81; scaled 0–1)",
    ],
    "cancer": [
        "0. Clump thickness (1–10; scaled 0–1)",
        "1. Uniformity of cell size (1–10; scaled 0–1)",
        "2. Uniformity of cell shape (1–10; scaled 0–1)",
        "3. Marginal adhesion (1–10; scaled 0–1)",
        "4. Single epithelial cell size (1–10; scaled 0–1)",
        "5. Bare nuclei (1–10; scaled 0–1)",
        "6. Bland chromatin (1–10; scaled 0–1)",
        "7. Normal nucleoli (1–10; scaled 0–1)",
        "8. Mitoses (1–10; scaled 0–1)",
    ],
    "heart": [
        "0. Age (29–77; scaled 0–1)",
        "1. Sex (0=female,1=male)",
        "2. Chest pain type (0–3)",
        "3. Resting blood pressure (94–200; scaled 0–1)",
        "4. Serum cholesterol (126–564; scaled 0–1)",
        "5. Fasting blood sugar (>120mg/dl, 0/1)",
        "6. Resting ECG results (0–2)",
        "7. Max heart rate achieved (71–202; scaled 0–1)",
        "8. Exercise induced angina (0/1)",
        "9. Oldpeak = ST depression (0–6.2; scaled 0–1)",
        "10. Slope of peak exercise ST (0–2)",
        "11. Number of major vessels (0–3)",
        "12. Thalassemia (3=normal; 6=fixed defect; 7=reversible defect)",
    ],
}

# Label mappings for different tasks
LABELS = {
    "diabetes": {0: "Diabetic", 1: "Non-Diabetic"},
    "heart": {0: "Negative", 1: "Positive"},
    "cancer": {0: "Benign", 1: "Malignant"},
}

def _label_str(task: str, v: int | float) -> str:
    return LABELS[task][int(v)]

def _load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first by running gnb.py"
        )
    return load(model_path)

def _predict_single(task, model_path):
    model = _load_model(model_path)
    pre = get_preprocessor(task)
    
    if task == "cancer":
        print("Enter feature values for Cancer dataset:")
        prompts = [
            "Clump thickness (1–10): ",
            "Uniformity of cell size (1–10): ",
            "Uniformity of cell shape (1–10): ",
            "Marginal adhesion (1–10): ",
            "Single epithelial cell size (1–10): ",
            "Bare nuclei (1–10): ",
            "Bland chromatin (1–10): ",
            "Normal nucleoli (1–10): ",
            "Mitoses (1–10): ",
        ]
        vals = [float(input(p)) for p in prompts]

        scaled_vals = [v / 10 for v in vals]
        x = np.array(scaled_vals).reshape(1, -1)

        y_pred = model.predict(x)[0]
        probs = model.predict_proba(x)[0]

        # Round scaled values to 1 decimal place
        scaled = [round(v, 1) for v in x.tolist()[0]]

        print(f"\nRaw input: {vals}")
        print(f"Scaled input: {scaled}")
        print(f"Predicted: {_label_str(task, y_pred)}")
        print(f"Class probabilities: [{probs[0]:.4f}, {probs[1]:.4f}]")
        return

    elif task == "diabetes":
        print("Enter feature values for diabetes:")
        prompts = [
            "Number of times pregnant: ",
            "Plasma glucose concentration (0–199): ",
            "Diastolic blood pressure (0–122): ",
            "Triceps skin fold thickness (0–99): ",
            "2-Hour serum insulin (0–846): ",
            "Body mass index (0–67.1): ",
            "Diabetes pedigree function (0.078–2.42): ",
            "Age: ",
        ]
        vals = [float(input(p)) for p in prompts]
        x = pre.preprocess_single(vals).reshape(1, -1)

        y_pred = model.predict(x)[0]
        probs = model.predict_proba(x)[0]
        
        # Round scaled values to 1 decimal place
        scaled = [round(v, 1) for v in x.tolist()[0]]

        print(f"\nRaw input: {vals}")
        print(f"Scaled input: {x.tolist()[0]}")
        print(f"Predicted: {_label_str(task, y_pred)}")
        print(f"Class probabilities: [{probs[0]:.4f}, {probs[1]:.4f}]")
        return

    elif task == "heart":
        if heart_raw2cod is None:
            raise RuntimeError(
                "heart_raw2cod module not available; cannot perform PROBEN1-faithful encoding in interactive mode."
            )

        print("Enter feature values for Heart dataset (use '?' for missing when applicable):")
        prompts = [
            "Age (years, 28–77): ",
            "Sex (0=female,1=male): ",
            "Chest pain type (1..4): ",
            "Resting blood pressure (80–200): ",
            "Serum cholesterol (85–605): ",
            "Fasting blood sugar (0/1, '?' if unknown): ",
            "Resting ECG (0..2, '?' if unknown): ",
            "Max heart rate achieved (60–210): ",
            "Exercise induced angina (0/1, '?' if unknown): ",
            "Oldpeak ST depression (-2.6..6.2, '?' if unknown): ",
            "Slope of peak exercise ST (1..3, '?' if unknown): ",
            "Number of major vessels (0..3, '?' if unknown): ",
            "Thalassemia (3|6|7, '?' if unknown): ",
        ]

        # Collect as strings to preserve '?' semantics
        raw_inputs: List[str] = []
        for p in prompts:
            s = input(p).strip()
            raw_inputs.append(s)

        # Append a dummy label (0) to match encoder's expected 14 fields
        fields = raw_inputs + ["0"]
        encoded_line = heart_raw2cod.encode_row(fields, classifier=True)
        nums = [float(tok) for tok in encoded_line.split()]
        features35 = nums[:35]

        # --- Predictions ---
        x = np.array(features35, dtype=float).reshape(1, -1)
        y_pred = model.predict(x)[0]
        probs = model.predict_proba(x)[0]

        print("\n--- RESULTS ---")
        print(f"Raw input (13 features): {raw_inputs}")
        print(f"Encoded/expanded (35 features, PROBEN1 raw2cod): {[round(float(v), 3) for v in x.flatten()]}")
        print(f"Predicted: {_label_str(task, y_pred)}")
        print(f"Class probabilities: [{probs[0]:.4f}, {probs[1]:.4f}]")
        return

def _predict_batch(task: str, model_path: str, data_path: str, out_path: str | None = None) -> None:
    clf = _load_model(model_path)
    pre = get_preprocessor(task)
    def _load_matrix(path: str) -> np.ndarray:
        # First, try common fast paths
        try:
            return np.loadtxt(path, delimiter=",", dtype=float)
        except Exception:
            pass
        try:
            return np.loadtxt(path, dtype=float)
        except Exception:
            pass
        with open(path, "rb") as fb:
            raw = fb.read()
        text: str
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            text = raw.decode("utf-16")
        elif raw.startswith(b"\xef\xbb\xbf"):
            text = raw.decode("utf-8-sig")
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp1252")
        # Normalize delimiters: collapse any commas/tabs to spaces
        text = text.replace(",", " ")
        text = text.replace("\x00", "")
        sio = io.StringIO(text)
        return np.loadtxt(sio, dtype=float)

    # If heart/cancer and file appears to contain raw text (commas or '?' missing markers), route through text preprocessing
    use_text_path = False
    if task in ("heart", "cancer"):
        with open(data_path, "rb") as fb:
            raw = fb.read(4096)
        dec: str
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            dec = raw.decode("utf-16", errors="ignore")
        elif raw.startswith(b"\xef\xbb\xbf"):
            dec = raw.decode("utf-8-sig", errors="ignore")
        else:
            try:
                dec = raw.decode("utf-8")
            except UnicodeDecodeError:
                dec = raw.decode("cp1252", errors="ignore")
        probe = dec.replace("\x00", "")
        if "," in probe or "?" in probe:
            use_text_path = True

    if task in ("heart", "cancer") and use_text_path:
        # Read full text with robust decoding and delegate to HeartPreprocessor
        with open(data_path, "rb") as fb:
            raw_full = fb.read()
        if raw_full.startswith(b"\xff\xfe") or raw_full.startswith(b"\xfe\xff"):
            text_full = raw_full.decode("utf-16")
        elif raw_full.startswith(b"\xef\xbb\xbf"):
            text_full = raw_full.decode("utf-8-sig")
        else:
            try:
                text_full = raw_full.decode("utf-8")
            except UnicodeDecodeError:
                text_full = raw_full.decode("cp1252")
        X, y_true, msgs = pre.preprocess_batch_text(text_full)
        for m in msgs:
            print(m)
    else:
        arr = _load_matrix(data_path)
        # If single row, reshape to 2D for preprocessing
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Delegate all alignment/scaling/expansion to preprocessor
        X, y_true, msgs = pre.preprocess_batch_matrix(arr)
        for m in msgs:
            print(m)

    # Predict
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)

    # Save predictions
    if not out_path:
        base = os.path.splitext(os.path.basename(data_path))[0]
        out_path = os.path.join(RESULTS_DIR[task], f"{base}_predictions.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["index", "pred", "prob0", "prob1"]
        if y_true is not None:
            header.append("y_true")
        writer.writerow(header)

        for i, (yp, pr) in enumerate(zip(y_pred, proba)):
            row = [i, _label_str(task, yp), f"{pr[0]:.6f}", f"{pr[1]:.6f}"]
            if y_true is not None:
                row.append(_label_str(task, y_true[i]))
            writer.writerow(row)

    print(f"Saved predictions to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with saved GaussianNB")
    parser.add_argument("--interactive", action="store_true", help="Prompt for single input")
    parser.add_argument("--batch", help="Path to .dat/.csv file for batch predictions")
    args = parser.parse_args()

    # Dataset choice
    print("Select dataset: [0] Diabetes, [1] Cancer, [2] Heart")
    choice = input("").strip()
    if choice == "0":
        task = "diabetes"
    elif choice == "1":
        task = "cancer"
    elif choice == "2":
        task = "heart"
    else:
        raise SystemExit("Invalid choice. Please enter 0, 1, or 2.")

    model_path = MODELS[task]

    def _print_batch_column_hint(selected_task: str):  # local helper
        pre = get_preprocessor(selected_task)
        print(pre.column_hint())

    if args.interactive:
        _predict_single(task, model_path)
    elif args.batch:
        _print_batch_column_hint(task)
        _predict_batch(task, model_path, args.batch)
    else:
        mode = input("Choose mode: [0] Single Prediction or [1] File Prediction: ").strip()
        if mode == "0":
            _predict_single(task, model_path)
        elif mode == "1":
            _print_batch_column_hint(task)
            data_path = input("Enter data file path: ").strip()
            _predict_batch(task, model_path, data_path)
        else:
            print("Invalid mode. Please enter 0 or 1.")
