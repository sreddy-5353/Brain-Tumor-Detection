"""
Brain Tumor Classifier — Prediction & Testing Script
Usage:
    python predict.py --image path/to/scan.jpg
    python predict.py --folder path/to/test_images/
    python predict.py --evaluate          # runs full test-set evaluation
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 160       # ✅ FIXED — must match train_model.py and app.py
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH  = "models/best_model.h5"
DATASET_DIR = "dataset"

CLASS_INFO = {
    "glioma":     {"icon": "🔴", "severity": "HIGH RISK"},
    "meningioma": {"icon": "🟡", "severity": "MODERATE"},
    "notumor":    {"icon": "🟢", "severity": "NORMAL"},
    "pituitary":  {"icon": "🟣", "severity": "MODERATE"},
}

# Low-confidence threshold
CONFIDENCE_THRESHOLD = 0.60


# ─── Helpers ───────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    print(f"✅ Loading model from {MODEL_PATH} ...")
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img_path: str) -> np.ndarray:
    """Load, resize to IMG_SIZE×IMG_SIZE, normalise to [0,1], add batch dim."""
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def predict_single(model, img_path: str) -> dict:
    x    = preprocess(img_path)
    prob = model.predict(x, verbose=0)[0]
    idx  = int(np.argmax(prob))
    return {
        "class":      CLASSES[idx],
        "confidence": float(prob[idx]),
        "all_probs":  {c: float(p) for c, p in zip(CLASSES, prob)},
    }


def print_result(img_path: str, result: dict):
    cls  = result["class"]
    conf = result["confidence"]
    info = CLASS_INFO[cls]

    print(f"\n{'─'*55}")
    print(f"  File      : {os.path.basename(img_path)}")
    print(f"  Prediction: {info['icon']}  {cls.upper():<14}  [{info['severity']}]")
    print(f"  Confidence: {conf*100:.1f}%")

    if conf < CONFIDENCE_THRESHOLD:
        print(f"  ⚠️  Low confidence — consider manual review")

    print(f"\n  Probability breakdown:")
    for c in sorted(result["all_probs"], key=result["all_probs"].get, reverse=True):
        bar_len = int(result["all_probs"][c] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        marker  = " ◄" if c == cls else ""
        print(f"    {c:12s} {bar} {result['all_probs'][c]*100:5.1f}%{marker}")
    print(f"{'─'*55}")


# ─── Full Test-Set Evaluation ───────────────────────────────────────────────────
def evaluate_test_set(model):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    test_dir = os.path.join(DATASET_DIR, "test")
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return

    print("\n🔄 Running evaluation on test set ...")

    val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = val_gen.classes[:len(y_pred)]

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=13)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    out = "models/confusion_matrix_eval.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Confusion matrix saved → {out}")

    # ── notumor class deep-dive ────────────────────────────────────────────────
    notumor_idx = CLASSES.index("notumor")
    tp = cm[notumor_idx, notumor_idx]
    fn = cm[notumor_idx].sum() - tp
    fp = cm[:, notumor_idx].sum() - tp
    print(f"\n🟢 'No Tumor' class breakdown:")
    print(f"   True Positives  (correctly predicted no tumor): {tp}")
    print(f"   False Negatives (missed — predicted as tumor) : {fn}")
    print(f"   False Positives (tumor predicted as no tumor) : {fp}")
    if fn > 0:
        print(f"   ⚠️  {fn} real no-tumor scan(s) wrongly flagged as tumor.")


# ─── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classifier — Prediction")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",    type=str,        help="Path to a single MRI image")
    group.add_argument("--folder",   type=str,        help="Path to a folder of MRI images")
    group.add_argument("--evaluate", action="store_true", help="Evaluate on full test set")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ File not found: {args.image}")
            return
        result = predict_single(model, args.image)
        print_result(args.image, result)

    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        files = [
            os.path.join(args.folder, f)
            for f in sorted(os.listdir(args.folder))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not files:
            print(f"❌ No images found in {args.folder}")
            return
        print(f"\n🔬 Classifying {len(files)} image(s) in {args.folder} ...\n")
        counts = {c: 0 for c in CLASSES}
        for path in files:
            result = predict_single(model, path)
            print_result(path, result)
            counts[result["class"]] += 1

        print(f"\n📊 Summary:")
        for c, n in counts.items():
            print(f"   {CLASS_INFO[c]['icon']}  {c:12s}: {n}")

    elif args.evaluate:
        evaluate_test_set(model)


if __name__ == "__main__":
    main()

    