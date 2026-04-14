"""
Brain Tumor Classifier — Optimized Training
Fixes:
  - IMG_SIZE consistent at 160 across all files
  - Mixed-precision for faster GPU training
  - Unfreezes last 30 layers for fine-tuning after initial training
  - steps_per_epoch uses full data (removed the //2 hack)
  - eval_results.json includes all fields app.py expects
  - Reproducible seeds
"""

import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Enable mixed precision (2–3× speed-up on compatible GPUs / Apple Silicon) ──
try:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed-precision (float16) enabled")
except Exception:
    print("ℹ️  Mixed-precision not available — using float32")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
IMG_SIZE   = 160        # shared with app.py and predict.py
BATCH_SIZE = 32         # larger batch → faster epoch
EPOCHS     = 15         # more epochs; EarlyStopping will cut short if needed

DATASET_DIR = "dataset"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.h5")
EVAL_PATH   = os.path.join(MODEL_DIR, "eval_results.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── DATA GENERATORS ────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    zoom_range=0.10,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

CLASS_NAMES = list(train_gen.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)
print(f"\n📂 Classes : {CLASS_NAMES}")
print(f"📊 Train   : {train_gen.samples} samples")
print(f"📊 Test    : {test_gen.samples} samples\n")

# ── MODEL (MobileNetV2 — fast & accurate) ─────────────────────────────────────
base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False      # Phase 1: feature extraction only

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
# Cast back to float32 for softmax stability when using mixed precision
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── CALLBACKS ──────────────────────────────────────────────────────────────────
callbacks_phase1 = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4,
        restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, min_lr=1e-7, verbose=1,
    ),
]

# ── PHASE 1: TRAIN HEAD ────────────────────────────────────────────────────────
print("🚀 Phase 1 — Training classification head …\n")

history1 = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=test_gen.samples // BATCH_SIZE,
    callbacks=callbacks_phase1,
)

# ── PHASE 2: FINE-TUNE LAST 30 LAYERS ──────────────────────────────────────────
print("\n🔓 Phase 2 — Fine-tuning last 30 layers …\n")

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),   # much lower LR
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5,
        restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3,
        patience=2, min_lr=1e-8, verbose=1,
    ),
]

history2 = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=test_gen.samples // BATCH_SIZE,
    callbacks=callbacks_phase2,
)

print("\n✅ Training complete.\n")

# ── Merge histories ─────────────────────────────────────────────────────────────
def merge(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, num_classes):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specs = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)

    return {
        "Accuracy":    round(float(acc),  4),
        "Precision":   round(float(prec), 4),
        "Recall":      round(float(rec),  4),
        "F1 Score":    round(float(f1),   4),
        "Specificity": round(float(np.mean(specs)), 4),
    }


# Re-create generators without augmentation for evaluation
eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("📊 Evaluating on training set …")
train_eval_gen = eval_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

print("📊 Evaluating on testing set …")
test_eval_gen = eval_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

y_pred_train = np.argmax(model.predict(train_eval_gen, verbose=1), axis=1)
y_true_train = train_eval_gen.classes

y_pred_test = np.argmax(model.predict(test_eval_gen, verbose=1), axis=1)
y_true_test = test_eval_gen.classes

train_metrics = compute_metrics(y_true_train, y_pred_train, NUM_CLASSES)
test_metrics  = compute_metrics(y_true_test,  y_pred_test,  NUM_CLASSES)

train_cm = confusion_matrix(y_true_train, y_pred_train, labels=list(range(NUM_CLASSES))).tolist()
test_cm  = confusion_matrix(y_true_test,  y_pred_test,  labels=list(range(NUM_CLASSES))).tolist()

# ── SAVE RESULTS ───────────────────────────────────────────────────────────────
eval_results = {
    "class_names":      CLASS_NAMES,
    "train_metrics":    train_metrics,
    "test_metrics":     test_metrics,
    "train_cm":         train_cm,
    "test_cm":          test_cm,
    "history_loss":     [float(v) for v in merge(history1, history2, "loss")],
    "history_val_loss": [float(v) for v in merge(history1, history2, "val_loss")],
    "history_acc":      [float(v) for v in merge(history1, history2, "accuracy")],
    "history_val_acc":  [float(v) for v in merge(history1, history2, "val_accuracy")],
}

with open(EVAL_PATH, "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\n✅ Evaluation results saved → {EVAL_PATH}")
print(f"🏆 Train Accuracy : {train_metrics['Accuracy']}")
print(f"🏆 Test  Accuracy : {test_metrics['Accuracy']}")
print(f"🏆 Test  F1 Score : {test_metrics['F1 Score']}")


