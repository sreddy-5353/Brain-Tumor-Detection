"""
🧠 Brain Tumor Detection — Streamlit App
Fixes applied:
  - Single preprocess_image() function with IMG_SIZE=160 (was duplicated + mismatched)
  - retrain_on_feedback() uses IMG_SIZE=160 (was hardcoded 224)
  - Model load also checks for keras format (.keras) in addition to .h5
  - Plotly chart keys are always unique (no duplicate-key crashes)
  - Minor UX polish
"""

import streamlit as st
import numpy as np
import json
import os
import time
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CONFIG (must match train_model.py) ────────────────────────────────
IMG_SIZE = 160   # ✅ single source of truth


# ─── Image helpers ─────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize, normalise, add batch dim — uses global IMG_SIZE."""
    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def img_to_b64(img: Image.Image, size=(56, 56)) -> str:
    buf = BytesIO()
    img.copy().resize(size).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Custom CSS (clean white theme) ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg:          #ffffff;
        --bg-soft:     #f8fafc;
        --bg-card:     #ffffff;
        --border:      #e2e8f0;
        --border-md:   #cbd5e1;
        --text:        #0f172a;
        --text-2:      #334155;
        --text-muted:  #94a3b8;
        --blue:        #2563eb;
        --blue-lt:     #eff6ff;
        --red:         #dc2626;
        --red-lt:      #fef2f2;
        --amber:       #d97706;
        --amber-lt:    #fffbeb;
        --green:       #16a34a;
        --green-lt:    #f0fdf4;
        --purple:      #7c3aed;
        --purple-lt:   #f5f3ff;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--bg) !important;
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: var(--bg-soft) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    .topbar {
        background: var(--blue);
        color: #fff;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .topbar h1 { font-size:1.45rem; font-weight:700; margin:0 0 0.2rem; color:#fff !important; }
    .topbar p  { margin:0; font-size:0.8rem; color:rgba(255,255,255,0.75) !important; }

    .card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:1.2rem 1.4rem; margin:0.4rem 0; }
    .card-blue   { border-left:4px solid var(--blue); }
    .card-red    { border-left:4px solid var(--red); }
    .card-green  { border-left:4px solid var(--green); }
    .card-amber  { border-left:4px solid var(--amber); }
    .card-purple { border-left:4px solid var(--purple); }

    .badge { display:inline-block; padding:0.35rem 1.1rem; border-radius:50px; font-family:'JetBrains Mono',monospace; font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
    .badge-glioma     { background:var(--red-lt);    color:var(--red);    border:1px solid #fca5a5; }
    .badge-meningioma { background:var(--amber-lt);  color:var(--amber);  border:1px solid #fcd34d; }
    .badge-notumor    { background:var(--green-lt);  color:var(--green);  border:1px solid #86efac; }
    .badge-pituitary  { background:var(--purple-lt); color:var(--purple); border:1px solid #c4b5fd; }

    .chip { display:inline-block; padding:2px 9px; border-radius:20px; font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
    .chip-high { background:#fee2e2; color:#991b1b; }
    .chip-mod  { background:#fef3c7; color:#92400e; }
    .chip-norm { background:#dcfce7; color:#166534; }

    .hist-row { display:flex; align-items:center; gap:0.75rem; padding:0.6rem 0.9rem; border:1px solid var(--border); border-radius:8px; margin-bottom:0.35rem; background:var(--bg-soft); font-size:0.82rem; }
    .hist-thumb { width:42px; height:42px; object-fit:cover; border-radius:6px; border:1px solid var(--border); flex-shrink:0; }
    .hist-info  { flex:1; }
    .hist-name  { font-weight:600; color:var(--text); }
    .hist-meta  { color:var(--text-muted); font-size:0.73rem; font-family:'JetBrains Mono',monospace; }

    .sec-title { font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; color:var(--text-muted); margin:1rem 0 0.4rem; }

    hr { border-color:var(--border) !important; }

    .stButton > button { background:var(--blue) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-family:'Inter',sans-serif !important; font-weight:600 !important; font-size:0.86rem !important; padding:0.5rem 1.3rem !important; }
    .stButton > button:hover { background:#1d4ed8 !important; }

    div[data-testid="stFileUploadDropzone"] { border:2px dashed var(--border-md) !important; border-radius:12px !important; background:var(--bg-soft) !important; }
    h2, h3, h4, h5 { color:var(--text) !important; }
    p, label, div  { color:var(--text); }
    .stRadio label  { color:var(--text) !important; }
    [data-testid="stMetricValue"] { color:var(--blue) !important; font-family:'JetBrains Mono',monospace; }
    .stAlert { border-radius:10px !important; }
    #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_COLORS = {
    "glioma":     "#dc2626",
    "meningioma": "#d97706",
    "notumor":    "#16a34a",
    "pituitary":  "#7c3aed",
}
CLASS_INFO = {
    "glioma": {
        "full_name":   "Glioma Tumor",
        "severity":    "High Risk",
        "chip":        "chip-high",
        "icon":        "🔴",
        "badge":       "badge-glioma",
        "description": "Gliomas originate in glial cells and are among the most common primary brain tumors. They vary in grade (I–IV), with grade IV (Glioblastoma) being the most aggressive.",
        "symptoms":    ["Headaches", "Seizures", "Memory loss", "Vision/speech changes"],
    },
    "meningioma": {
        "full_name":   "Meningioma Tumor",
        "severity":    "Moderate Risk",
        "chip":        "chip-mod",
        "icon":        "🟡",
        "badge":       "badge-meningioma",
        "description": "Meningiomas arise from the meninges (brain's protective membranes). Most are benign and slow-growing, but can cause significant neurological symptoms.",
        "symptoms":    ["Headaches", "Vision problems", "Hearing loss", "Memory issues"],
    },
    "notumor": {
        "full_name":   "No Tumor Detected",
        "severity":    "Normal",
        "chip":        "chip-norm",
        "icon":        "🟢",
        "badge":       "badge-notumor",
        "description": "The scan shows no evidence of a tumor. Brain tissue appears within normal parameters. Regular check-ups are still recommended.",
        "symptoms":    ["No concerning findings", "Normal brain structure"],
    },
    "pituitary": {
        "full_name":   "Pituitary Tumor",
        "severity":    "Moderate Risk",
        "chip":        "chip-mod",
        "icon":        "🟣",
        "badge":       "badge-pituitary",
        "description": "Pituitary adenomas develop in the pituitary gland. Most are non-cancerous but can disrupt hormone production and exert pressure on the optic nerves.",
        "symptoms":    ["Hormonal imbalance", "Vision changes", "Headaches", "Fatigue"],
    },
}

HISTORY_FILE  = "prediction_history.json"
FEEDBACK_FILE = "user_feedback.json"
EVAL_PATH     = "models/eval_results.json"


# ─── Persistence helpers ────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    return []

def save_feedback(feedback):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    candidates = [
        "models/best_model.keras",
        "models/best_model.h5",
        "models/brain_tumor_model.keras",
        "models/brain_tumor_model.h5",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                import tensorflow as tf
                m = tf.keras.models.load_model(p)
                return m, True
            except Exception as e:
                st.warning(f"Failed to load {p}: {e}")
    return None, False


def predict(model, img: Image.Image) -> dict:
    preds = model.predict(preprocess_image(img), verbose=0)[0]
    idx   = int(np.argmax(preds))
    return {
        "class":      CLASSES[idx],
        "confidence": float(preds[idx]),
        "all_probs":  {c: float(p) for c, p in zip(CLASSES, preds)},
    }


# ─── Adaptive Retraining ────────────────────────────────────────────────────────
def retrain_on_feedback(model):
    import tensorflow as tf
    feedback = load_feedback()
    if not feedback:
        return model, 0

    images, labels = [], []
    for entry in feedback:
        try:
            img_bytes = base64.b64decode(entry["image_b64"])
            img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))  # ✅ fixed
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(CLASSES.index(entry["true_label"]))
        except Exception:
            continue

    if not images:
        return model, 0

    X = np.array(images)
    Y = tf.keras.utils.to_categorical(labels, num_classes=len(CLASSES))

    # Unfreeze last 20 layers for fine-tuning
    for layer in model.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X, Y, epochs=3, batch_size=max(1, len(X)), verbose=0)

    os.makedirs("models", exist_ok=True)
    model.save("models/best_model.h5")
    save_feedback([])

    st.session_state["model_obj"] = model
    return model, len(images)


# ─── Evaluation helpers ─────────────────────────────────────────────────────────
def load_eval_results():
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            return json.load(f)
    return None


def _cm_fig(cm_matrix, class_names, title):
    cm    = np.array(cm_matrix)
    n     = len(class_names)
    total = cm.sum()
    annotations = [
        [f"{cm[i][j]}<br>({100 * cm[i][j] / max(total, 1):.1f}%)" for j in range(n)]
        for i in range(n)
    ]
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=[f"Pred: {c}" for c in class_names],
        y=[f"Actual: {c}" for c in class_names],
        annotation_text=annotations,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, family="Inter", color="#0f172a")),
        xaxis=dict(title="Predicted Label", side="bottom", tickfont=dict(family="Inter", size=11)),
        yaxis=dict(title="Actual Label", autorange="reversed", tickfont=dict(family="Inter", size=11)),
        margin=dict(t=70, l=160, b=110, r=20),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _metrics_bar_fig(metrics_dict, title):
    names  = list(metrics_dict.keys())
    values = [round(v, 4) for v in metrics_dict.values()]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family="Inter", color="#0f172a")),
        xaxis=dict(title="Score", range=[0, 1.15]),
        yaxis=dict(autorange="reversed", tickfont=dict(family="Inter", size=12)),
        height=300,
        margin=dict(t=50, l=120, r=90, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _comparison_fig(train_m, test_m):
    names      = list(train_m.keys())
    train_vals = [round(train_m[k], 4) for k in names]
    test_vals  = [round(test_m[k],  4) for k in names]
    fig = go.Figure(data=[
        go.Bar(name="Train", x=names, y=train_vals, marker_color="#4C78A8",
               text=[f"{v:.4f}" for v in train_vals], textposition="outside"),
        go.Bar(name="Test",  x=names, y=test_vals,  marker_color="#F58518",
               text=[f"{v:.4f}" for v in test_vals], textposition="outside"),
    ])
    fig.update_layout(
        barmode="group",
        title=dict(text="📊 Metrics Comparison: Train vs Test",
                   font=dict(size=17, family="Inter", color="#0f172a")),
        xaxis=dict(title="Metric", tickfont=dict(family="Inter", size=12)),
        yaxis=dict(title="Score", range=[0, 1.18]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420, margin=dict(t=80, b=60, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _history_figs(loss, val_loss, acc, val_acc):
    epochs = list(range(1, len(loss) + 1))

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=loss, mode="lines+markers", name="Train Loss",
                                  line=dict(color="#4C78A8", width=2), marker=dict(size=5)))
    loss_fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val Loss",
                                  line=dict(color="#E45756", width=2, dash="dash"), marker=dict(size=5)))
    loss_fig.update_layout(
        title=dict(text="Loss vs Epochs", font=dict(size=15, family="Inter", color="#0f172a")),
        xaxis=dict(title="Epoch"), yaxis=dict(title="Loss"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=epochs, y=acc, mode="lines+markers", name="Train Accuracy",
                                  line=dict(color="#72B7B2", width=2), marker=dict(size=5)))
    acc_fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Val Accuracy",
                                  line=dict(color="#54A24B", width=2, dash="dash"), marker=dict(size=5)))
    acc_fig.update_layout(
        title=dict(text="Accuracy vs Epochs", font=dict(size=15, family="Inter", color="#0f172a")),
        xaxis=dict(title="Epoch"), yaxis=dict(title="Accuracy", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return loss_fig, acc_fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 1.2rem;'>
        <div style='font-size:2rem;'>🧠</div>
        <div style='font-weight:700; font-size:1.05rem; color:#0f172a;'>NeuroScan</div>
        <div style='font-size:0.72rem; color:#94a3b8; letter-spacing:1px;'>Brain Tumor Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🔍 Classify", "🕘 History", "📊 Model Evaluation", "📚 Tumor Guide"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    model, model_ok = load_model()

    # Use retrained model from session_state if available
    if "model_obj" in st.session_state and st.session_state["model_obj"] is not None:
        model    = st.session_state["model_obj"]
        model_ok = True

    if model_ok:
        st.success("Model ready", icon="✅")
    else:
        st.warning("Model not found. Run `train_model.py` first.", icon="⚠️")

    st.markdown('<div class="sec-title">Adaptive Learning</div>', unsafe_allow_html=True)
    feedback_data = load_feedback()
    history_all   = load_history()

    col_a, col_b = st.columns(2)
    col_a.metric("Corrections", len(feedback_data))
    col_b.metric("Scans done",  len(history_all))

    if len(feedback_data) > 0 and model_ok:
        if st.button("🔁 Retrain on feedback", use_container_width=True):
            with st.spinner(f"Fine-tuning on {len(feedback_data)} correction(s)…"):
                updated_model, n = retrain_on_feedback(model)
            if n:
                st.cache_resource.clear()
                st.success(f"Model updated with {n} sample(s)!")
                st.rerun()
            else:
                st.error("No valid samples found.")
    else:
        st.caption("Correct a prediction below to start teaching the model.")

    st.markdown("---")
    if len(history_all) > 0:
        if st.button("🗑 Clear history", use_container_width=True):
            save_history([])
            st.rerun()


# ─── Top Bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <h1>🧠 Brain Tumor Detection</h1>
    <p>Upload a brain MRI scan · Get instant AI classification · Correct mistakes to improve the model</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — CLASSIFY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Classify":

    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown("#### Upload MRI Scan")
        uploaded = st.file_uploader(
            "Drag & drop or browse (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True, caption="Uploaded scan")
            st.markdown(f"""
            <div class="card card-blue" style="font-size:0.83rem; margin-top:0.6rem;">
                <b>File:</b> {uploaded.name} &nbsp;|&nbsp;
                <b>Size:</b> {img.size[0]}×{img.size[1]} px &nbsp;|&nbsp;
                <b>Mode:</b> {img.mode}
            </div>
            """, unsafe_allow_html=True)
            classify_btn = st.button("🔬 Classify Scan", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 1rem; background:#f8fafc;
                        border:2px dashed #e2e8f0; border-radius:12px; color:#94a3b8;">
                <div style="font-size:2.5rem;">🩻</div>
                <div style="font-size:0.88rem; margin-top:0.5rem;">Upload a brain MRI image to begin</div>
            </div>
            """, unsafe_allow_html=True)
            classify_btn = False

    with col_right:
        st.markdown("#### Results")

        if "last_result" not in st.session_state:
            st.session_state.last_result = None
        if "last_img" not in st.session_state:
            st.session_state.last_img = None

        if uploaded and classify_btn:
            if not model_ok:
                st.error("Model not loaded. Run `train_model.py` first.")
            else:
                with st.spinner("Analysing scan…"):
                    time.sleep(0.4)
                    result = predict(model, img)

                st.session_state.last_result = result
                st.session_state.last_img    = img

                history = load_history()
                history.append({
                    "filename":   uploaded.name,
                    "prediction": result["class"],
                    "confidence": round(result["confidence"] * 100, 1),
                    "all_probs":  {k: round(v * 100, 1) for k, v in result["all_probs"].items()},
                    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "thumb_b64":  img_to_b64(img),
                })
                save_history(history)

        if st.session_state.last_result:
            result = st.session_state.last_result
            cls    = result["class"]
            conf   = result["confidence"]
            info   = CLASS_INFO[cls]
            probs  = result["all_probs"]

            st.markdown(f"""
            <div class="card" style="text-align:center; padding:1.6rem;">
                <div style="font-size:2rem;">{info['icon']}</div>
                <div style="margin:0.5rem 0;">
                    <span class="badge {info['badge']}">{info['full_name']}</span>
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.3rem; font-family:'JetBrains Mono',monospace;">CONFIDENCE</div>
                <div style="font-size:2.1rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:{CLASS_COLORS[cls]};">
                    {conf*100:.1f}%
                </div>
                <span class="chip {info['chip']}">{info['severity']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sec-title">Confidence Breakdown</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for c, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                fig.add_trace(go.Bar(
                    x=[p * 100], y=[CLASS_INFO[c]["full_name"]],
                    orientation="h",
                    marker_color=CLASS_COLORS[c],
                    marker_opacity=1.0 if c == cls else 0.25,
                    text=f"{p*100:.1f}%", textposition="inside",
                    insidetextanchor="middle", showlegend=False,
                ))
            fig.update_layout(
                height=155, margin=dict(l=5, r=5, t=5, b=5),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
                yaxis=dict(color="#334155", tickfont=dict(family="Inter", size=11)),
            )
            ts_key = datetime.now().strftime("%H%M%S%f")
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"classify_conf_{ts_key}")

            st.markdown(f"""
            <div class="card card-{'green' if cls=='notumor' else 'red'}" style="font-size:0.84rem;">
                <div style="font-weight:600; margin-bottom:0.4rem;">Clinical Information</div>
                <p style="color:#334155; margin:0 0 0.4rem;">{info['description']}</p>
                <div style="color:#64748b; font-size:0.78rem;">
                    <b>Indicators:</b> {' · '.join(info['symptoms'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sec-title">Was this prediction correct?</div>', unsafe_allow_html=True)
            fb_correct = st.radio(
                "", ["✅ Yes, correct", "❌ No, let me correct it"],
                label_visibility="collapsed", key="fb_correct",
            )

            if fb_correct == "❌ No, let me correct it":
                true_label = st.selectbox(
                    "Select the correct label:",
                    options=CLASSES,
                    format_func=lambda c: f"{CLASS_INFO[c]['icon']}  {CLASS_INFO[c]['full_name']}",
                    key="fb_label",
                )
                if st.button("💾 Submit Correction", key="fb_submit"):
                    img_src = st.session_state.last_img or img
                    buf = BytesIO()
                    img_src.convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(buf, format="PNG")  # ✅ fixed
                    b64_full = base64.b64encode(buf.getvalue()).decode()

                    fb = load_feedback()
                    fb.append({
                        "image_b64":  b64_full,
                        "true_label": true_label,
                        "predicted":  cls,
                        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    save_feedback(fb)

                    hist = load_history()
                    if hist:
                        hist[-1]["corrected_label"] = true_label
                        save_history(hist)

                    st.success(
                        f"✅ Correction saved as **{CLASS_INFO[true_label]['full_name']}**. "
                        "Click **🔁 Retrain on feedback** in the sidebar to update the model."
                    )
            else:
                st.caption("Thank you for confirming! This helps validate the model.")

            st.caption("⚠️ For research use only — not a medical diagnostic tool.")

        elif not uploaded:
            st.markdown("""
            <div style="text-align:center; padding:5rem 1rem; background:#f8fafc;
                        border-radius:12px; border:1px solid #e2e8f0; color:#94a3b8;">
                <div style="font-size:2rem;">🔬</div>
                <div style="font-size:0.88rem; margin-top:0.5rem;">Results will appear here after classification</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🕘 History":
    history = load_history()
    st.markdown("#### Prediction History")

    if not history:
        st.markdown("""
        <div style="text-align:center; padding:4rem; background:#f8fafc;
                    border:1px solid #e2e8f0; border-radius:12px; color:#94a3b8;">
            <div style="font-size:2rem;">📭</div>
            <div style="margin-top:0.5rem; font-size:0.88rem;">No predictions yet. Classify a scan first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        counts    = {c: sum(1 for h in history if h["prediction"] == c) for c in CLASSES}
        corrected = sum(1 for h in history if "corrected_label" in h)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Scans",  len(history))
        c2.metric("Glioma",       counts["glioma"])
        c3.metric("Meningioma",   counts["meningioma"])
        c4.metric("No Tumor",     counts["notumor"])
        c5.metric("Pituitary",    counts["pituitary"])

        if corrected:
            st.info(f"🔄 {corrected} prediction(s) have been manually corrected.", icon="ℹ️")

        st.markdown('<div class="sec-title">Prediction Distribution</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=[CLASS_INFO[c]["full_name"] for c in CLASSES],
            y=[counts[c] for c in CLASSES],
            marker_color=[CLASS_COLORS[c] for c in CLASSES],
            text=[counts[c] for c in CLASSES],
            textposition="outside",
        ))
        fig2.update_layout(
            height=210, margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#334155", tickfont=dict(family="Inter", size=11), showgrid=False),
            yaxis=dict(color="#94a3b8", showgrid=True, gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": False},
                        key="history_dist_chart")

        st.markdown('<div class="sec-title">All Predictions — Most Recent First</div>', unsafe_allow_html=True)

        for entry in reversed(history):
            cls_  = entry["prediction"]
            info_ = CLASS_INFO[cls_]
            corrected_label = entry.get("corrected_label")

            thumb_html = ""
            if entry.get("thumb_b64"):
                thumb_html = f'<img class="hist-thumb" src="data:image/png;base64,{entry["thumb_b64"]}" />'

            correction_badge = ""
            if corrected_label:
                correction_badge = f"""
                <span style="font-size:0.7rem; background:#fef3c7; color:#92400e;
                             padding:1px 8px; border-radius:10px; margin-left:6px;">
                    ✏️ Corrected → {CLASS_INFO[corrected_label]['full_name']}
                </span>"""

            st.markdown(f"""
            <div class="hist-row">
                {thumb_html}
                <div class="hist-info">
                    <div class="hist-name">{info_['icon']} {info_['full_name']}{correction_badge}</div>
                    <div class="hist-meta">
                        {entry['filename']} &nbsp;·&nbsp;
                        {entry['confidence']}% confidence &nbsp;·&nbsp;
                        {entry['timestamp']}
                    </div>
                </div>
                <span class="badge {info_['badge']}" style="font-size:0.7rem; padding:2px 9px;">{cls_}</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📊 Full probability breakdown for every scan"):
            for i, entry in enumerate(reversed(history)):
                st.markdown(f"**#{i+1} — {entry['filename']}** &nbsp; `{entry['timestamp']}`")
                pf = go.Figure()
                for c in CLASSES:
                    p = entry["all_probs"].get(c, 0)
                    pf.add_trace(go.Bar(
                        x=[p], y=[CLASS_INFO[c]["full_name"]],
                        orientation="h",
                        marker_color=CLASS_COLORS[c],
                        marker_opacity=1.0 if c == entry["prediction"] else 0.25,
                        text=f"{p}%", textposition="inside",
                        insidetextanchor="middle", showlegend=False,
                    ))
                pf.update_layout(
                    height=125, margin=dict(l=5, r=5, t=5, b=5),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
                    yaxis=dict(color="#334155", tickfont=dict(family="Inter", size=10)),
                )
                safe_ts = entry['timestamp'].replace(' ', '_').replace(':', '-')
                st.plotly_chart(pf, use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"hist_prob_{i}_{safe_ts}")
                if entry.get("corrected_label"):
                    st.markdown(f"✏️ **Corrected label:** {CLASS_INFO[entry['corrected_label']]['full_name']}")
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":

    st.markdown("#### 📊 Model Evaluation Dashboard")
    st.markdown(
        "<p style='color:#64748b; font-size:0.88rem;'>"
        "Confusion matrices, performance metrics, and training curves "
        "generated after the last <code>train_model.py</code> run."
        "</p>",
        unsafe_allow_html=True,
    )

    eval_data = load_eval_results()

    if eval_data is None:
        st.warning(
            "**No evaluation data found.**\n\n"
            "Run `train_model.py` once to generate `models/eval_results.json`.",
            icon="⚠️",
        )
        st.code("python train_model.py", language="bash")
    else:
        class_names   = eval_data["class_names"]
        train_metrics = eval_data["train_metrics"]
        test_metrics  = eval_data["test_metrics"]
        train_cm      = eval_data["train_cm"]
        test_cm       = eval_data["test_cm"]
        loss          = eval_data["history_loss"]
        val_loss      = eval_data["history_val_loss"]
        acc           = eval_data["history_acc"]
        val_acc       = eval_data["history_val_acc"]

        # Training CM
        st.markdown("---")
        st.subheader("🔷 Training Confusion Matrix")
        st.plotly_chart(_cm_fig(train_cm, class_names, "Training Confusion Matrix"),
                        use_container_width=True, config={"displayModeBar": False},
                        key="eval_train_cm")

        st.markdown("##### Performance Metrics — Training")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",    f"{train_metrics['Accuracy']:.4f}")
        c2.metric("Precision",   f"{train_metrics['Precision']:.4f}")
        c3.metric("Recall",      f"{train_metrics['Recall']:.4f}")
        c4.metric("F1 Score",    f"{train_metrics['F1 Score']:.4f}")
        c5.metric("Specificity", f"{train_metrics['Specificity']:.4f}")
        st.plotly_chart(_metrics_bar_fig(train_metrics, "Training Metrics"),
                        use_container_width=True, config={"displayModeBar": False},
                        key="eval_train_metrics_bar")

        # Testing CM
        st.markdown("---")
        st.subheader("🔶 Testing Confusion Matrix")
        st.plotly_chart(_cm_fig(test_cm, class_names, "Testing Confusion Matrix"),
                        use_container_width=True, config={"displayModeBar": False},
                        key="eval_test_cm")

        st.markdown("##### Performance Metrics — Testing")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",    f"{test_metrics['Accuracy']:.4f}")
        c2.metric("Precision",   f"{test_metrics['Precision']:.4f}")
        c3.metric("Recall",      f"{test_metrics['Recall']:.4f}")
        c4.metric("F1 Score",    f"{test_metrics['F1 Score']:.4f}")
        c5.metric("Specificity", f"{test_metrics['Specificity']:.4f}")
        st.plotly_chart(_metrics_bar_fig(test_metrics, "Testing Metrics"),
                        use_container_width=True, config={"displayModeBar": False},
                        key="eval_test_metrics_bar")

        # Curves
        st.markdown("---")
        st.subheader("📈 Training vs Validation Performance")
        loss_fig, acc_fig = _history_figs(loss, val_loss, acc, val_acc)
        col_loss, col_acc = st.columns(2)
        with col_loss:
            st.plotly_chart(loss_fig, use_container_width=True,
                            config={"displayModeBar": False}, key="eval_loss_curve")
        with col_acc:
            st.plotly_chart(acc_fig, use_container_width=True,
                            config={"displayModeBar": False}, key="eval_acc_curve")

        # Comparison
        st.markdown("---")
        st.subheader("📊 Metrics Comparison: Train vs Test")
        st.plotly_chart(_comparison_fig(train_metrics, test_metrics),
                        use_container_width=True, config={"displayModeBar": False},
                        key="eval_comparison_chart")

        st.caption("⚠️ For research use only — not a medical diagnostic tool.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — TUMOR GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Tumor Guide":
    st.markdown("#### Brain Tumor Reference Guide")
    st.markdown(
        "<p style='color:#64748b; font-size:0.88rem;'>Educational overview of the four categories classified by the model.</p>",
        unsafe_allow_html=True,
    )
    for cls, info in CLASS_INFO.items():
        with st.expander(f"{info['icon']}  {info['full_name']}  —  {info['severity']}", expanded=False):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown("**Common symptoms / indicators:**")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
            with col_b:
                st.markdown(f"""
                <div class="card" style="text-align:center; padding:1.5rem;">
                    <div style="font-size:2.2rem;">{info['icon']}</div>
                    <div style="margin:0.5rem 0;"><span class="badge {info['badge']}">{cls}</span></div>
                    <span class="chip {info['chip']}">{info['severity']}</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not medical advice. Consult a licensed neurologist for diagnosis.")



    