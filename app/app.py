"""
app.py — SynFeasNet Streamlit UI
==================================
BUG FIX:
  - predict() returns a dict, not a tuple.
    Old code: prob, label = predict(smiles)  ← crashes with ValueError
    Fixed   : result = predict(smiles); prob = result["probability"]; ...
"""

import streamlit as st
from PIL import Image
import sys
import os

# Allow import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predict import predict


# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="SynFeasNet",
    layout="centered"
)


# ============================
# CUSTOM STYLE (WHITE + SUBTLE)
# ============================

st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f7fa;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
    }
    .warning-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #fff8e1;
        border: 1px solid #ffe082;
        margin-top: 10px;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)


# ============================
# TITLE
# ============================

st.title("SynFeasNet")
st.markdown("### Synthetic Feasibility Prediction System")


# ============================
# INPUT SECTION
# ============================

st.markdown("#### Enter SMILES string")

smiles = st.text_input(
    "",
    placeholder="e.g., CCO or CC1NC(=O)..."
)


# ============================
# PREDICTION
# ============================

if st.button("Predict"):

    if smiles.strip() == "":
        st.warning("Please enter a SMILES string")
    else:
        try:
            # BUG FIX: predict() returns a dict, not (prob, label) tuple.
            result = predict(smiles.strip())

            prob       = result["probability"]
            label      = result["label"]
            threshold  = result["threshold"]
            confidence = result["confidence"]
            chemistry  = result.get("chemistry", {})
            warning    = result.get("warning", "")

            # Choose colour based on label
            label_colour = "#27ae60" if "Synthesizable" in label and "Not" not in label else "#e74c3c"

            st.markdown(f"""
            <div class="result-box">
              <b>Prediction:</b>
              <span style="color:{label_colour}; font-weight:bold;">{label}</span>
              <br><br>
              <b>Probability:</b> {prob:.4f}
              &nbsp;&nbsp;|&nbsp;&nbsp;
              <b>Threshold:</b> {threshold:.4f}
              &nbsp;&nbsp;|&nbsp;&nbsp;
              <b>Confidence:</b> {confidence}
            </div>
            """, unsafe_allow_html=True)

            # Chemistry context
            if chemistry:
                st.markdown("**Molecular Properties**")
                cols = st.columns(3)
                cols[0].metric("MW",           f"{chemistry.get('molecular_weight', 'N/A')}")
                cols[1].metric("Heavy Atoms",  chemistry.get("num_heavy_atoms", "N/A"))
                cols[2].metric("Max Ring Size", chemistry.get("max_ring_size", "N/A"))

                if chemistry.get("is_macrocycle"):
                    st.info("🔵 Macrocyclic molecule detected (ring ≥ 8 atoms)")

            # Warning (e.g. very large molecule)
            if warning:
                st.markdown(f"""
                <div class="warning-box">⚠️ {warning}</div>
                """, unsafe_allow_html=True)

        except ValueError as e:
            st.error(f"Invalid SMILES: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)


# ============================
# BATCH PREDICTION (optional)
# ============================

with st.expander("Batch Prediction (paste one SMILES per line)"):
    batch_input = st.text_area("SMILES list", height=150,
                               placeholder="CCO\nCC(=O)Oc1ccccc1C(=O)O\n...")
    if st.button("Run Batch"):
        lines = [s.strip() for s in batch_input.strip().splitlines() if s.strip()]
        if not lines:
            st.warning("Enter at least one SMILES.")
        else:
            from inference.predict import predict_batch
            results = predict_batch(lines)
            rows = []
            for r in results:
                if "error" in r:
                    rows.append({"SMILES": r["smiles"], "Label": "Error",
                                 "Probability": "", "Confidence": r["error"]})
                else:
                    rows.append({
                        "SMILES":      r["smiles"],
                        "Label":       r["label"],
                        "Probability": f"{r['probability']:.4f}",
                        "Confidence":  r["confidence"],
                    })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows))


# ============================
# VISUALIZATION SECTION
# ============================

st.markdown("---")
st.markdown("### Model Performance")

# Resolve paths relative to this file
_HERE = os.path.dirname(os.path.abspath(__file__))

# Confusion Matrix
for _cm in [
    os.path.join(_HERE, "..", "checkpoints", "confusion_hybrid.png"),
    os.path.join(_HERE, "..", "training", "confusion_matrix.png"),
]:
    if os.path.exists(_cm):
        st.markdown("#### Confusion Matrix")
        st.image(Image.open(_cm), use_column_width=True)
        break

# Training Curve
for _cv in [
    os.path.join(_HERE, "..", "checkpoints", "curves_hybrid.png"),
    os.path.join(_HERE, "..", "training", "training_curve.png"),
]:
    if os.path.exists(_cv):
        st.markdown("#### Training Curve")
        st.image(Image.open(_cv), use_column_width=True)
        break