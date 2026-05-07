"""
predict.py  —  SynFeasNet v2
=====================================
Inference module.

Usage:
  python inference/predict.py          # runs built-in demo

  Or import:
    from inference.predict import predict
    result = predict("CCO")
    print(result)

Returns dict:
    {
        "probability": float,
        "label":       "Synthesizable" | "Not Synthesizable",
        "threshold":   float,
        "confidence":  "high" | "moderate" | "marginal (borderline)",
        "chemistry":   dict,
        "warning":     str,
    }

BUG FIXES:
  - CRITICAL: Uses SynFeasNetV2 (attention fusion + EGNN) instead of old SynFeasNet (v1).
              Old predict.py defined a local SynFeasNet that completely bypassed
              the EGNN branch and attention fusion — inference was running on
              the wrong architecture even if a v2 checkpoint was saved.
  - CRITICAL: _G3D_BUILDER now added globally so inference graphs have .pos
              attribute required by EGNNBranch.forward(). Without this, any
              checkpoint trained with train.py (v2) would crash at inference.
  - PROJECT_ROOT is auto-detected (no longer hardcoded to a Windows path).
  - HYBRID_CKPT / ORIGINAL_CKPT built from the auto-detected root.
"""

import os, sys
import torch
import torch.nn as nn
from torch_geometric.data import Batch

# ── Project root — auto-detected, works on all OS ────────────────────────────
_THIS_FILE   = os.path.abspath(__file__)
_HERE        = os.path.dirname(_THIS_FILE)

for _candidate in [os.path.dirname(_HERE), _HERE]:
    if os.path.isdir(os.path.join(_candidate, "models")):
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = _HERE

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ann_branch       import ANNBranch, ANNFeatureExtractor
from models.gat_branch       import GATBranch, GraphBuilder
from models.chemBERTa_branch import ChemBERTaBranch, SMILESTokenizer, DEFAULT_MAX_LENGTH
from models.attention_fusion import SynFeasNetV2          # ← v2 architecture
from models.egnn_branch      import Graph3DBuilder        # ← 3D coords for inference
from models.calibration      import TemperatureScaling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Checkpoint paths ──────────────────────────────────────────────────────────
HYBRID_CKPT   = os.path.join(PROJECT_ROOT, "checkpoints", "best_synfeasnet_hybrid.pth")
CACHE_HYBRID  = os.path.join(PROJECT_ROOT, "data", "cache_hybrid")
CACHE_ORIG    = os.path.join(PROJECT_ROOT, "data", "cache")


# ══════════════════════════════════════════════════════════════════════════════
# SMILES VALIDATION + CHEMISTRY CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def _validate(smiles: str):
    """
    Returns (is_valid: bool, warning: str, chemistry_context: dict)
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "RDKit could not parse this SMILES.", {}

    ring_info  = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    max_ring   = max(ring_sizes) if ring_sizes else 0

    ctx = {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "num_heavy_atoms":  mol.GetNumHeavyAtoms(),
        "max_ring_size":    max_ring,
        "is_macrocycle":    max_ring >= 8,
        "num_rings":        len(ring_sizes),
    }

    warning = ""
    if mol.GetNumHeavyAtoms() > 150:
        warning = (f"Large molecule ({mol.GetNumHeavyAtoms()} heavy atoms) "
                   "— prediction may be less reliable.")

    return True, warning, ctx


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _safe_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_model():
    """
    Loads the best available checkpoint into SynFeasNetV2.

    Returns (model, threshold, max_smiles_len, cache_hash, cache_dir)

    NOTE: Only the hybrid checkpoint (trained with the new train.py) is
    supported. If you have an old best_synfeasnet.pth from the v1 model,
    retrain using training/train.py to get a v2 checkpoint.
    """
    if not os.path.exists(HYBRID_CKPT):
        raise FileNotFoundError(
            f"No checkpoint found at: {HYBRID_CKPT}\n\n"
            "You need to train the model first.  Run:\n"
            "  python training/train.py\n\n"
            "This will create checkpoints/best_synfeasnet_hybrid.pth"
        )

    print(f"[predict] Loading SynFeasNetV2 from: {HYBRID_CKPT}")
    ckpt = _safe_load(HYBRID_CKPT)

    # Validate the checkpoint was produced by the v2 training pipeline
    cfg  = ckpt.get("config", {})
    arch = cfg.get("architecture", "unknown")
    if arch not in ("SynFeasNetV2", "unknown"):
        print(f"  [WARNING] Unexpected architecture in checkpoint: {arch}. "
              "Expected SynFeasNetV2.  Attempting to load anyway...")

    # BUG FIX: instantiate SynFeasNetV2, not the old SynFeasNet.
    model = SynFeasNetV2(dropout=0.3).to(device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        threshold  = float(ckpt.get("threshold", 0.5))
        max_len    = int(cfg.get("max_smiles_len", DEFAULT_MAX_LENGTH))
        cache_hash = cfg.get("cache_hash", None)
        print(f"  Epoch     : {ckpt.get('epoch', '?')}")
        print(f"  Label ver : {cfg.get('label_version', '?')}")
        print(f"  Arch      : {arch}")
        roc = ckpt.get("val_roc_auc"); pr = ckpt.get("val_pr_auc")
        if roc: print(f"  Val ROC   : {roc:.4f}")
        if pr:  print(f"  Val PR    : {pr:.4f}")
        print(f"  Threshold : {threshold:.4f}")
    else:
        # raw state dict without metadata
        model.load_state_dict(ckpt)
        threshold  = 0.5
        max_len    = DEFAULT_MAX_LENGTH
        cache_hash = None

    model.eval()
    return model, threshold, max_len, cache_hash, CACHE_HYBRID


def _load_extractors(max_len, cache_hash, cache_dir):
    """
    Loads the descriptor scaler that was saved during training.
    Also returns a Graph3DBuilder for 3D coordinate generation at inference.
    """
    if cache_hash is None:
        raise RuntimeError(
            "Checkpoint has no cache_hash — it may be from an old run.\n"
            "Retrain using training/train.py to get a compatible checkpoint."
        )

    scaler_path = os.path.join(cache_dir, f"descriptor_scaler_{cache_hash}.npz")

    if not os.path.exists(scaler_path):
        alt = os.path.join(CACHE_ORIG, f"descriptor_scaler_{cache_hash}.npz")
        if os.path.exists(alt):
            scaler_path = alt
        else:
            raise FileNotFoundError(
                f"Descriptor scaler not found: {scaler_path}\n"
                "Delete the cache folder and retrain."
            )

    ann_extractor = ANNFeatureExtractor()
    ann_extractor.load_descriptor_scaler(scaler_path)
    graph_builder = GraphBuilder()
    tokenizer     = SMILESTokenizer(max_length=max_len)

    # BUG FIX: Graph3DBuilder required for SynFeasNetV2 inference.
    # EGNNBranch.forward() does `x = data.pos` — without .pos the model crashes.
    g3d_builder   = Graph3DBuilder()

    return ann_extractor, graph_builder, tokenizer, g3d_builder


def _confidence_label(prob: float, threshold: float) -> str:
    margin = abs(prob - threshold)
    if margin < 0.05:
        return "marginal (borderline)"
    if margin < 0.20:
        return "moderate"
    return "high"


# ══════════════════════════════════════════════════════════════════════════════
# LAZY GLOBALS  — loaded once on first call
# ══════════════════════════════════════════════════════════════════════════════

_MODEL         = None
_THRESHOLD     = None
_ANN_EXTRACTOR = None
_GRAPH_BUILDER = None
_TOKENIZER     = None
_G3D_BUILDER   = None   # BUG FIX: was missing — required for EGNN at inference


def _ensure_loaded():
    global _MODEL, _THRESHOLD, _ANN_EXTRACTOR, _GRAPH_BUILDER, _TOKENIZER, _G3D_BUILDER
    if _MODEL is None:
        _MODEL, _THRESHOLD, max_len, cache_hash, cache_dir = _load_model()
        (_ANN_EXTRACTOR, _GRAPH_BUILDER,
         _TOKENIZER, _G3D_BUILDER) = _load_extractors(max_len, cache_hash, cache_dir)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def predict(smiles: str) -> dict:
    """
    Predict synthesizability for a single SMILES string.

    Returns:
        {
            "probability"  : float in [0, 1],
            "label"        : "Synthesizable" or "Not Synthesizable",
            "threshold"    : float,
            "confidence"   : "high" / "moderate" / "marginal (borderline)",
            "chemistry"    : dict  (MW, num_heavy_atoms, max_ring_size, ...),
            "warning"      : str   (empty string if no issues),
        }
    """
    _ensure_loaded()

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    smiles = smiles.strip()

    valid, warning, chemistry = _validate(smiles)
    if not valid:
        raise ValueError(f"Invalid SMILES — {warning}")

    # ANN features
    ann_feat = torch.tensor(
        _ANN_EXTRACTOR.compute(smiles), dtype=torch.float32
    ).unsqueeze(0).to(device)

    # Graph features — 2D topology
    raw_graph = _GRAPH_BUILDER.build(smiles)

    # BUG FIX: Add 3D coordinates for EGNN branch.
    # Without .pos, EGNNBranch.forward() raises AttributeError: 'Data' object
    # has no attribute 'pos'. This was missing in the old predict.py.
    graph_3d = _G3D_BUILDER.add_coords(raw_graph, smiles)
    graph    = Batch.from_data_list([graph_3d]).to(device)

    # SMILES tokens
    tok       = _TOKENIZER(smiles)
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)

    with torch.no_grad():
        logits = _MODEL(ann_feat, graph, input_ids, attn_mask)
        prob   = torch.sigmoid(logits).item()

    return {
        "probability": round(prob, 4),
        "label":       "Synthesizable" if prob >= _THRESHOLD else "Not Synthesizable",
        "threshold":   round(_THRESHOLD, 4),
        "confidence":  _confidence_label(prob, _THRESHOLD),
        "chemistry":   chemistry,
        "warning":     warning,
    }


def predict_batch(smiles_list: list) -> list:
    """Predict synthesizability for a list of SMILES strings."""
    results = []
    for smi in smiles_list:
        try:
            out           = predict(smi)
            out["smiles"] = smi
            results.append(out)
        except Exception as e:
            results.append({"smiles": smi, "error": str(e)})
    return results


def predict_with_uncertainty(smiles: str, n_samples: int = 20) -> dict:
    """
    MC-Dropout uncertainty estimation for a single SMILES.

    Returns the standard predict() dict plus:
        "prob_std"  : std-dev across MC samples (uncertainty proxy)
        "prob_mean" : mean probability across MC samples
    """
    _ensure_loaded()

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    smiles = smiles.strip()

    valid, warning, chemistry = _validate(smiles)
    if not valid:
        raise ValueError(f"Invalid SMILES — {warning}")

    ann_feat  = torch.tensor(
        _ANN_EXTRACTOR.compute(smiles), dtype=torch.float32
    ).unsqueeze(0).to(device)

    raw_graph = _GRAPH_BUILDER.build(smiles)
    graph_3d  = _G3D_BUILDER.add_coords(raw_graph, smiles)
    graph     = Batch.from_data_list([graph_3d]).to(device)

    tok       = _TOKENIZER(smiles)
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)

    unc = _MODEL.predict_with_uncertainty(
        ann_feat, graph, input_ids, attn_mask, n_samples=n_samples
    )

    prob_mean = float(unc["prob_mean"].item())
    prob_std  = float(unc["prob_std"].item())

    return {
        "probability": round(prob_mean, 4),
        "prob_std":    round(prob_std, 4),
        "label":       "Synthesizable" if prob_mean >= _THRESHOLD else "Not Synthesizable",
        "threshold":   round(_THRESHOLD, 4),
        "confidence":  _confidence_label(prob_mean, _THRESHOLD),
        "chemistry":   chemistry,
        "warning":     warning,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Caffeine",      "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
        ("Ethanol",       "CCO"),
        ("Cyclosporin A", "CC[C@@H]1NC(=O)[C@H]([C@H](O)[C@H](C)C/C=C/C)N(C)"
                          "C(=O)[C@H](C(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)"),
        ("Impossible",    "C(#N)(#N)(#N)(#N)"),
    ]

    print("\n" + "=" * 70)
    print("SynFeasNet v2 — Inference Demo (SynFeasNetV2)")
    print("=" * 70)

    for name, smi in test_cases:
        print(f"\n{name}")
        print(f"  SMILES : {smi[:80]}{'...' if len(smi) > 80 else ''}")
        try:
            r = predict(smi)
            print(f"  Result : {r['label']}")
            print(f"  Prob   : {r['probability']:.4f}  (threshold={r['threshold']})")
            print(f"  Conf   : {r['confidence']}")
            c = r["chemistry"]
            print(f"  Chem   : MW={c['molecular_weight']}  "
                  f"Atoms={c['num_heavy_atoms']}  "
                  f"MaxRing={c['max_ring_size']}")
            if r["warning"]:
                print(f"  ⚠      : {r['warning']}")
        except Exception as e:
            print(f"  ERROR  : {e}")