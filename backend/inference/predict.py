"""
predict.py — SynPractIQ v3
==============================
Inference module for the Synthetic Practicality Index (SPI).

Returns a rich SPI report instead of a single binary label.

Usage:
  python inference/predict.py         # runs built-in demo

  Or import:
    from inference.predict import predict
    result = predict("CCO")
    print(result["spi_score"])
    print(result["spi_report"])

Returns dict:
    {
        "stage1_pass"    : bool,
        "stage1_prob"    : float,
        "spi_score"      : float,           # composite [0,1]
        "spi_class"      : int,             # 0=intractable … 4=trivial
        "spi_label"      : str,
        "spi_dimensions" : dict[str, float], # 6 sub-scores
        "spi_report"     : str,             # human-readable summary
        "chemistry"      : dict,
        "uncertainty"    : dict,            # when predict_with_uncertainty() used
        "warning"        : str,
    }
"""

import os, sys, json
import torch
import numpy as np
from torch_geometric.data import Batch

# ── Project root auto-detection ───────────────────────────────────────────────
_THIS_FILE = os.path.abspath(__file__)
_HERE      = os.path.dirname(_THIS_FILE)
for _candidate in [os.path.dirname(_HERE), _HERE]:
    if os.path.isdir(os.path.join(_candidate, "models")):
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = _HERE
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ann_branch       import ANNFeatureExtractor
from models.gat_branch       import GATBranch, GraphBuilder
from models.chemBERTa_branch import SMILESTokenizer, DEFAULT_MAX_LENGTH
from models.attention_fusion import SynPractIQModel, SPI_DIMENSION_NAMES
from models.egnn_branch      import Graph3DBuilder
from spi_labels              import (
    SPILabelGenerator, SPI_CLASS_THRESHOLDS, FEASIBILITY_GATE_THRESHOLD
)
from retrosynthesis.engine   import RetrosynthesisEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH  = os.path.join(PROJECT_ROOT, "checkpoints", "best_synpractiq.pth")
CACHE_DIR  = os.path.join(PROJECT_ROOT, "data", "cache_spi")

# SPI class labels (index 0–4)
_CLASS_LABELS = ["intractable", "difficult", "challenging", "practical", "trivial"]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _validate_smiles(smiles: str) -> tuple:
    """Returns (is_valid: bool, warning: str, chemistry: dict)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "RDKit could not parse this SMILES.", {}

    ring_info  = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    max_ring   = max(ring_sizes) if ring_sizes else 0

    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    ctx = {
        "molecular_weight":    round(Descriptors.MolWt(mol), 2),
        "num_heavy_atoms":     mol.GetNumHeavyAtoms(),
        "max_ring_size":       max_ring,
        "is_macrocycle":       max_ring >= 8,
        "num_rings":           len(ring_sizes),
        "num_stereocenters":   len(chiral_centers),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "logp":                round(Descriptors.MolLogP(mol), 2),
        "tpsa":                round(Descriptors.TPSA(mol), 1),
    }

    warnings = []
    if mol.GetNumHeavyAtoms() > 150:
        warnings.append(f"Very large molecule ({mol.GetNumHeavyAtoms()} heavy atoms) — less reliable prediction.")
    if max_ring >= 12:
        warnings.append(f"Macrocycle (ring size {max_ring}) — synthesis may be especially challenging.")
    if len(chiral_centers) > 6:
        warnings.append(f"{len(chiral_centers)} stereocenters — stereoselective synthesis required.")

    return True, " ".join(warnings), ctx


def _spi_class_from_score(spi: float) -> tuple:
    """Returns (class_int, label_str)."""
    thresholds = [
        (0.75, 4, "trivial"),
        (0.55, 3, "practical"),
        (0.35, 2, "challenging"),
        (0.15, 1, "difficult"),
        (0.00, 0, "intractable"),
    ]
    for lo, cls_int, label in thresholds:
        if spi >= lo:
            return cls_int, label
    return 0, "intractable"


def _build_spi_report(spi_score: float, spi_class: int, spi_label: str,
                      dimensions: dict, stage1_pass: bool,
                      chemistry: dict) -> str:
    """Build a human-readable SPI report string."""
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        f"║  Synthetic Practicality Index (SPI)                ║",
        f"║  Score  : {spi_score:.3f}   Class {spi_class} — {spi_label.upper():12s}     ║",
        f"║  Stage 1 Gate: {'✓ PASS' if stage1_pass else '✗ FAIL'}                            ║",
        "╠══════════════════════════════════════════════════════╣",
    ]

    dim_labels = {
        "synthetic_complexity":   "Synthetic Complexity  ",
        "route_practicality":     "Route Practicality    ",
        "precursor_availability": "Precursor Availability",
        "scalability":            "Scalability           ",
        "retro_confidence":       "Retro Confidence      ",
        "medchem_realism":        "MedChem Realism       ",
    }

    for dim, label in dim_labels.items():
        score = dimensions.get(dim, 0.0)
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"║  {label}: {bar} {score:.3f} ║")

    lines.append("╠══════════════════════════════════════════════════════╣")

    # Interpretation
    mw      = chemistry.get("molecular_weight", "?")
    n_rings = chemistry.get("num_rings", "?")
    n_ster  = chemistry.get("num_stereocenters", "?")
    lines.append(f"║  MW={mw}  Rings={n_rings}  Stereocenters={n_ster:<2}           ║")
    lines.append("╚══════════════════════════════════════════════════════╝")

    # Narrative
    if spi_class == 4:
        lines.append("→ Trivial synthesis — standard building blocks, high yield expected.")
    elif spi_class == 3:
        lines.append("→ Practical synthesis — feasible in a well-equipped medicinal chemistry lab.")
    elif spi_class == 2:
        lines.append("→ Challenging — requires specialist knowledge; multi-step route expected.")
    elif spi_class == 1:
        lines.append("→ Difficult — significant synthetic expertise needed; low throughput likely.")
    else:
        lines.append("→ Intractable — not practical for standard pharmaceutical manufacturing.")

    # Flag the weakest dimension
    if dimensions:
        worst_dim = min(dimensions, key=dimensions.get)
        worst_val = dimensions[worst_dim]
        if worst_val < 0.35:
            lines.append(f"⚠  Bottleneck: {worst_dim.replace('_', ' ').title()} (score={worst_val:.3f})")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (lazy globals)
# ══════════════════════════════════════════════════════════════════════════════

_MODEL         = None
_ANN_EXTRACTOR = None
_GRAPH_BUILDER = None
_TOKENIZER     = None
_G3D_BUILDER   = None
_RETRO_ENGINE  = None
_MAX_LEN       = DEFAULT_MAX_LENGTH


def _load_model():
    global _MODEL, _ANN_EXTRACTOR, _GRAPH_BUILDER, _TOKENIZER, _G3D_BUILDER, _MAX_LEN

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at: {CKPT_PATH}\n\n"
            "Train the model first:\n  python training/train.py\n\n"
            "This creates checkpoints/best_synpractiq.pth"
        )

    print(f"[predict] Loading SynPractIQModel from: {CKPT_PATH}")
    ckpt = _safe_load(CKPT_PATH)
    cfg  = ckpt.get("config", {})

    _MAX_LEN = int(cfg.get("max_smiles_len", DEFAULT_MAX_LENGTH))
    cache_hash = cfg.get("cache_hash", None)

    _MODEL = SynPractIQModel(dropout=0.3).to(device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        _MODEL.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  Val SPI-MAE: {ckpt.get('val_spi_mae', '?')}")
        print(f"  Val ρ      : {ckpt.get('val_spearman', '?')}")
    else:
        _MODEL.load_state_dict(ckpt)

    _MODEL.eval()

    # Load descriptor scaler
    if cache_hash:
        scaler_path = os.path.join(CACHE_DIR, f"descriptor_scaler_{cache_hash}.npz")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Descriptor scaler not found: {scaler_path}\n"
                "Delete cache and retrain."
            )
        _ANN_EXTRACTOR = ANNFeatureExtractor()
        _ANN_EXTRACTOR.load_descriptor_scaler(scaler_path)
    else:
        _ANN_EXTRACTOR = ANNFeatureExtractor()
        print("  [WARNING] No cache_hash in checkpoint — scaler not loaded. "
              "Predictions may be less accurate.")

    _GRAPH_BUILDER = GraphBuilder()
    _TOKENIZER     = SMILESTokenizer(max_length=_MAX_LEN)
    _G3D_BUILDER   = Graph3DBuilder()
    print("[predict] Model and extractors ready.")


def _ensure_loaded():
    if _MODEL is None:
        _load_model()


def _ensure_retro_loaded():
    global _RETRO_ENGINE

    if _RETRO_ENGINE is None:
        _RETRO_ENGINE = RetrosynthesisEngine()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_inputs(smiles: str):
    """Build all model inputs for one molecule."""
    ann_feat = torch.tensor(
        _ANN_EXTRACTOR.compute(smiles), dtype=torch.float32
    ).unsqueeze(0).to(device)

    raw_graph = _GRAPH_BUILDER.build(smiles)
    graph_3d  = _G3D_BUILDER.add_coords(raw_graph, smiles)
    graph     = Batch.from_data_list([graph_3d]).to(device)

    tok       = _TOKENIZER(smiles)
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)

    return ann_feat, graph, input_ids, attn_mask


def _parse_output(out: dict, chemistry: dict, warning: str) -> dict:
    """Convert raw model output dict into user-facing result dict."""
    sub_scores = out["sub_scores"].squeeze(0).cpu().numpy()    # (6,)
    spi_score  = float(out["spi_score"].squeeze().cpu().item())
    stage1_prob = float(torch.sigmoid(out["stage1_logit"]).squeeze().cpu().item())
    stage1_pass = stage1_prob >= FEASIBILITY_GATE_THRESHOLD

    dimensions = {
        SPI_DIMENSION_NAMES[i]: float(sub_scores[i])
        for i in range(len(SPI_DIMENSION_NAMES))
    }

    spi_class, spi_label = _spi_class_from_score(spi_score)
    report = _build_spi_report(
        spi_score, spi_class, spi_label, dimensions,
        stage1_pass, chemistry
    )

    return {
        "stage1_pass":    stage1_pass,
        "stage1_prob":    round(stage1_prob, 4),
        "spi_score":      round(spi_score, 4),
        "spi_class":      spi_class,
        "spi_label":      spi_label,
        "spi_dimensions": {k: round(v, 4) for k, v in dimensions.items()},
        "spi_report":     report,
        "chemistry":      chemistry,
        "warning":        warning,
    }


def _retro_metric_defaults(result: dict) -> dict:
    """Map SPI dimension scores to lightweight retrosynthesis scoring inputs."""
    dimensions = result.get("spi_dimensions", {})

    synthetic_complexity = float(
        dimensions.get("synthetic_complexity", result.get("spi_score", 0.5))
    )
    retro_confidence = float(
        dimensions.get("retro_confidence", result.get("spi_score", 0.5))
    )
    medchem_realism = float(
        dimensions.get("medchem_realism", result.get("spi_score", 0.5))
    )

    return {
        "spi_score": float(result.get("spi_score", 0.0)),
        "sa_score": round(10.0 - 9.0 * synthetic_complexity, 4),
        "scscore": round(5.0 - 4.0 * retro_confidence, 4),
        "syba_score": round(medchem_realism, 4),
    }


def _run_retrosynthesis(smiles: str, result: dict) -> dict:
    _ensure_retro_loaded()

    metrics = _retro_metric_defaults(result)

    try:
        with _RETRO_LOCK:
            retro_json = _RETRO_ENGINE.run_json(
                smiles=smiles,
                **metrics
            )

        retro_json["enabled"] = True
        retro_json["scoring_inputs"] = metrics

        if retro_json["status"] == "no_route":
            retro_json["message"] = (
                "No retrosynthesis route generated. The molecule may be a "
                "macrocycle, invalid, already terminal, or outside the current "
                "template coverage."
            )

        return retro_json

    except Exception as exc:
        return {
            "enabled": True,
            "status": "error",
            "target_smiles": smiles,
            "error": str(exc),
            "scoring_inputs": metrics,
            "n_routes": 0,
            "summary": {
                "best_score": None,
                "best_solved_fraction": 0.0,
                "best_n_steps": 0,
            },
            "routes": [],
            "visualization": {
                "nodes": [],
                "edges": [],
                "layout": "tree",
                "root_id": None,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

import threading

# Global lock for model inference to ensure thread-safety in FastAPI/multiprocessing
_MODEL_LOCK = threading.Lock()
_RETRO_LOCK = threading.Lock()

def predict(smiles: str, include_retrosynthesis: bool = True) -> dict:
    """
    Predict the Synthetic Practicality Index for a SMILES string.
    """
    _ensure_loaded()

    smiles = smiles.strip()
    valid, warning, chemistry = _validate_smiles(smiles)
    if not valid:
        raise ValueError(f"Invalid SMILES — {warning}")

    ann_feat, graph, input_ids, attn_mask = _prepare_inputs(smiles)

    with _MODEL_LOCK:
        with torch.no_grad():
            out = _MODEL(ann_feat, graph, input_ids, attn_mask)

    result = _parse_output(out, chemistry, warning)

    if include_retrosynthesis:
        result["retrosynthesis"] = _run_retrosynthesis(
            smiles,
            result
        )

    return result



def predict_batch(smiles_list: list, include_retrosynthesis: bool = True) -> list:
    """Predict SPI for a list of SMILES strings."""
    _ensure_loaded()
    results = []
    for smi in smiles_list:
        try:
            r = predict(
                smi,
                include_retrosynthesis=include_retrosynthesis
            )
            r["smiles"] = smi
            results.append(r)
        except Exception as e:
            results.append({"smiles": smi, "error": str(e)})
    return results


def predict_with_uncertainty(
    smiles: str,
    n_samples: int = 20,
    include_retrosynthesis: bool = True
) -> dict:
    """
    MC-Dropout uncertainty estimation.
    Returns standard predict() dict plus 'uncertainty' sub-dict.
    """
    _ensure_loaded()

    smiles = smiles.strip()
    valid, warning, chemistry = _validate_smiles(smiles)
    if not valid:
        raise ValueError(f"Invalid SMILES — {warning}")

    ann_feat, graph, input_ids, attn_mask = _prepare_inputs(smiles)

    unc = _MODEL.predict_with_uncertainty(
        ann_feat, graph, input_ids, attn_mask, n_samples=n_samples
    )

    spi_mean = float(unc["spi_mean"].squeeze().item())
    spi_std  = float(unc["spi_std"].squeeze().item())
    sub_mean = unc["sub_mean"].squeeze(0).cpu().numpy()
    sub_std  = unc["sub_std"].squeeze(0).cpu().numpy()

    result = _parse_output(
        {
            "sub_scores":   unc["sub_mean"],
            "spi_score":    unc["spi_mean"],
            "stage1_logit": torch.zeros(1, 1),  # not used in MC mode
        },
        chemistry, warning
    )
    result["uncertainty"] = {
        "spi_std":      round(spi_std, 4),
        "spi_mean":     round(spi_mean, 4),
        "sub_score_std": {
            SPI_DIMENSION_NAMES[i]: round(float(sub_std[i]), 4)
            for i in range(len(SPI_DIMENSION_NAMES))
        },
    }

    if include_retrosynthesis:
        result["retrosynthesis"] = _run_retrosynthesis(
            smiles,
            result
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Caffeine",      "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
        ("Ethanol",       "CCO"),
        ("Cyclosporin A", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ]

    print("\n" + "=" * 70)
    print("SynPractIQ v3 — Inference Demo")
    print("=" * 70)

    for name, smi in test_cases:
        print(f"\n{'─'*70}\n{name}")
        try:
            r = predict(smi)
            print(r["spi_report"])
            print("\nRETROSYNTHESIS JSON")
            print(json.dumps(
                r.get("retrosynthesis", {}),
                indent=2
            ))
            if r["warning"]:
                print(f"⚠  {r['warning']}")
        except Exception as e:
            print(f"  ERROR: {e}")
