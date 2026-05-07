"""
train.py  —  SynFeasNet v2
===================================
Place this file in:  SynFeasNet/training/train.py   (or scripts/train.py)

Run:  python training/train.py

What this file does vs the old train.py:
  1. Trains SynFeasNetV2 (4-branch + attention fusion) — NOT the old v1 cat model
  2. Graph pre-computation now adds 3D coordinates (.pos) for the EGNN branch
  3. Config hash version bumped to "hybrid_v3_egnn" — forces cache regeneration
     so old caches without .pos are NOT reused
  4. All other logic (focal loss, threshold search, plots, XGBoost baseline) kept
  5. Checkpoint saved to best_synfeasnet_hybrid.pth  (same path as before)

BUG FIXES:
  - CRITICAL: SynFeasNet (v1 concat) replaced with SynFeasNetV2 (v2 attention)
  - CRITICAL: Graph3DBuilder.add_coords() now called in precompute_features()
              so EGNN branch receives proper .pos coordinates
  - label_final / label_confidence auto-derived when missing from CSV
  - PROJECT_ROOT auto-detected (no hardcoded Windows path)
  - CSV_PATH falls back gracefully: dataset_hybrid.csv → dataset.csv
  - torch.amp import made backward-compatible (PyTorch 2.x)
"""

import os, sys, time, json, hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score,
    precision_recall_curve, classification_report,
)
from torch_geometric.data import Batch

# ── torch.amp backward-compatibility ────────────────────────────────────────
try:
    from torch.amp import GradScaler
    def _autocast(enabled: bool):
        return torch.amp.autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler       # type: ignore[no-redef]
    from torch.cuda.amp import autocast as _cuda_autocast
    def _autocast(enabled: bool):
        return _cuda_autocast(enabled=enabled)

# ── Project root — auto-detected ─────────────────────────────────────────────
_THIS_FILE   = os.path.abspath(__file__)
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if not os.path.isdir(os.path.join(PROJECT_ROOT, "models")):
    PROJECT_ROOT = _SCRIPTS_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Imports (v2 architecture) ─────────────────────────────────────────────────
from models.ann_branch       import ANNBranch, ANNFeatureExtractor
from models.gat_branch       import GATBranch, GraphBuilder
from models.chemBERTa_branch import ChemBERTaBranch, SMILESTokenizer
from models.attention_fusion import SynFeasNetV2          # ← v2 attention model
from models.egnn_branch      import Graph3DBuilder        # ← 3D coords
from models.calibration      import TemperatureScaling

# ── Paths ─────────────────────────────────────────────────────────────────────
_HYBRID_CSV  = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_hybrid.csv")
_RAW_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.csv")
CSV_PATH     = _HYBRID_CSV if os.path.exists(_HYBRID_CSV) else _RAW_CSV
CACHE_DIR    = os.path.join(PROJECT_ROOT, "data", "cache_hybrid")
SAVE_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE     = 16     # reduced from 32 — v2 model is larger (EGNN + attention)
EPOCHS         = 25
LR             = 2e-4
MAX_SMILES_LEN = 320
FOCAL_GAMMA    = 2.0
RUN_BASELINE   = True

WEIGHT_HIGH   = 1.0   # ChEMBL / PubChem / chemistry-impossible  (certain)
WEIGHT_MEDIUM = 0.5   # heuristic agreement (2/3 scores)          (uncertain)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,  exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════════
# LABEL DERIVATION  — runs when dataset.csv (no label_final) is loaded
# ══════════════════════════════════════════════════════════════════════════════

_TRUSTED_POS_SOURCES = {"ChEMBL", "PubChem"}
_SA_SYNTH  = 4.5;  _SA_HARD  = 6.5
_SYBA_SYNTH = 0.0
_SCS_SYNTH = 3.8;  _SCS_HARD = 4.5


def _classify_row(row) -> tuple:
    source  = str(row.get("source", "")).strip()
    sa      = row.get("sascore",    np.nan)
    syba    = row.get("syba_score", np.nan)
    scs     = row.get("scscore",    np.nan)

    def _val(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else np.nan
        except Exception:
            return np.nan

    sa, syba, scs = _val(sa), _val(syba), _val(scs)

    if source in _TRUSTED_POS_SOURCES:
        return 1, "high"

    sa_ok    = (not np.isnan(sa))   and (sa   <= _SA_SYNTH)
    syba_ok  = (not np.isnan(syba)) and (syba >  _SYBA_SYNTH)
    scs_ok   = (not np.isnan(scs))  and (scs  <= _SCS_SYNTH)

    sa_hard   = (not np.isnan(sa))   and (sa   >= _SA_HARD)
    syba_hard = (not np.isnan(syba)) and (syba < -10.0)
    scs_hard  = (not np.isnan(scs))  and (scs  >= _SCS_HARD)

    synth_votes = int(sa_ok) + int(syba_ok) + int(scs_ok)
    hard_votes  = int(sa_hard) + int(syba_hard) + int(scs_hard)

    if synth_votes >= 2:
        return 1, "medium"
    if hard_votes >= 2:
        return 0, "medium"

    fallback = int(row.get("label_synthesizable", 0))
    return fallback, "medium"


def _derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("  Deriving label_final + label_confidence from chemistry-first rules...")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    labels, confidences = [], []
    for _, row in df.iterrows():
        smi = str(row.get("smiles_canonical", ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            labels.append(0); confidences.append("high")
            continue
        lbl, conf = _classify_row(row)
        labels.append(lbl); confidences.append(conf)

    df = df.copy()
    df["label_final"]      = labels
    df["label_confidence"] = confidences
    print(f"  label_final distribution:\n{pd.Series(labels).value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CACHE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _config_hash() -> str:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    stat = os.stat(CSV_PATH)
    cfg  = {
        "csv_size":     stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
        "smiles_len":   MAX_SMILES_LEN,
        "node_dim":     GATBranch.NODE_DIM,
        # BUG FIX: version bumped to "hybrid_v3_egnn" so old caches without
        # 3D .pos coordinates are NOT reused — they would crash EGNNBranch.
        "version":      "hybrid_v3_egnn",
    }
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


CONFIG_HASH = _config_hash()
print(f"Cache hash : {CONFIG_HASH}")


def _cache_paths(tag: str):
    h = CONFIG_HASH
    return (
        os.path.join(CACHE_DIR, f"{tag}_ann_{h}.pt"),
        os.path.join(CACHE_DIR, f"{tag}_graphs_{h}.pt"),
        os.path.join(CACHE_DIR, f"{tag}_tokens_{h}.pt"),
        os.path.join(CACHE_DIR, f"{tag}_meta_{h}.pt"),
    )


def _purge_stale(tag: str):
    import glob
    for kind in ["ann", "graphs", "tokens", "meta"]:
        for f in glob.glob(os.path.join(CACHE_DIR, f"{tag}_{kind}_*.pt")):
            if CONFIG_HASH not in f:
                os.remove(f)
                print(f"  Deleted stale cache: {os.path.basename(f)}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PRE-COMPUTATION
# BUG FIX: g3d_builder parameter added so graphs get .pos (required by EGNN).
# Without this, EGNNBranch.forward() crashes on `x = data.pos` (AttributeError).
# ══════════════════════════════════════════════════════════════════════════════

def precompute_features(smiles_list, labels_list, weights_list, tag,
                        ann_extractor, graph_builder, tokenizer,
                        g3d_builder=None):
    """
    Pre-compute and cache all model inputs for a split.

    Args:
        g3d_builder : Graph3DBuilder instance.  Must be provided for SynFeasNetV2
                      so that each graph has a .pos attribute for the EGNN branch.
                      If None, graphs are built without 3D coords (v1 compat mode).
    """
    ann_p, graph_p, token_p, meta_p = _cache_paths(tag)
    _purge_stale(tag)

    if all(os.path.exists(p) for p in [ann_p, graph_p, token_p, meta_p]):
        print(f"  [{tag}] Loading from cache...")
        ann_feats = _safe_load(ann_p)
        graphs    = _safe_load(graph_p)
        tok_data  = _safe_load(token_p)
        meta_data = _safe_load(meta_p)
        print(f"  [{tag}] {len(ann_feats)} molecules loaded")
        return {"ann_feats": ann_feats, "graphs": graphs, **tok_data, **meta_data}

    print(f"  [{tag}] Computing features for {len(smiles_list)} molecules...")
    if g3d_builder is None:
        print(f"  [{tag}] WARNING: g3d_builder not provided — graphs will lack .pos "
              "and EGNN branch will use zero coordinates.")

    ann_feats, graphs = [], []
    all_ids, all_mask = [], []
    kept_labels       = []
    kept_weights      = []
    kept_smiles       = []
    skipped = 0
    t0 = time.time()

    for i, (smi, lbl, wt) in enumerate(zip(smiles_list, labels_list, weights_list)):
        try:
            ann_feat = torch.tensor(ann_extractor.compute(smi), dtype=torch.float32)

            # Build 2D graph
            graph = graph_builder.build(smi)
            assert graph.x.shape[1] == GATBranch.NODE_DIM, \
                f"Node dim mismatch: got {graph.x.shape[1]}, expected {GATBranch.NODE_DIM}"

            # BUG FIX: Add 3D coordinates for EGNNBranch.
            # graph.pos is required by EGNNBranch.forward(): x = data.pos
            if g3d_builder is not None:
                graph = g3d_builder.add_coords(graph, smi)
            else:
                # Fallback: zero pos so EGNN degrades gracefully
                n_atoms = graph.x.size(0)
                graph.pos = torch.zeros((n_atoms, 3), dtype=torch.float32)

            tok  = tokenizer(smi)
            ids  = tok["input_ids"].squeeze(0)
            mask = tok["attention_mask"].squeeze(0)

            ann_feats.append(ann_feat)
            graphs.append(graph)
            all_ids.append(ids)
            all_mask.append(mask)
            kept_labels.append(float(lbl))
            kept_weights.append(float(wt))
            kept_smiles.append(smi)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"    Skip [{i}]: {type(e).__name__}: {e}")

        if (i + 1) % 500 == 0 or (i + 1) == len(smiles_list):
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-8)
            eta     = (len(smiles_list) - i - 1) / max(rate, 1e-8)
            print(f"    {i+1}/{len(smiles_list)} | {rate:.0f} mol/s | "
                  f"ETA {eta/60:.1f} min")

    if not ann_feats:
        raise RuntimeError(f"[{tag}] No valid molecules processed.")

    ids_t  = torch.stack(all_ids)
    mask_t = torch.stack(all_mask)

    torch.save(ann_feats,                                        ann_p)
    torch.save(graphs,                                           graph_p)
    torch.save({"input_ids": ids_t, "attention_masks": mask_t}, token_p)
    torch.save({"labels": kept_labels, "weights": kept_weights,
                "smiles": kept_smiles},                          meta_p)

    print(f"  [{tag}] Done in {(time.time()-t0)/60:.1f} min. Skipped {skipped}.")
    return {
        "ann_feats":       ann_feats,
        "graphs":          graphs,
        "input_ids":       ids_t,
        "attention_masks": mask_t,
        "labels":          kept_labels,
        "weights":         kept_weights,
        "smiles":          kept_smiles,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATASET + COLLATE
# ══════════════════════════════════════════════════════════════════════════════

class SynFeasDataset(Dataset):
    def __init__(self, cache: dict):
        self.ann_feats  = cache["ann_feats"]
        self.graphs     = cache["graphs"]
        self.input_ids  = cache["input_ids"]
        self.attn_masks = cache["attention_masks"]
        self.labels     = torch.tensor(cache["labels"],  dtype=torch.float32)
        self.weights    = torch.tensor(cache["weights"], dtype=torch.float32)
        self.smiles     = cache["smiles"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.ann_feats[idx],
            self.graphs[idx],
            self.input_ids[idx],
            self.attn_masks[idx],
            self.labels[idx],
            self.weights[idx],
            self.smiles[idx],
        )


def collate_fn(batch):
    ann, graphs, ids, masks, labels, weights, smiles = zip(*batch)
    return (
        torch.stack(ann),
        Batch.from_data_list(graphs),
        torch.stack(ids),
        torch.stack(masks),
        torch.stack(labels),
        torch.stack(weights),
        list(smiles),
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOSS  — weighted focal loss
# ══════════════════════════════════════════════════════════════════════════════

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,
                logits:         torch.Tensor,
                targets:        torch.Tensor,
                sample_weights: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        prob    = torch.sigmoid(logits)
        ce      = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal   = alpha_t * ((1.0 - p_t) ** self.gamma) * ce
        return (focal * sample_weights).mean()


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# NOTE: No local SynFeasNet or FusionHead defined here.
#       We import and train SynFeasNetV2 directly.
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    model.train()
    total_loss = 0.0
    amp_enabled = device.type == "cuda"
    for ann_x, graphs, ids, masks, labels, weights, _ in loader:
        ann_x   = ann_x.to(device,   non_blocking=True)
        graphs  = graphs.to(device)
        ids     = ids.to(device,     non_blocking=True)
        masks   = masks.to(device,   non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(amp_enabled):
            logits = model(ann_x, graphs, ids, masks).squeeze(-1)
            loss   = criterion(logits, labels, weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, threshold: float = 0.5) -> dict:
    model.eval()
    all_probs, all_labels, all_smiles = [], [], []
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for ann_x, graphs, ids, masks, labels, _w, smiles in loader:
            ann_x  = ann_x.to(device,  non_blocking=True)
            graphs = graphs.to(device)
            ids    = ids.to(device,    non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with _autocast(amp_enabled):
                logits = model(ann_x, graphs, ids, masks).squeeze(-1)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(np.atleast_1d(probs).tolist())
            all_labels.extend(labels.numpy().tolist())
            all_smiles.extend(smiles)

    y     = np.array(all_labels)
    p     = np.array(all_probs)
    preds = (p >= threshold).astype(float)
    uniq  = len(np.unique(y))

    return {
        "acc":        (preds == y).mean(),
        "roc_auc":    roc_auc_score(y, p)           if uniq > 1 else 0.5,
        "pr_auc":     average_precision_score(y, p) if uniq > 1 else 0.0,
        "f1":         f1_score(y, preds,       zero_division=0),
        "precision":  precision_score(y, preds, zero_division=0),
        "recall":     recall_score(y, preds,    zero_division=0),
        "all_probs":  p,
        "all_preds":  preds,
        "all_labels": y,
        "all_smiles": all_smiles,
    }


def find_best_threshold(labels, probs):
    pre, rec, thr = precision_recall_curve(labels, probs)
    if len(thr) == 0:
        return 0.5, 0.0
    f1s  = 2 * pre[:-1] * rec[:-1] / (pre[:-1] + rec[:-1] + 1e-8)
    best = int(np.argmax(f1s))
    return float(thr[best]), float(f1s[best])


def audit_false_negatives(results, n: int = 15):
    y   = results["all_labels"]
    p   = results["all_preds"]
    prb = results["all_probs"]
    smi = np.array(results["all_smiles"])
    fn  = (y == 1) & (p == 0)
    fp  = (y == 0) & (p == 1)
    print(f"\n{'─'*60}")
    print(f"AUDIT  False Negatives={fn.sum()}  False Positives={fp.sum()}")
    if fn.sum() > 0:
        print(f"Top {min(n, fn.sum())} False Negatives:")
        fn_prb = prb[fn]; fn_smi = smi[fn]
        for i in np.argsort(fn_prb)[::-1][:n]:
            print(f"  p={fn_prb[i]:.3f}  {fn_smi[i][:100]}")


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_xgboost_baseline(tr_df, va_df, te_df, ann_extractor):
    try:
        import xgboost as xgb
    except ImportError:
        print("[Baseline] xgboost not installed — skipping.")
        return

    print("\n" + "═" * 60)
    print("XGBoost Baseline  (ECFP4 + Descriptors)")
    print("═" * 60)

    X_tr = ann_extractor.compute_batch(tr_df["smiles_canonical"].tolist())
    y_tr = tr_df["label_final"].values
    w_tr = tr_df["label_confidence"].map(
        {"high": WEIGHT_HIGH, "medium": WEIGHT_MEDIUM}
    ).fillna(WEIGHT_MEDIUM).values

    X_va = ann_extractor.compute_batch(va_df["smiles_canonical"].tolist())
    y_va = va_df["label_final"].values

    X_te = ann_extractor.compute_batch(te_df["smiles_canonical"].tolist())
    y_te = te_df["label_final"].values

    scale_pw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)

    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

    va_probs   = clf.predict_proba(X_va)[:, 1]
    va_thr, _  = find_best_threshold(y_va, va_probs)
    te_probs   = clf.predict_proba(X_te)[:, 1]
    te_preds   = (te_probs >= va_thr).astype(int)

    print(f"  Val threshold : {va_thr:.4f}")
    print(f"  Test ROC-AUC  : {roc_auc_score(y_te, te_probs):.4f}")
    print(f"  Test PR-AUC   : {average_precision_score(y_te, te_probs):.4f}")
    print(f"  Test F1       : {f1_score(y_te, te_preds, zero_division=0):.4f}")
    print(f"  Test Precision: {precision_score(y_te, te_preds, zero_division=0):.4f}")
    print(f"  Test Recall   : {recall_score(y_te, te_preds, zero_division=0):.4f}")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, path):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    a1.plot(history["loss"], color="steelblue"); a1.set_title("Train Loss")
    a2.plot(history["val_roc_auc"], label="ROC-AUC", color="green")
    a2.plot(history["val_pr_auc"],  label="PR-AUC",  color="orange")
    a2.set_title("Val AUC"); a2.legend()
    a3.plot(history["val_f1"],        label="F1",        color="purple")
    a3.plot(history["val_recall"],    label="Recall",    color="red")
    a3.plot(history["val_precision"], label="Precision", color="blue")
    a3.set_title("Val Metrics"); a3.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_pr_curve(labels, probs, path):
    pre, rec, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    plt.figure(figsize=(7, 5))
    plt.plot(rec, pre, color="steelblue", lw=2, label=f"PR-AUC={pr_auc:.4f}")
    plt.axhline(labels.mean(), color="gray", linestyle="--", label="Random baseline")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve — Test Set")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(labels, preds, path, thr):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: Not Synth", "Pred: Synth"],
                yticklabels=["True: Not Synth", "True: Synth"])
    plt.title(f"Confusion Matrix (thr={thr:.3f})")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load CSV ──────────────────────────────────────────────────────────
    print(f"\nLoading {CSV_PATH} ...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            f"Expected at: {_HYBRID_CSV}\n  or: {_RAW_CSV}\n"
            "Place dataset.csv in data/processed/ and re-run."
        )

    df = pd.read_csv(CSV_PATH, low_memory=False)

    for required_col in ("smiles_canonical", "split"):
        if required_col not in df.columns:
            raise ValueError(
                f"Required column '{required_col}' missing from {CSV_PATH}.\n"
                f"Available columns: {list(df.columns)}"
            )

    df["smiles_canonical"] = df["smiles_canonical"].astype(str)
    df["split"]            = df["split"].astype(str).str.lower().str.strip()

    if "label_final" not in df.columns:
        print(
            "\n⚠  'label_final' column not found in CSV.\n"
            "   Deriving label_final + label_confidence via chemistry-first logic...\n"
        )
        df = _derive_labels(df)
    else:
        df["label_final"] = df["label_final"].astype(float)

    if "label_confidence" not in df.columns:
        df["label_confidence"] = "high"
    else:
        df["label_confidence"] = df["label_confidence"].astype(str).str.strip()

    df = df.dropna(subset=["smiles_canonical", "label_final", "split"])
    df = df[df["smiles_canonical"].str.strip() != ""]

    df["sample_weight"] = df["label_confidence"].map(
        {"high": WEIGHT_HIGH, "medium": WEIGHT_MEDIUM}
    ).fillna(WEIGHT_MEDIUM)

    print(f"\nTotal rows  : {len(df)}")
    print("label_final :\n", df["label_final"].value_counts())
    print("split dist  :\n", df["split"].value_counts())

    # ── Split ─────────────────────────────────────────────────────────────
    all_train = df[df["split"] == "train"].reset_index(drop=True)
    all_val   = df[df["split"] == "val"].reset_index(drop=True)
    all_test  = df[df["split"] == "test"].reset_index(drop=True)

    if len(all_val) == 0 or len(all_test) == 0:
        print("⚠  val/test splits empty — creating 80/10/10 split from all data.")
        from sklearn.model_selection import train_test_split
        all_data   = df[df["label_final"].isin([0.0, 1.0])].reset_index(drop=True)
        all_train, temp = train_test_split(all_data, test_size=0.20,
                                           random_state=42,
                                           stratify=all_data["label_final"])
        all_val, all_test = train_test_split(temp, test_size=0.50,
                                             random_state=42,
                                             stratify=temp["label_final"])
        all_train = all_train.reset_index(drop=True)
        all_val   = all_val.reset_index(drop=True)
        all_test  = all_test.reset_index(drop=True)

    train_df = all_train[all_train["label_final"].isin([0.0, 1.0])].reset_index(drop=True)
    val_df   = all_val[all_val["label_final"].isin([0.0, 1.0])].reset_index(drop=True)
    test_df  = all_test[all_test["label_final"].isin([0.0, 1.0])].reset_index(drop=True)

    pos = (train_df["label_final"] == 1).sum()
    neg = (train_df["label_final"] == 0).sum()
    print(f"\nTrain: {len(train_df):,} | Pos={pos:,} Neg={neg:,} ratio=1:{neg//max(pos,1):.0f}")
    print(f"Val  : {len(val_df):,}")
    print(f"Test : {len(test_df):,}")

    if len(train_df) == 0:
        raise RuntimeError("No training data found after filtering.")

    # ── Feature extractors ────────────────────────────────────────────────
    ann_extractor = ANNFeatureExtractor()
    graph_builder = GraphBuilder()
    tokenizer     = SMILESTokenizer(max_length=MAX_SMILES_LEN)

    # BUG FIX: Graph3DBuilder is now instantiated and passed to precompute_features.
    # Previously it was imported but never used — graphs had no .pos and EGNN crashed.
    g3d_builder = Graph3DBuilder()
    print("\nGraph3DBuilder ready — will generate ETKDG+MMFF 3D coordinates.")

    scaler_path = os.path.join(CACHE_DIR, f"descriptor_scaler_{CONFIG_HASH}.npz")
    if os.path.exists(scaler_path):
        ann_extractor.load_descriptor_scaler(scaler_path)
    else:
        print("\nFitting descriptor scaler on training SMILES...")
        ann_extractor.fit_descriptors(train_df["smiles_canonical"].tolist())
        ann_extractor.save_descriptor_scaler(scaler_path)

    # ── Pre-compute features ──────────────────────────────────────────────
    # NOTE: 3D coordinate generation is slow (~1-3 sec/molecule for macrocycles).
    # With ~8,000 training molecules, expect 30-90 min on first run.
    # Subsequent runs use cache and complete in seconds.
    print("\nPre-computing features (3D coord generation may take time on first run)...")
    ones    = [WEIGHT_HIGH] * len(val_df)
    ones_te = [WEIGHT_HIGH] * len(test_df)

    train_cache = precompute_features(
        train_df["smiles_canonical"].tolist(),
        train_df["label_final"].tolist(),
        train_df["sample_weight"].tolist(),
        "train", ann_extractor, graph_builder, tokenizer, g3d_builder,
    )
    val_cache = precompute_features(
        val_df["smiles_canonical"].tolist(),
        val_df["label_final"].tolist(),
        ones,
        "val", ann_extractor, graph_builder, tokenizer, g3d_builder,
    )
    test_cache = precompute_features(
        test_df["smiles_canonical"].tolist(),
        test_df["label_final"].tolist(),
        ones_te,
        "test", ann_extractor, graph_builder, tokenizer, g3d_builder,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    pin = device.type == "cuda"
    train_loader = DataLoader(SynFeasDataset(train_cache), batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=0, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(SynFeasDataset(val_cache),   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(SynFeasDataset(test_cache),  batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0, pin_memory=pin)

    # ── Loss / optimiser ──────────────────────────────────────────────────
    train_labels = np.array(train_cache["labels"], dtype=np.float32)
    pos_frac     = float(train_labels.mean())
    focal_alpha  = float(np.clip(1.0 - pos_frac, 0.5, 0.95))
    print(f"\nPos fraction={pos_frac:.3f}  focal_alpha={focal_alpha:.4f}")

    # BUG FIX: model is now SynFeasNetV2 (attention fusion + EGNN branch).
    # The old code was using a locally-defined SynFeasNet (v1 concatenation model)
    # which ignored both the EGNN branch and the attention fusion module entirely.
    print("\nInstantiating SynFeasNetV2 (4-branch attention fusion)...")
    model     = SynFeasNetV2(dropout=0.3).to(device)
    params    = model.count_parameters()
    print(f"  Total params    : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")

    criterion = WeightedFocalLoss(alpha=focal_alpha, gamma=FOCAL_GAMMA)

    # Separate LR for ChemBERTa (LoRA) vs rest — transformer fine-tuning needs
    # a lower LR to avoid catastrophic forgetting.
    optimizer = torch.optim.AdamW([
        {"params": model.chemberta_branch.parameters(), "lr": LR * 0.1},
        {"params": list(model.ann_branch.parameters()) +
                   list(model.gat_branch.parameters()) +
                   list(model.egnn_branch.parameters() if model.egnn_branch else []) +
                   list(model.fusion.parameters()) +
                   list(model.output_head.parameters()),
         "lr": LR},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR * 0.1, LR],
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )
    amp_scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────
    best_pr_auc = 0.0
    best_thresh = 0.5
    history = {k: [] for k in ["loss", "val_roc_auc", "val_pr_auc",
                                "val_f1", "val_precision", "val_recall"]}
    ckpt_path = os.path.join(SAVE_DIR, "best_synfeasnet_hybrid.pth")

    print(f"\nTraining SynFeasNetV2 for {EPOCHS} epochs")
    print("-" * 65)

    for epoch in range(EPOCHS):
        t0   = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion,
                           amp_scaler, scheduler)

        v05     = evaluate(model, val_loader, threshold=0.5)
        thr, _  = find_best_threshold(v05["all_labels"], v05["all_probs"])
        val     = evaluate(model, val_loader, threshold=thr)

        # Log modality importance once per epoch
        mw = model.fusion.get_modality_weights()
        mw_str = " ".join(f"{k}={v:.2f}" for k, v in mw.items())

        history["loss"].append(loss)
        history["val_roc_auc"].append(val["roc_auc"])
        history["val_pr_auc"].append(val["pr_auc"])
        history["val_f1"].append(val["f1"])
        history["val_precision"].append(val["precision"])
        history["val_recall"].append(val["recall"])

        print(f"Ep {epoch+1:02d}/{EPOCHS} | Loss={loss:.4f} | "
              f"ROC={val['roc_auc']:.4f} | PR={val['pr_auc']:.4f} | "
              f"F1={val['f1']:.4f} | Thr={thr:.3f} | {time.time()-t0:.0f}s")
        print(f"  Modality weights: [{mw_str}]")

        if val["pr_auc"] > best_pr_auc:
            best_pr_auc = val["pr_auc"]
            best_thresh = thr
            torch.save({
                "epoch":       epoch + 1,
                "state_dict":  model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_roc_auc": val["roc_auc"],
                "val_pr_auc":  val["pr_auc"],
                "val_f1":      val["f1"],
                "threshold":   thr,
                "config": {
                    "max_smiles_len":  MAX_SMILES_LEN,
                    "node_dim":        GATBranch.NODE_DIM,
                    "cache_hash":      CONFIG_HASH,
                    "label_version":   "hybrid_v3_egnn",
                    "architecture":    "SynFeasNetV2",
                },
            }, ckpt_path)
            print(f"  ✓ Saved (PR-AUC={best_pr_auc:.4f}, thr={best_thresh:.3f})")

    print("-" * 65)
    print(f"Training done. Best Val PR-AUC={best_pr_auc:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("FINAL TEST SET EVALUATION")
    print("═" * 65)

    ckpt      = _safe_load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    final_thr = ckpt["threshold"]
    arch      = ckpt["config"].get("architecture", "SynFeasNetV2")
    print(f"Loaded epoch {ckpt['epoch']} | arch={arch} | threshold={final_thr:.4f}")

    test_res = evaluate(model, test_loader, threshold=final_thr)
    print(f"ROC-AUC  : {test_res['roc_auc']:.4f}")
    print(f"PR-AUC   : {test_res['pr_auc']:.4f}")
    print(f"F1       : {test_res['f1']:.4f}")
    print(f"Precision: {test_res['precision']:.4f}")
    print(f"Recall   : {test_res['recall']:.4f}")
    print(f"Accuracy : {test_res['acc']:.4f}")
    print()
    print(classification_report(
        test_res["all_labels"], test_res["all_preds"],
        target_names=["Not Synthesizable", "Synthesizable"],
        zero_division=0,
    ))

    audit_false_negatives(test_res, n=15)

    plot_training_curves(history, os.path.join(SAVE_DIR, "curves_hybrid.png"))
    plot_pr_curve(test_res["all_labels"], test_res["all_probs"],
                  os.path.join(SAVE_DIR, "pr_curve_hybrid.png"))
    plot_confusion(test_res["all_labels"], test_res["all_preds"],
                   os.path.join(SAVE_DIR, "confusion_hybrid.png"), final_thr)

    if RUN_BASELINE:
        run_xgboost_baseline(train_df, val_df, test_df, ann_extractor)

    print(f"\nOutputs saved to: {SAVE_DIR}")
    print("Done.")