"""
train.py — SynPractIQ v3
===========================
Multi-task training pipeline for the Synthetic Practicality Index (SPI).

WHAT CHANGED vs v2 (SynFeasNet binary classifier):
  1. Labels: 6 SPI sub-scores + composite SPI + Stage-1 gate
             (derived on-the-fly from rich dataset.csv columns via SPILabelGenerator)
  2. Loss:   SPIMultiTaskLoss — combines:
               • MSE on each sub-score (regression)
               • MSE on composite SPI
               • BCE on Stage-1 gate (binary: realistic vs not)
               • Entropy regularization on fusion gate
  3. Model:  SynPractIQModel (SynFeasNetV2 alias, SPIOutputHead replaces OutputHead)
  4. Metrics: Per-dimension MAE, SPI-score MAE, Stage-1 ROC-AUC, Spearman ρ
  5. Dataset: uses 'smiles' column (not 'smiles_canonical') to match dataset.csv
  6. Checkpoint saves all SPI dimension stats for introspection

Run:  python training/train.py
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
from scipy.stats import spearmanr

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, mean_absolute_error
from torch_geometric.data import Batch

# ── torch.amp backward-compatibility ────────────────────────────────────────
try:
    from torch.amp import GradScaler
    def _autocast(enabled: bool):
        return torch.amp.autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler
    from torch.cuda.amp import autocast as _cuda_autocast
    def _autocast(enabled: bool):
        return _cuda_autocast(enabled=enabled)

# ── Project root auto-detection ───────────────────────────────────────────────
_THIS_FILE   = os.path.abspath(__file__)
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if not os.path.isdir(os.path.join(PROJECT_ROOT, "models")):
    PROJECT_ROOT = _SCRIPTS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Imports ───────────────────────────────────────────────────────────────────
from models.ann_branch       import ANNBranch, ANNFeatureExtractor
from models.gat_branch       import GATBranch, GraphBuilder
from models.chemBERTa_branch import ChemBERTaBranch, SMILESTokenizer
from models.attention_fusion import SynPractIQModel, SPI_DIMENSION_NAMES
from models.egnn_branch      import Graph3DBuilder
from models.calibration      import TemperatureScaling
from spi_labels              import SPILabelGenerator, FEASIBILITY_GATE_THRESHOLD

# ── Paths ─────────────────────────────────────────────────────────────────────
_RAW_CSV  = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.csv")
CSV_PATH  = _RAW_CSV
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache_spi")
SAVE_DIR  = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,  exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE     = 16
EPOCHS         = 0
LR             = 2e-4
MAX_SMILES_LEN = 320
ENTROPY_REG    = 0.05

# Loss weights for multi-task objective
LOSS_WEIGHTS = {
    "sub_scores": 0.40,   # sum of 6 sub-score MSE losses
    "spi_score":  0.40,   # composite SPI MSE
    "stage1":     0.20,   # feasibility gate BCE
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
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
        "version":      "synpractiq_v3_spi",
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
# ══════════════════════════════════════════════════════════════════════════════

def precompute_features(df_split: pd.DataFrame, tag: str,
                        ann_extractor, graph_builder,
                        tokenizer, g3d_builder):
    """
    Pre-compute and cache all model inputs + SPI labels for one split.

    Labels stored in meta:
      spi_sub_scores  : (N, 6)  — one float per SPI dimension
      spi_score       : (N,)    — composite SPI
      stage1_label    : (N,)    — binary feasibility gate
      spi_class       : (N,)    — ordinal class 0-4
      sample_weights  : (N,)
      smiles          : list[str]
    """
    ann_p, graph_p, token_p, meta_p = _cache_paths(tag)
    _purge_stale(tag)

    if all(os.path.exists(p) for p in [ann_p, graph_p, token_p, meta_p]):
        print(f"  [{tag}] Loading from cache...")
        ann_feats = _safe_load(ann_p)
        graphs    = _safe_load(graph_p)
        tok_data  = _safe_load(token_p)
        meta_data = _safe_load(meta_p)
        print(f"  [{tag}] {len(ann_feats)} molecules loaded from cache")
        return {"ann_feats": ann_feats, "graphs": graphs, **tok_data, **meta_data}

    smiles_col = "smiles" if "smiles" in df_split.columns else "smiles_canonical"
    smiles_list = df_split[smiles_col].astype(str).tolist()

    # SPI label columns required
    sub_score_cols = [f"spi_{d}" for d in SPI_DIMENSION_NAMES]
    spi_col     = "spi_score"
    gate_col    = "stage1_pass"
    weight_col  = "spi_sample_weight"
    class_col   = "spi_class"

    print(f"  [{tag}] Computing features for {len(smiles_list)} molecules...")

    ann_feats, graphs_out = [], []
    all_ids, all_mask     = [], []
    sub_scores_list       = []
    spi_score_list        = []
    stage1_list           = []
    class_list            = []
    weight_list           = []
    kept_smiles           = []
    skipped = 0
    t0 = time.time()

    for i, (_, row) in enumerate(df_split.iterrows()):
        smi = str(row[smiles_col])
        try:
            ann_feat = torch.tensor(ann_extractor.compute(smi), dtype=torch.float32)

            graph = graph_builder.build(smi)
            assert graph.x.shape[1] == GATBranch.NODE_DIM

            graph = g3d_builder.add_coords(graph, smi)

            tok  = tokenizer(smi)
            ids  = tok["input_ids"].squeeze(0)
            mask = tok["attention_mask"].squeeze(0)

            # SPI targets
            sub_scores = [float(row.get(c, 0.5)) for c in sub_score_cols]
            spi_score  = float(row.get(spi_col, 0.5))
            stage1     = float(bool(row.get(gate_col, True)))
            spi_class  = int(row.get(class_col, 2))
            weight     = float(row.get(weight_col, 0.5))

            # Clamp targets to [0, 1]
            sub_scores = [max(0.0, min(1.0, s)) for s in sub_scores]
            spi_score  = max(0.0, min(1.0, spi_score))

            ann_feats.append(ann_feat)
            graphs_out.append(graph)
            all_ids.append(ids)
            all_mask.append(mask)
            sub_scores_list.append(sub_scores)
            spi_score_list.append(spi_score)
            stage1_list.append(stage1)
            class_list.append(spi_class)
            weight_list.append(weight)
            kept_smiles.append(smi)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"    Skip [{i}]: {type(e).__name__}: {e}")

        if (i + 1) % 1000 == 0 or (i + 1) == len(smiles_list):
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-8)
            eta     = (len(smiles_list) - i - 1) / max(rate, 1e-8)
            print(f"    {i+1}/{len(smiles_list)} | {rate:.0f} mol/s | ETA {eta/60:.1f} min")

    if not ann_feats:
        raise RuntimeError(f"[{tag}] No valid molecules processed.")

    ids_t  = torch.stack(all_ids)
    mask_t = torch.stack(all_mask)

    meta = {
        "sub_scores":    sub_scores_list,
        "spi_score":     spi_score_list,
        "stage1":        stage1_list,
        "spi_class":     class_list,
        "sample_weights": weight_list,
        "smiles":        kept_smiles,
    }

    torch.save(ann_feats,                                        ann_p)
    torch.save(graphs_out,                                       graph_p)
    torch.save({"input_ids": ids_t, "attention_masks": mask_t}, token_p)
    torch.save(meta,                                             meta_p)

    print(f"  [{tag}] Done in {(time.time()-t0)/60:.1f} min. Skipped={skipped}")
    return {
        "ann_feats":       ann_feats,
        "graphs":          graphs_out,
        "input_ids":       ids_t,
        "attention_masks": mask_t,
        **meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATASET + COLLATE
# ══════════════════════════════════════════════════════════════════════════════

class SPIDataset(Dataset):
    def __init__(self, cache: dict):
        self.ann_feats   = cache["ann_feats"]
        self.graphs      = cache["graphs"]
        self.input_ids   = cache["input_ids"]
        self.attn_masks  = cache["attention_masks"]
        self.sub_scores  = torch.tensor(cache["sub_scores"],     dtype=torch.float32)
        self.spi_score   = torch.tensor(cache["spi_score"],      dtype=torch.float32)
        self.stage1      = torch.tensor(cache["stage1"],         dtype=torch.float32)
        self.spi_class   = torch.tensor(cache["spi_class"],      dtype=torch.long)
        self.weights     = torch.tensor(cache["sample_weights"], dtype=torch.float32)
        self.smiles      = cache["smiles"]

    def __len__(self):
        return len(self.spi_score)

    def __getitem__(self, idx):
        return (
            self.ann_feats[idx],
            self.graphs[idx],
            self.input_ids[idx],
            self.attn_masks[idx],
            self.sub_scores[idx],    # (6,)
            self.spi_score[idx],     # scalar
            self.stage1[idx],        # scalar
            self.spi_class[idx],     # scalar int
            self.weights[idx],       # scalar
            self.smiles[idx],
        )


def collate_fn(batch):
    (ann, graphs, ids, masks,
     sub_scores, spi, stage1, spi_class, weights, smiles) = zip(*batch)
    return (
        torch.stack(ann),
        Batch.from_data_list(graphs),
        torch.stack(ids),
        torch.stack(masks),
        torch.stack(sub_scores),     # (B, 6)
        torch.stack(spi),            # (B,)
        torch.stack(stage1),         # (B,)
        torch.stack(spi_class),      # (B,)
        torch.stack(weights),        # (B,)
        list(smiles),
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOSS — SPI Multi-Task
# ══════════════════════════════════════════════════════════════════════════════

class SPIMultiTaskLoss(nn.Module):
    """
    Combined loss for SPI multi-task regression + gate classification.

    L = w_sub  * mean(MSE_i for i in 6 dimensions)
      + w_spi  * MSE(composite SPI)
      + w_gate * BCE(stage1_logit, stage1_label)

    All terms are sample-weighted.
    """

    def __init__(self, weights: dict = None):
        super().__init__()
        w = weights or LOSS_WEIGHTS
        self.w_sub   = w["sub_scores"]
        self.w_spi   = w["spi_score"]
        self.w_gate  = w["stage1"]

    def forward(self, output: dict, sub_targets: torch.Tensor,
                    spi_targets: torch.Tensor, gate_targets: torch.Tensor,
                    sample_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output         : dict from SynPractIQModel.forward()
            sub_targets    : (B, 6)
            spi_targets    : (B,)
            gate_targets   : (B,)  binary
            sample_weights : (B,)
        """
        sw = sample_weights.unsqueeze(-1)  # (B, 1)

        # Sub-score regression (MSE per dimension, then average)
        sub_pred = output["sub_scores"]   # (B, 6)
        sub_mse  = ((sub_pred - sub_targets) ** 2 * sw).mean()
        
        # Stability: Replace NaN with 0.0 to prevent training collapse
        if torch.isnan(sub_mse):
            sub_mse = torch.tensor(0.0, device=sub_mse.device)

        # Composite SPI regression
        spi_pred = output["spi_score"].squeeze(-1)   # (B,)
        spi_mse  = ((spi_pred - spi_targets) ** 2 * sample_weights).mean()
        
        if torch.isnan(spi_mse):
            spi_mse = torch.tensor(0.0, device=spi_mse.device)

        # Stage 1 gate classification
        gate_logit = output["stage1_logit"].squeeze(-1)  # (B,)
        gate_bce   = F.binary_cross_entropy_with_logits(
            gate_logit, gate_targets, weight=sample_weights, reduction="mean"
        )
        
        if torch.isnan(gate_bce):
            gate_bce = torch.tensor(0.0, device=gate_bce.device)

        total = self.w_sub * sub_mse + self.w_spi * spi_mse + self.w_gate * gate_bce
        return total



# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    model.train()
    total_loss = 0.0
    amp_enabled = device.type == "cuda"

    for (ann_x, graphs, ids, masks,
         sub_tgt, spi_tgt, gate_tgt, _, weights, _) in loader:

        ann_x   = ann_x.to(device,    non_blocking=True)
        graphs  = graphs.to(device)
        ids     = ids.to(device,      non_blocking=True)
        masks   = masks.to(device,    non_blocking=True)
        sub_tgt = sub_tgt.to(device,  non_blocking=True)
        spi_tgt = spi_tgt.to(device,  non_blocking=True)
        gate_tgt= gate_tgt.to(device, non_blocking=True)
        weights = weights.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(amp_enabled):
            out  = model(ann_x, graphs, ids, masks)
            loss = criterion(out, sub_tgt, spi_tgt, gate_tgt, weights)
            
            # Entropy regularization: 
            # model.fusion.entropy_loss() returns -entropy.
            # We want to MAXIMIZE entropy to prevent modality collapse.
            # To maximize entropy, we minimize -entropy.
            # We keep the positive sign here because the function already returns -entropy.
            loss = loss + ENTROPY_REG * model.fusion.entropy_loss()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model, loader) -> dict:
    """
    Returns per-dimension MAE, composite SPI MAE, Stage-1 ROC-AUC,
    Spearman ρ for SPI, and full arrays for plotting.
    """
    model.eval()
    all_sub_pred,  all_sub_true  = [], []
    all_spi_pred,  all_spi_true  = [], []
    all_gate_prob, all_gate_true = [], []
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for (ann_x, graphs, ids, masks,
             sub_tgt, spi_tgt, gate_tgt, _, _, _) in loader:

            ann_x   = ann_x.to(device)
            graphs  = graphs.to(device)
            ids     = ids.to(device)
            masks   = masks.to(device)

            with _autocast(amp_enabled):
                out = model(ann_x, graphs, ids, masks)

            all_sub_pred.append(out["sub_scores"].cpu().numpy())
            all_sub_true.append(sub_tgt.numpy())
            all_spi_pred.append(out["spi_score"].squeeze(-1).cpu().numpy())
            all_spi_true.append(spi_tgt.numpy())
            all_gate_prob.append(
                torch.sigmoid(out["stage1_logit"]).squeeze(-1).cpu().numpy()
            )
            all_gate_true.append(gate_tgt.numpy())

    sub_pred  = np.concatenate(all_sub_pred,  axis=0)   # (N, 6)
    sub_true  = np.concatenate(all_sub_true,  axis=0)   # (N, 6)
    spi_pred  = np.concatenate(all_spi_pred)
    spi_true  = np.concatenate(all_spi_true)
    gate_prob = np.concatenate(all_gate_prob)
    gate_true = np.concatenate(all_gate_true)

    # Per-dimension MAE
    per_dim_mae = {
        SPI_DIMENSION_NAMES[i]: float(mean_absolute_error(sub_true[:, i], sub_pred[:, i]))
        for i in range(len(SPI_DIMENSION_NAMES))
    }

    spi_mae   = float(mean_absolute_error(spi_true, spi_pred))
    spearman  = float(spearmanr(spi_true, spi_pred).statistic)

    n_unique_gate = len(np.unique(gate_true))
    gate_auc = (float(roc_auc_score(gate_true, gate_prob))
                if n_unique_gate > 1 else 0.5)

    return {
        "per_dim_mae": per_dim_mae,
        "mean_sub_mae": float(np.mean(list(per_dim_mae.values()))),
        "spi_mae":     spi_mae,
        "spearman":    spearman,
        "gate_auc":    gate_auc,
        "spi_pred":    spi_pred,
        "spi_true":    spi_true,
        "gate_prob":   gate_prob,
        "gate_true":   gate_true,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["loss"], color="steelblue")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history["spi_mae"], label="SPI MAE", color="green")
    axes[1].plot(history["mean_sub_mae"], label="Avg Sub MAE", color="orange")
    axes[1].set_title("Validation MAE"); axes[1].legend()
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["spearman"], label="Spearman ρ", color="purple")
    axes[2].plot(history["gate_auc"], label="Gate ROC-AUC", color="red")
    axes[2].set_title("Val Correlation / Gate AUC"); axes[2].legend()
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_spi_scatter(spi_true, spi_pred, path, split="val"):
    plt.figure(figsize=(6, 6))
    plt.scatter(spi_true, spi_pred, alpha=0.3, s=5, color="steelblue")
    mn, mx = min(spi_true.min(), spi_pred.min()), max(spi_true.max(), spi_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    rho = float(spearmanr(spi_true, spi_pred).statistic)
    mae = float(mean_absolute_error(spi_true, spi_pred))
    plt.title(f"SPI Predicted vs True ({split}) — ρ={rho:.3f}, MAE={mae:.4f}")
    plt.xlabel("SPI True")
    plt.ylabel("SPI Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_dim_maes(per_dim_mae: dict, path):
    dims = list(per_dim_mae.keys())
    maes = [per_dim_mae[d] for d in dims]
    plt.figure(figsize=(10, 4))
    plt.bar(dims, maes, color="steelblue")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE")
    plt.title("Per-Dimension SPI MAE (Validation)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Place dataset.csv in data/processed/ and re-run."
        )

    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Normalize SMILES column name
    if "smiles" not in df.columns and "smiles_canonical" in df.columns:
        df = df.rename(columns={"smiles_canonical": "smiles"})

    for col in ("smiles", "split"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing. Available: {list(df.columns)}")

    df["smiles"] = df["smiles"].astype(str)
    df["split"]  = df["split"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    print(f"Total rows: {len(df):,}")
    print("Split dist:\n", df["split"].value_counts())

    # ── Generate SPI labels ───────────────────────────────────────────────────
    spi_cols = [f"spi_{d}" for d in SPI_DIMENSION_NAMES]
    if not all(c in df.columns for c in spi_cols):
        print("\nGenerating SPI labels (first run — will be fast)...")
        gen = SPILabelGenerator()
        df  = gen.generate(df)
    else:
        print("SPI label columns already present — skipping generation.")

    # ── Splits ────────────────────────────────────────────────────────────────
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    if len(val_df) == 0 or len(test_df) == 0:
        print("⚠  Val/test splits empty — creating 80/10/10 split.")
        from sklearn.model_selection import train_test_split
        train_df, temp = train_test_split(df, test_size=0.20, random_state=42)
        val_df, test_df = train_test_split(temp, test_size=0.50, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

    print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # ── Feature extractors ────────────────────────────────────────────────────
    ann_extractor = ANNFeatureExtractor()
    graph_builder = GraphBuilder()
    tokenizer     = SMILESTokenizer(max_length=MAX_SMILES_LEN)
    g3d_builder   = Graph3DBuilder()

    scaler_path = os.path.join(CACHE_DIR, f"descriptor_scaler_{CONFIG_HASH}.npz")
    if os.path.exists(scaler_path):
        ann_extractor.load_descriptor_scaler(scaler_path)
    else:
        print("\nFitting descriptor scaler on training data...")
        ann_extractor.fit_descriptors(train_df["smiles"].tolist())
        ann_extractor.save_descriptor_scaler(scaler_path)

    # ── Pre-compute features ──────────────────────────────────────────────────
    print("\nPre-computing features (3D coords may be slow on first run)...")
    train_cache = precompute_features(
        train_df, "train", ann_extractor, graph_builder, tokenizer, g3d_builder
    )
    val_cache = precompute_features(
        val_df, "val", ann_extractor, graph_builder, tokenizer, g3d_builder
    )
    test_cache = precompute_features(
        test_df, "test", ann_extractor, graph_builder, tokenizer, g3d_builder
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    pin = device.type == "cuda"
    train_loader = DataLoader(
        SPIDataset(train_cache), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        SPIDataset(val_cache), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        SPIDataset(test_cache), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=pin,
    )

    # ── Model + optimizer ─────────────────────────────────────────────────────
    print("\nInstantiating SynPractIQModel...")
    model  = SynPractIQModel(dropout=0.3, modality_dropout_p=0.15).to(device)
    params = model.count_parameters()
    print(f"  Total params    : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")

    criterion = SPIMultiTaskLoss()

    # Lower LR for ChemBERTa (LoRA fine-tuning)
    optimizer = torch.optim.AdamW([
        {"params": model.chemberta_branch.parameters(), "lr": LR * 0.1},
        {"params": (
            list(model.ann_branch.parameters()) +
            list(model.gat_branch.parameters()) +
            list(model.egnn_branch.parameters() if model.egnn_branch else []) +
            list(model.fusion.parameters()) +
            list(model.output_head.parameters())
        ), "lr": LR},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR * 0.1, LR],
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )
    amp_scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_spi_mae = float("inf")
    ckpt_path    = os.path.join(SAVE_DIR, "best_synpractiq.pth")
    history      = {
        "loss": [], "spi_mae": [], "mean_sub_mae": [],
        "spearman": [], "gate_auc": [],
    }

    print(f"\nTraining SynPractIQ for {EPOCHS} epochs")
    print("-" * 70)

    for epoch in range(EPOCHS):
        t0   = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion,
                           amp_scaler, scheduler)
        val  = evaluate(model, val_loader)

        mw     = model.fusion.get_modality_weights()
        mw_str = " ".join(f"{k}={v:.2f}" for k, v in mw.items())
        
        # Fusion Collapse Detection
        for mod, weight in mw.items():
            if weight > 0.70:
                print(f"  ⚠ WARNING: Modality collapse detected! {mod} dominates ({weight:.2%})")
            elif weight < 0.05:
                print(f"  ⚠ WARNING: Modality underutilization! {mod} is nearly ignored ({weight:.2%})")

        history["loss"].append(loss)
        history["spi_mae"].append(val["spi_mae"])
        history["mean_sub_mae"].append(val["mean_sub_mae"])
        history["spearman"].append(val["spearman"])
        history["gate_auc"].append(val["gate_auc"])

        print(
            f"Ep {epoch+1:02d}/{EPOCHS} | Loss={loss:.4f} | "
            f"SPI-MAE={val['spi_mae']:.4f} | ρ={val['spearman']:.4f} | "
            f"Gate-AUC={val['gate_auc']:.4f} | {time.time()-t0:.0f}s"
        )
        print(f"  Modality weights: [{mw_str}]")
        for dim, mae in val["per_dim_mae"].items():
            print(f"    {dim:30s}: MAE={mae:.4f}")

        if val["spi_mae"] < best_spi_mae:
            best_spi_mae = val["spi_mae"]
            torch.save({
                "epoch":        epoch + 1,
                "state_dict":   model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_spi_mae":  val["spi_mae"],
                "val_spearman": val["spearman"],
                "val_gate_auc": val["gate_auc"],
                "val_per_dim_mae": val["per_dim_mae"],
                "config": {
                    "max_smiles_len": MAX_SMILES_LEN,
                    "node_dim":       GATBranch.NODE_DIM,
                    "cache_hash":     CONFIG_HASH,
                    "architecture":   "SynPractIQModel",
                    "version":        "v3_spi",
                },
            }, ckpt_path)
            print(f"  ✓ Saved best (SPI-MAE={best_spi_mae:.4f})")

    print("-" * 70)
    print(f"Training done. Best Val SPI-MAE={best_spi_mae:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    ckpt = _safe_load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    test_res = evaluate(model, test_loader)

    print(f"SPI MAE          : {test_res['spi_mae']:.4f}")
    print(f"Spearman ρ (SPI) : {test_res['spearman']:.4f}")
    print(f"Stage-1 Gate AUC : {test_res['gate_auc']:.4f}")
    print(f"Avg Sub-Score MAE: {test_res['mean_sub_mae']:.4f}")
    print("\nPer-Dimension MAE:")
    for dim, mae in test_res["per_dim_mae"].items():
        print(f"  {dim:30s}: {mae:.4f}")

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_training_curves(history, os.path.join(SAVE_DIR, "curves_spi.png"))
    plot_spi_scatter(
        test_res["spi_true"], test_res["spi_pred"],
        os.path.join(SAVE_DIR, "spi_scatter_test.png"), split="test"
    )
    plot_dim_maes(
        test_res["per_dim_mae"],
        os.path.join(SAVE_DIR, "dim_mae_test.png")
    )

    print(f"\nOutputs saved to: {SAVE_DIR}")
    print("Done.")