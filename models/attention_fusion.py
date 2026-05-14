"""
attention_fusion.py — SynPractIQ v3
======================================
Multi-output fusion model for the Synthetic Practicality Index (SPI).

ARCHITECTURE CHANGES vs v2:
  - OutputHead replaced by SPIOutputHead:
      * 6 sub-score heads (one per SPI dimension), each → [0,1] via Sigmoid
      * 1 composite SPI head → [0,1] via Sigmoid
      * 1 feasibility gate head → binary logit (Stage 1)
  - SynPractIQModel wraps the full pipeline with all 4 branches + fusion.
  - All modality-collapse fixes from v2 are preserved (entropy reg,
    modality dropout, learnable gate temperature).

Sub-score heads produce:
  spi_synthetic_complexity, spi_route_practicality,
  spi_precursor_availability, spi_scalability,
  spi_retro_confidence, spi_medchem_realism,
  spi_score (composite), stage1_logit (gate)
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from models.ann_branch       import ANNBranch
from models.gat_branch       import GATBranch
from models.chemBERTa_branch import ChemBERTaBranch

try:
    from models.egnn_branch import EGNNBranch, Graph3DBuilder
    _EGNN_AVAILABLE = True
except ImportError as _e:
    _EGNN_AVAILABLE = False
    print(f"[attention_fusion] ⚠ EGNNBranch not available: {_e}")
    Graph3DBuilder = None


# SPI dimension names (order must match SPIOutputHead and dataset labels)
SPI_DIMENSION_NAMES = [
    "synthetic_complexity",
    "route_practicality",
    "precursor_availability",
    "scalability",
    "retro_confidence",
    "medchem_realism",
]


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION FUSION  (unchanged from v2, collapse fixes preserved)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionFusion(nn.Module):
    """
    Attention-based fusion of N modality embeddings.
    Identical to v2 — all collapse-prevention fixes retained.
    """

    MODALITY_NAMES = ["ANN", "GAT", "ChemBERTa", "EGNN"]

    def __init__(
        self,
        embed_dim:          int   = 256,
        num_modalities:     int   = 4,
        num_heads:          int   = 4,
        dropout:            float = 0.1,
        modality_dropout_p: float = 0.15,
    ):
        super().__init__()
        self.embed_dim          = embed_dim
        self.num_modalities     = num_modalities
        self.modality_dropout_p = modality_dropout_p

        self.modality_embed = nn.Parameter(
            torch.randn(num_modalities, embed_dim) * 0.02
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

        self.gate_fc          = nn.Linear(embed_dim * num_modalities, num_modalities)
        self.gate_temperature = nn.Parameter(torch.tensor(2.0))

        self._attn_weights = None
        self._gate_weights = None

    def _apply_modality_dropout(self, embeddings: list) -> list:
        if not self.training or self.modality_dropout_p <= 0:
            return embeddings

        n    = len(embeddings)
        keep = torch.ones(n, device=embeddings[0].device)
        for i in range(n):
            if torch.rand(1).item() < self.modality_dropout_p:
                keep[i] = 0.0

        n_kept = int(keep.sum().item())
        if n_kept < 2:
            zeros   = (keep == 0).nonzero(as_tuple=True)[0].tolist()
            restore = torch.randperm(len(zeros))[:2 - n_kept]
            for idx in restore:
                keep[zeros[idx]] = 1.0

        return [
            torch.zeros_like(emb) if keep[i] < 0.5 else emb
            for i, emb in enumerate(embeddings)
        ]

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        assert len(embeddings) == self.num_modalities
        B = embeddings[0].size(0)

        embeddings = self._apply_modality_dropout(list(embeddings))
        x = torch.stack(embeddings, dim=1)
        x = x + self.modality_embed.unsqueeze(0)

        attn_out, attn_weights = self.self_attn(x, x, x)
        self._attn_weights = attn_weights.detach()
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        flat     = x.reshape(B, -1)
        raw_gates = self.gate_fc(flat)
        temp     = torch.clamp(self.gate_temperature, min=0.5, max=5.0)
        gates    = F.softmax(raw_gates / temp, dim=-1)
        self._gate_weights = gates.detach()

        fused = (x * gates.unsqueeze(-1)).sum(dim=1)
        return fused

    def entropy_loss(self) -> torch.Tensor:
        if self._gate_weights is None:
            return torch.tensor(0.0, device=next(self.gate_fc.parameters()).device)
        gates   = self._gate_weights.to(next(self.gate_fc.parameters()).device)
        eps     = 1e-8
        # Standard entropy: -sum(p * log p)
        # We return the negative entropy so that minimizing this loss maximizes entropy.
        entropy = -(gates * torch.log(gates + eps)).sum(dim=-1).mean()
        return -entropy



    def get_modality_weights(self) -> dict:
        if self._gate_weights is None:
            return {}
        avg = self._gate_weights.mean(dim=0).cpu().numpy()
        return {name: float(w) for name, w in zip(self.MODALITY_NAMES, avg)}


# ══════════════════════════════════════════════════════════════════════════════
# SPI OUTPUT HEAD  (multi-task: 6 sub-scores + composite + gate)
# ══════════════════════════════════════════════════════════════════════════════

class SPIOutputHead(nn.Module):
    """
    Multi-task output head for the Synthetic Practicality Index.

    Produces:
      - 6 sub-score predictions (sigmoid → [0,1])
      - 1 composite SPI prediction (sigmoid → [0,1])
      - 1 feasibility gate logit (raw, for BCEWithLogitsLoss)

    Total output shape: (B, 8)
      [:6]  → sub-scores
      [6]   → spi_score
      [7]   → stage1_logit
    """

    N_SUBSCORES = 6

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Per-dimension sub-score heads
        self.sub_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(self.N_SUBSCORES)
        ])

        # Composite SPI head (weighted sum is learnable here too)
        self.spi_head = nn.Linear(hidden_dim + self.N_SUBSCORES, 1)

        # Stage 1 feasibility gate head
        self.gate_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        trunk_out = self.trunk(x)
        
        # NaN Protection
        if torch.isnan(trunk_out).any():
            trunk_out = torch.nan_to_num(trunk_out, nan=0.0)

        sub_logits = torch.cat(
            [head(trunk_out) for head in self.sub_heads], dim=-1
        )  # (B, 6)
        sub_scores = torch.sigmoid(sub_logits)  # (B, 6)

        # ISSUE 6: Detach sub_scores before composite SPI prediction to prevent shortcut learning
        spi_in      = torch.cat([trunk_out, sub_scores.detach()], dim=-1)
        spi_score   = torch.sigmoid(self.spi_head(spi_in))  # (B, 1)

        stage1_logit = self.gate_head(trunk_out)  # (B, 1) raw

        return {
            "sub_scores":   sub_scores,
            "spi_score":    spi_score,
            "stage1_logit": stage1_logit,
        }



# Legacy alias for backward compat with old smoke tests
class OutputHead(nn.Module):
    """Thin wrapper kept for backward compatibility with test_smoke.py."""
    def __init__(self, input_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# SynPractIQ MODEL  (main model class)
# ══════════════════════════════════════════════════════════════════════════════

class SynPractIQModel(nn.Module):
    """
    SynPractIQ — Synthetic Practicality Intelligence System

    4-branch multi-modal molecular model with attention fusion,
    producing multi-dimensional Synthetic Practicality Index predictions.

    Branches:
      1. ANN       : ECFP4 (2048) + 208 descriptors → 256-d
      2. GAT       : 2D molecular graph topology    → 256-d
      3. ChemBERTa : LoRA fine-tuned transformer    → 256-d
      4. EGNN      : 3D equivariant geometry        → 256-d

    Output (per molecule):
      sub_scores   (6,) — one per SPI dimension
      spi_score    (1,) — composite SPI
      stage1_logit (1,) — synthesizability gate
    """

    def __init__(self, dropout: float = 0.3, modality_dropout_p: float = 0.15):
        super().__init__()

        self.ann_branch       = ANNBranch(dropout=dropout)
        self.gat_branch       = GATBranch(dropout=dropout)
        self.chemberta_branch = ChemBERTaBranch()

        if _EGNN_AVAILABLE:
            self.egnn_branch = EGNNBranch(hidden_dim=128, num_layers=4, dropout=0.1)
        else:
            self.egnn_branch = None

        self.fusion = AttentionFusion(
            embed_dim=256, num_modalities=4, num_heads=4,
            dropout=0.1, modality_dropout_p=modality_dropout_p,
        )

        self.output_head = SPIOutputHead(
            input_dim=256, hidden_dim=128, dropout=dropout
        )

    def forward(self, ann_x, graphs, input_ids, attention_mask) -> dict:
        """
        Args:
            ann_x         : (B, 2256) ANN features
            graphs        : torch_geometric Batch (must have .pos for EGNN)
            input_ids     : (B, max_len)
            attention_mask: (B, max_len)

        Returns:
            dict: sub_scores (B,6), spi_score (B,1), stage1_logit (B,1)
        """
        ann_emb  = self.ann_branch(ann_x)
        gat_emb  = self.gat_branch(graphs)
        chem_emb = self.chemberta_branch(input_ids, attention_mask)

        if self.egnn_branch is not None:
            egnn_emb = self.egnn_branch(graphs)
        else:
            egnn_emb = torch.zeros_like(ann_emb)

        fused = self.fusion(ann_emb, gat_emb, chem_emb, egnn_emb)
        return self.output_head(fused)

    def predict_with_uncertainty(self, ann_x, graphs, input_ids,
                                 attention_mask, n_samples: int = 20) -> dict:
        """Monte Carlo Dropout uncertainty estimation."""
        self.train()
        spi_samples, sub_samples = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(ann_x, graphs, input_ids, attention_mask)
                spi_samples.append(out["spi_score"])
                sub_samples.append(out["sub_scores"])

        spi_stack = torch.stack(spi_samples, dim=0)   # (S, B, 1)
        sub_stack = torch.stack(sub_samples, dim=0)   # (S, B, 6)
        self.eval()

        return {
            "spi_mean":      spi_stack.mean(dim=0),
            "spi_std":       spi_stack.std(dim=0),
            "sub_mean":      sub_stack.mean(dim=0),
            "sub_std":       sub_stack.std(dim=0),
        }

    def count_parameters(self) -> dict:
        def _c(m):  return sum(p.numel() for p in m.parameters())
        def _tr(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "ann":       _c(self.ann_branch),
            "gat":       _c(self.gat_branch),
            "chemberta": _c(self.chemberta_branch),
            "egnn":      _c(self.egnn_branch) if self.egnn_branch else 0,
            "fusion":    _c(self.fusion),
            "head":      _c(self.output_head),
            "total":     _c(self),
            "trainable": _tr(self),
        }


# ── Backward-compat alias so old imports still work ───────────────────────────
SynFeasNetV2 = SynPractIQModel


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Batch
    from models.gat_branch       import GraphBuilder
    from models.ann_branch       import ANNFeatureExtractor
    from models.chemBERTa_branch import SMILESTokenizer

    print("=" * 65)
    print("SynPractIQ — Attention Fusion + SPIOutputHead smoke test")
    print("=" * 65)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_smiles  = ["CC(=O)Oc1ccccc1C(=O)O", "CCO"]

    ann_ext = ANNFeatureExtractor()
    ann_ext.fit_descriptors(test_smiles)
    gb  = GraphBuilder()
    tok = SMILESTokenizer(max_length=128)

    ann_feats = torch.tensor(ann_ext.compute_batch(test_smiles), dtype=torch.float32)
    tokens    = tok(test_smiles)

    if _EGNN_AVAILABLE and Graph3DBuilder is not None:
        from models.egnn_branch import Graph3DBuilder as _G3D
        g3d    = _G3D()
        graphs = [g3d.add_coords(gb.build(s), s) for s in test_smiles]
    else:
        graphs = [gb.build(s) for s in test_smiles]
        for g in graphs:
            g.pos = torch.zeros((g.x.size(0), 3), dtype=torch.float32)

    batch_g = Batch.from_data_list(graphs)

    model = SynPractIQModel(modality_dropout_p=0.15)
    model.eval()

    with torch.no_grad():
        out = model(ann_feats, batch_g, tokens["input_ids"], tokens["attention_mask"])

    print(f"sub_scores   : {out['sub_scores'].shape}   (expect [2, 6])")
    print(f"spi_score    : {out['spi_score'].shape}    (expect [2, 1])")
    print(f"stage1_logit : {out['stage1_logit'].shape} (expect [2, 1])")
    assert out["sub_scores"].shape == (2, 6)
    assert out["spi_score"].shape  == (2, 1)

    print("\nModality weights:", model.fusion.get_modality_weights())
    params = model.count_parameters()
    print(f"Total params: {params['total']:,}  Trainable: {params['trainable']:,}")
    print("\n✅ SynPractIQ fusion: all checks passed!")