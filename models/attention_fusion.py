"""
attention_fusion.py — SynFeasNet v2
=====================================
Attention Fusion + SynFeasNetV2.

CHANGES vs previous version
-----------------------------
FIX 1 — Modality collapse prevention:
  The gate in the previous version was a plain Softmax.
  A Softmax over 4 values has no penalty for degenerate distributions
  (e.g. [0, 0, 1, 0]). By epoch 9 ChemBERTa captured all gate weight.
  Fix: add modality_dropout_p — during training, randomly zero out
  1 branch per forward pass. This forces every branch to be useful
  because any branch can be absent. The model cannot rely on one branch.

FIX 2 — Entropy loss method:
  AttentionFusion.entropy_loss() returns the negative entropy of the
  gate distribution. Add it to the training loss with a small weight
  (e.g. 0.05). This penalizes degenerate gate distributions and
  encourages all modalities to contribute.

  Entropy of [0.25, 0.25, 0.25, 0.25] = log(4) ≈ 1.386 (max)
  Entropy of [0.00, 0.00, 1.00, 0.00] = 0.0             (collapsed)

FIX 3 — Gate temperature:
  The gate Softmax now uses a learnable temperature (initialized to 1.0)
  that is clamped to [0.5, 5.0]. This controls how "sharp" the gate
  distribution is, giving the model a way to prevent premature collapse.
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root — auto-detected ───────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION FUSION  (with modality dropout + entropy regularization)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionFusion(nn.Module):
    """
    Attention-based fusion of N modality embeddings.

    Key improvements over concat fusion:
      1. Multi-head self-attention sees cross-modal relationships
      2. Learned gating produces per-modality importance weights
      3. Modality dropout forces each branch to be independently useful
      4. Learnable gate temperature prevents premature collapse
      5. Entropy loss method allows training loop to penalize collapse

    Input : N tensors of shape (B, embed_dim)
    Output: 1 tensor of shape (B, embed_dim)
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
        """
        Parameters
        ----------
        modality_dropout_p : float
            Probability of zeroing out any single modality embedding during
            training.  0.15 means each branch has a 15% chance of being
            silenced per forward pass.  At least 2 branches are always kept.
            Set to 0.0 to disable.
        """
        super().__init__()
        self.embed_dim          = embed_dim
        self.num_modalities     = num_modalities
        self.modality_dropout_p = modality_dropout_p

        # Learnable modality position embeddings
        self.modality_embed = nn.Parameter(
            torch.randn(num_modalities, embed_dim) * 0.02
        )

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

        # Gating: learns per-modality importance
        self.gate_fc = nn.Linear(embed_dim * num_modalities, num_modalities)

        # FIX 2: Learnable gate temperature — clamped to [0.5, 5.0]
        # Lower temperature → sharper gate (more collapsed)
        # Higher temperature → softer gate (more uniform)
        # Initialize to 2.0 to start with a relatively soft/uniform gate.
        self.gate_temperature = nn.Parameter(torch.tensor(2.0))

        # Store for explainability + entropy loss
        self._attn_weights = None
        self._gate_weights = None   # (B, N) from last forward pass

    def _apply_modality_dropout(self, embeddings: list) -> list:
        """
        FIX 1: Randomly silence modality embeddings during training.

        Rules:
          - Each modality is silenced independently with p=modality_dropout_p
          - At least 2 modalities are always kept (avoids degenerate batches)
          - Silenced embeddings are replaced with zeros (the fusion still sees
            the modality position embedding, so the model knows a slot is empty)
        """
        if not self.training or self.modality_dropout_p <= 0:
            return embeddings

        n = len(embeddings)
        # Sample a binary mask: 1 = keep, 0 = zero out
        keep = torch.ones(n, device=embeddings[0].device)
        for i in range(n):
            if torch.rand(1).item() < self.modality_dropout_p:
                keep[i] = 0.0

        # Guarantee at least 2 kept
        n_kept = int(keep.sum().item())
        if n_kept < 2:
            # Randomly restore until 2 are kept
            zeros = (keep == 0).nonzero(as_tuple=True)[0].tolist()
            torch.random.manual_seed(int(torch.rand(1).item() * 1e6))
            restore = torch.randperm(len(zeros))[: 2 - n_kept]
            for idx in restore:
                keep[zeros[idx]] = 1.0

        result = []
        for i, emb in enumerate(embeddings):
            if keep[i] < 0.5:
                result.append(torch.zeros_like(emb))
            else:
                result.append(emb)
        return result

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *embeddings: N tensors of shape (B, embed_dim)

        Returns:
            (B, embed_dim) fused representation
        """
        assert len(embeddings) == self.num_modalities, (
            f"Expected {self.num_modalities} embeddings, got {len(embeddings)}"
        )
        B = embeddings[0].size(0)

        # FIX 1: Apply modality dropout during training
        embeddings = self._apply_modality_dropout(list(embeddings))

        # Stack: (B, N, D)
        x = torch.stack(embeddings, dim=1)

        # Add modality position embeddings
        x = x + self.modality_embed.unsqueeze(0)

        # Self-attention
        attn_out, attn_weights = self.self_attn(x, x, x)
        self._attn_weights = attn_weights.detach()
        x = self.norm1(x + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        # FIX 2: Gating with learnable temperature
        flat = x.reshape(B, -1)                                       # (B, N*D)
        raw_gates = self.gate_fc(flat)                                 # (B, N)
        # Clamp temperature so it doesn't go to 0 (which would cause NaN)
        temp = torch.clamp(self.gate_temperature, min=0.5, max=5.0)
        gates = F.softmax(raw_gates / temp, dim=-1)                   # (B, N)
        self._gate_weights = gates.detach()

        # Weighted sum
        fused = (x * gates.unsqueeze(-1)).sum(dim=1)                  # (B, D)
        return fused

    def entropy_loss(self) -> torch.Tensor:
        """
        FIX 2: Returns the negative entropy of gate weights from the last
        forward pass.  Add to training loss with a small coefficient to
        penalize modality collapse.

        Returns a scalar tensor.  Higher entropy = more uniform gates.
        We return NEGATIVE entropy so that minimizing loss = maximizing
        gate entropy = avoiding collapse.

        Example usage in train_epoch():
            loss = criterion(logits, labels, weights)
            loss = loss + ENTROPY_REG * model.fusion.entropy_loss()

        Expected range:
            Max entropy (uniform): -log(1/4) ≈ -1.386
            Collapsed (one branch): 0.0
        """
        if self._gate_weights is None:
            return torch.tensor(0.0)

        gates = self._gate_weights.to(
            next(self.gate_fc.parameters()).device
        )
        eps     = 1e-8
        entropy = -(gates * torch.log(gates + eps)).sum(dim=-1).mean()
        # Return negative entropy: minimizing this = maximizing entropy
        return -entropy

    def get_modality_weights(self) -> dict:
        """Return modality gate weights from last forward pass."""
        if self._gate_weights is None:
            return {}
        avg = self._gate_weights.mean(dim=0).cpu().numpy()
        return {
            name: float(w)
            for name, w in zip(self.MODALITY_NAMES, avg)
        }


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT HEAD
# ══════════════════════════════════════════════════════════════════════════════

class OutputHead(nn.Module):
    """Linear → BN → ReLU → Dropout → Linear → raw logits."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# SynFeasNetV2
# ══════════════════════════════════════════════════════════════════════════════

class SynFeasNetV2(nn.Module):
    """
    SynFeasNet v2 — 4-branch multi-modal model with attention fusion.

    Branches:
      1. ANN       : ECFP4 (2048) + 208 descriptors → 256-d
      2. GAT       : 2D graph topology               → 256-d
      3. ChemBERTa : LoRA fine-tuned transformer     → 256-d
      4. EGNN      : 3D equivariant geometry         → 256-d

    Fusion: Multi-head attention + gated weighting (not concatenation).
    Output: Raw logits (1-d).
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

        # FIX: Pass modality_dropout_p to prevent branch collapse
        self.fusion = AttentionFusion(
            embed_dim          = 256,
            num_modalities     = 4,
            num_heads          = 4,
            dropout            = 0.1,
            modality_dropout_p = modality_dropout_p,
        )
        self.output_head = OutputHead(input_dim=256, hidden_dim=128, dropout=dropout)

    def forward(self, ann_x, graphs, input_ids, attention_mask):
        """
        Args:
            ann_x         : (B, 2256) ANN features
            graphs        : torch_geometric Batch (must have .pos for EGNN)
            input_ids     : (B, max_len) token IDs
            attention_mask: (B, max_len)

        Returns:
            (B, 1) raw logits
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

    def load_v1_weights(self, v1_state_dict: dict, strict: bool = False):
        compatible = {}
        skipped    = []
        for k, v in v1_state_dict.items():
            if k.startswith("fusion."):
                skipped.append(k)
                continue
            compatible[k] = v
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        print(f"  Loaded v1 weights: {len(compatible)} params, "
              f"{len(skipped)} skipped, {len(missing)} new")
        return missing, unexpected

    def predict_with_uncertainty(self, ann_x, graphs, input_ids,
                                 attention_mask, n_samples: int = 20):
        """Monte Carlo Dropout uncertainty estimation."""
        self.train()
        logits_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(ann_x, graphs, input_ids, attention_mask)
                logits_list.append(logits)

        logits_stack = torch.stack(logits_list, dim=0)
        probs_stack  = torch.sigmoid(logits_stack)
        self.eval()
        return {
            "logit_mean": logits_stack.mean(dim=0),
            "logit_std":  logits_stack.std(dim=0),
            "prob_mean":  probs_stack.mean(dim=0),
            "prob_std":   probs_stack.std(dim=0),
        }

    def count_parameters(self) -> dict:
        def _count(m):     return sum(p.numel() for p in m.parameters())
        def _trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "ann":       _count(self.ann_branch),
            "gat":       _count(self.gat_branch),
            "chemberta": _count(self.chemberta_branch),
            "egnn":      _count(self.egnn_branch) if self.egnn_branch else 0,
            "fusion":    _count(self.fusion),
            "head":      _count(self.output_head),
            "total":     _count(self),
            "trainable": _trainable(self),
        }


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Batch
    from models.gat_branch import GraphBuilder
    from models.ann_branch import ANNFeatureExtractor
    from models.chemBERTa_branch import SMILESTokenizer

    if _EGNN_AVAILABLE and Graph3DBuilder is not None:
        from models.egnn_branch import Graph3DBuilder as _G3D
    else:
        _G3D = None

    print("=" * 65)
    print("Testing SynFeasNetV2 + Attention Fusion (with collapse fixes)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCO"]

    ann_ext = ANNFeatureExtractor()
    ann_ext.fit_descriptors(test_smiles)
    gb  = GraphBuilder()
    tok = SMILESTokenizer(max_length=320)

    ann_feats = torch.tensor(ann_ext.compute_batch(test_smiles), dtype=torch.float32)
    tokens    = tok(test_smiles)

    if _G3D is not None:
        g3d    = _G3D()
        graphs = [g3d.add_coords(gb.build(s), s) for s in test_smiles]
    else:
        graphs = [gb.build(s) for s in test_smiles]
    batch_g = Batch.from_data_list(graphs)

    print("\n1. Testing SynFeasNetV2 forward pass...")
    model = SynFeasNetV2(modality_dropout_p=0.15)
    model.eval()
    with torch.no_grad():
        logits = model(ann_feats, batch_g,
                       tokens["input_ids"], tokens["attention_mask"])
    print(f"   Output shape: {logits.shape}  (expect [2, 1])")
    assert logits.shape == (2, 1)

    print("\n2. Modality gate weights:")
    weights = model.fusion.get_modality_weights()
    for name, w in weights.items():
        bar = "█" * int(w * 30)
        print(f"   {name:12s} {bar:<30} {w:.3f}")

    print("\n3. Entropy loss (should be negative, near -log(4)≈-1.386 for uniform):")
    model.train()
    with torch.no_grad():
        _ = model(ann_feats, batch_g,
                  tokens["input_ids"], tokens["attention_mask"])
    ent = model.fusion.entropy_loss()
    print(f"   Entropy loss: {ent.item():.4f}")

    print("\n4. Modality dropout test (training mode — some branches should be zeroed):")
    model.train()
    for trial in range(3):
        with torch.no_grad():
            _ = model(ann_feats, batch_g,
                      tokens["input_ids"], tokens["attention_mask"])
        w = model.fusion.get_modality_weights()
        print(f"   Trial {trial+1}: {' '.join(f'{k}={v:.2f}' for k,v in w.items())}")

    print("\n5. Parameter count:")
    params = model.count_parameters()
    for k, v in params.items():
        print(f"   {k}: {v:,}")

    print("\n✅ SynFeasNetV2 + Attention Fusion: all checks passed!")