"""
EGNN Branch — SynFeasNet v2
==============================
E(n)-Equivariant Graph Neural Network for 3D molecular geometry.

Input  : Molecular graph with 3D coordinates
         (same 13-dim node features as GATBranch + 3D coords from ETKDG)
Process: 4 EGNN layers with coordinate updates
         → global mean pooling
Output : 256-dim embedding

WHY EGNN:
  - Captures 3D spatial relationships (bond angles, torsions, ring strain)
  - Equivariant to rotations/translations (physically correct)
  - Lightweight: ~200K params, fits on 8GB GPU alongside other branches
  - Complementary to GAT (which only sees topology, not geometry)

COORDINATE GENERATION:
  Uses RDKit ETKDG conformer generation + MMFF optimization.
  If 3D generation fails, falls back to zero coordinates
  (EGNN degrades gracefully to a message-passing GNN in this case).

Compatible with existing GraphBuilder (models/gat_branch.py).
"""

"""
EGNN Branch — SynFeasNet v2
==============================
E(n)-Equivariant Graph Neural Network for 3D molecular geometry.
"""

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

RDLogger.DisableLog('rdApp.*')


# ════════════════════════════════════════════════════════════════════════
# 3D COORDINATE GENERATOR
# ════════════════════════════════════════════════════════════════════════

class ConformerGenerator:

    def __init__(
        self,
        max_attempts: int = 50,
        optimize: bool = True,
        ff_max_iters: int = 500,
        seed: int = 42
    ):
        self.max_attempts = max_attempts
        self.optimize = optimize
        self.ff_max_iters = ff_max_iters
        self.seed = seed

    def generate(self, smiles: str):

        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                return None

            mol_h = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.maxAttempts = self.max_attempts
            params.randomSeed = self.seed
            params.useSmallRingTorsions = True
            params.useMacrocycleTorsions = True

            conf_id = AllChem.EmbedMolecule(mol_h, params)

            if conf_id < 0:
                conf_id = AllChem.EmbedMolecule(
                    mol_h,
                    randomSeed=self.seed
                )

                if conf_id < 0:
                    return None

            if self.optimize:

                try:
                    result = AllChem.MMFFOptimizeMolecule(
                        mol_h,
                        maxIters=self.ff_max_iters
                    )

                    if result == -1:
                        AllChem.UFFOptimizeMolecule(
                            mol_h,
                            maxIters=self.ff_max_iters
                        )

                except Exception:
                    pass

            mol_3d = Chem.RemoveHs(mol_h)
            conf = mol_3d.GetConformer()

            positions = conf.GetPositions()

            return positions.astype(np.float32)

        except Exception:
            return None


# ════════════════════════════════════════════════════════════════════════
# EGNN LAYER
# ════════════════════════════════════════════════════════════════════════

class EGNNLayer(nn.Module):

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        edge_dim: int = 0,
        update_coords: bool = True
    ):
        super().__init__()

        self.update_coords = update_coords

        msg_input_dim = 2 * node_dim + 1 + edge_dim

        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        if update_coords:

            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )

            nn.init.xavier_uniform_(
                self.coord_mlp[-1].weight,
                gain=0.001
            )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, x, edge_index, edge_attr=None):

        # ============================================================
        # FIX DTYPE MISMATCH
        # ============================================================

        h = h.float()
        x = x.float()

        if edge_attr is not None:
            edge_attr = edge_attr.float()

        edge_index = edge_index.long()

        src, dst = edge_index

        # ============================================================
        # RELATIVE POSITIONS
        # ============================================================

        rel_pos = x[src] - x[dst]

        # ============================================================
        # DISTANCES
        # ============================================================

        dist_sq = (rel_pos ** 2).sum(
            dim=-1,
            keepdim=True
        ).float()

        # ============================================================
        # MESSAGE INPUT
        # ============================================================

        msg_parts = [
            h[src],
            h[dst],
            dist_sq
        ]

        if edge_attr is not None:
            msg_parts.append(edge_attr)

        msg_in = torch.cat(
            msg_parts,
            dim=-1
        ).float()

        # ============================================================
        # MESSAGE COMPUTATION
        # ============================================================

        msg = self.msg_mlp(msg_in).float()

        # ============================================================
        # AGGREGATION
        # ============================================================

        agg = torch.zeros(
            h.size(0),
            msg.size(-1),
            device=msg.device,
            dtype=msg.dtype
        )

        agg.scatter_add_(
            0,
            dst.unsqueeze(-1).expand_as(msg),
            msg
        )

        # ============================================================
        # NODE UPDATE
        # ============================================================

        h_new = self.node_mlp(
            torch.cat([h, agg], dim=-1)
        )

        h_new = self.norm(h_new + h)

        # ============================================================
        # COORDINATE UPDATE
        # ============================================================

        if self.update_coords:

            coord_w = self.coord_mlp(msg).float()

            weighted = (
                rel_pos * coord_w
            ).float()

            coord_agg = torch.zeros(
                x.size(0),
                x.size(1),
                device=x.device,
                dtype=weighted.dtype
            )

            coord_agg.scatter_add_(
                0,
                dst.unsqueeze(-1).expand_as(weighted),
                weighted
            )

            x_new = x + coord_agg

        else:
            x_new = x

        return h_new, x_new


# ════════════════════════════════════════════════════════════════════════
# EGNN BRANCH
# ════════════════════════════════════════════════════════════════════════

class EGNNBranch(nn.Module):

    NODE_DIM = 13
    EDGE_DIM = 4
    OUTPUT_DIM = 256

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            self.NODE_DIM,
            hidden_dim
        )

        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=hidden_dim,
                hidden_dim=hidden_dim,
                edge_dim=self.EDGE_DIM,
                update_coords=(i < num_layers - 1),
            )
            for i in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.OUTPUT_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(
            self.input_proj.weight
        )

        nn.init.zeros_(
            self.input_proj.bias
        )

        for m in self.output_proj.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):

        h = self.input_proj(data.x.float())

        x = data.pos.float()

        edge_index = data.edge_index.long()

        edge_attr = data.edge_attr

        if edge_attr is not None:
            edge_attr = edge_attr.float()

        for layer in self.layers:
            h, x = layer(
                h,
                x,
                edge_index,
                edge_attr
            )

        out = global_mean_pool(
            h,
            data.batch
        )

        out = self.output_proj(out)

        return out


# ════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER EXTENSION
# ════════════════════════════════════════════════════════════════════════

class Graph3DBuilder:

    def __init__(self):
        self.conformer_gen = ConformerGenerator()

    def add_coords(
        self,
        graph: Data,
        smiles: str
    ):

        n_atoms = graph.x.size(0)

        coords = self.conformer_gen.generate(smiles)

        if coords is not None and coords.shape[0] == n_atoms:

            graph.pos = torch.tensor(
                coords,
                dtype=torch.float32
            )

        else:

            graph.pos = torch.zeros(
                (n_atoms, 3),
                dtype=torch.float32
            )

        return graph

# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from torch_geometric.data import Batch
    from models.gat_branch import GraphBuilder

    print("=" * 65)
    print("Testing EGNN Branch — SynFeasNet v2")
    print("=" * 65)

    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
        "CC1NC(=O)C(C(O)CC)N(C)C(=O)C(CC(C)C)NC(=O)c2csc(n2)C(C)C",
    ]

    gb = GraphBuilder()
    g3d = Graph3DBuilder()

    print("\n1. Building graphs with 3D coordinates...")
    graphs = []
    for smi in test_smiles:
        g = gb.build(smi)
        g = g3d.add_coords(g, smi)
        print(f"   {smi[:50]}... | nodes={g.x.shape[0]} | "
              f"pos={g.pos.shape} | has_3d={g.pos.abs().sum() > 0}")
        graphs.append(g)

    print("\n2. Testing EGNNBranch forward pass...")
    model = EGNNBranch(hidden_dim=128, num_layers=4)
    model.eval()
    batch = Batch.from_data_list(graphs)

    with torch.no_grad():
        emb = model(batch)
    print(f"   Output shape: {emb.shape}  (expect [2, 256])")
    assert emb.shape == (2, 256), f"Wrong shape: {emb.shape}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n3. Parameters: {total_params:,}")

    print("\n4. Testing ConformerGenerator scalar features...")
    cg = ConformerGenerator()
    feats = cg.generate_scalar_features(test_smiles[0])
    print(f"   3D features: {feats}  (energy, mean_dist, max_dist, rog)")

    print("\n✅ EGNN Branch: all checks passed!")
