"""
GAT Branch - SynFeasNet
=======================
Graph/topology representation of molecules.

Input  : SMILES string
Process: Molecule -> graph -> 3 GAT layers -> global mean pool -> projection
Output : 256-dimensional graph embedding
"""

import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")

_BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE: 1.0,
    rdchem.BondType.DOUBLE: 2.0,
    rdchem.BondType.TRIPLE: 3.0,
    rdchem.BondType.AROMATIC: 4.0,
}

_MACROCYCLE_RING_MIN = 8
_MAX_RING_SIZE_NORM = 30.0


class GraphBuilder:
    NODE_DIM = 13
    EDGE_DIM = 4

    def build(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._dummy_graph()

        ring_info = mol.GetRingInfo()
        node_features = self._build_node_features(mol, ring_info)
        edge_index, edge_attr = self._build_edge_features(mol)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )

    def _build_node_features(self, mol, ring_info) -> np.ndarray:
        return np.array(
            [self._atom_features(atom, ring_info) for atom in mol.GetAtoms()],
            dtype=np.float32,
        )

    def _atom_features(self, atom, ring_info) -> list:
        hyb = atom.GetHybridization()
        atom_idx = atom.GetIdx()

        atom_rings = [len(r) for r in ring_info.AtomRings() if atom_idx in r]
        max_ring_size = max(atom_rings) if atom_rings else 0

        return [
            atom.GetAtomicNum() / 118.0,
            atom.GetDegree() / 10.0,
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(hyb == rdchem.HybridizationType.SP),
            float(hyb == rdchem.HybridizationType.SP2),
            float(hyb == rdchem.HybridizationType.SP3),
            float(atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED),
            float(len(atom_rings) > 0),
            atom.GetTotalNumHs() / 4.0,
            float(max_ring_size >= _MACROCYCLE_RING_MIN),
            min(max_ring_size / _MAX_RING_SIZE_NORM, 1.0),
            float(atom.GetNumRadicalElectrons()),
        ]

    def _build_edge_features(self, mol):
        edge_index = [[], []]
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            feat = self._bond_features(bond)

            edge_index[0].extend([i, j])
            edge_index[1].extend([j, i])
            edge_attr.extend([feat, feat])

        if not edge_attr:
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, self.EDGE_DIM), dtype=np.float32),
            )

        return (
            np.array(edge_index, dtype=np.int64),
            np.array(edge_attr, dtype=np.float32),
        )

    def _bond_features(self, bond) -> list:
        return [
            _BOND_TYPE_MAP.get(bond.GetBondType(), 1.0),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetStereo() != rdchem.BondStereo.STEREONONE),
        ]

    def _dummy_graph(self) -> Data:
        return Data(
            x=torch.zeros((1, self.NODE_DIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, self.EDGE_DIM), dtype=torch.float32),
        )


class GATBranch(nn.Module):
    NODE_DIM = 13
    EDGE_DIM = 4
    OUTPUT_DIM = 256

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=self.NODE_DIM,
            out_channels=64,
            heads=8,
            concat=True,
            dropout=dropout,
            edge_dim=self.EDGE_DIM,
            add_self_loops=True,
        )

        self.gat2 = GATConv(
            in_channels=512,
            out_channels=64,
            heads=8,
            concat=True,
            dropout=dropout,
            edge_dim=self.EDGE_DIM,
            add_self_loops=True,
        )

        self.gat3 = GATConv(
            in_channels=512,
            out_channels=32,
            heads=8,
            concat=True,
            dropout=dropout,
            edge_dim=self.EDGE_DIM,
            add_self_loops=True,
        )

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(256, self.OUTPUT_DIM)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        return self.projection(x)


if __name__ == "__main__":
    from torch_geometric.data import Batch

    test_smiles = [
        "CCO",
        "c1ccccc1",
        "CC1NC(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC1=O)CC(C)C",
    ]

    builder = GraphBuilder()
    graphs = [builder.build(smi) for smi in test_smiles]

    for graph in graphs:
        assert graph.x.shape[1] == GraphBuilder.NODE_DIM
        assert graph.edge_attr.shape[1] == GraphBuilder.EDGE_DIM

    batch = Batch.from_data_list(graphs)
    model = GATBranch()
    model.eval()

    with torch.no_grad():
        output = model(batch)

    print(f"GAT output shape: {output.shape}")
    assert output.shape == (len(test_smiles), 256)
    print("GATBranch check passed.")
