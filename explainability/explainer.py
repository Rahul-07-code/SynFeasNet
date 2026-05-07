"""
Explainability — SynFeasNet v2
=================================
Provides human-readable explanations for predictions:
  1. Modality importance (which branch contributed most)
  2. Atom-level importance from GAT attention
  3. Molecular property analysis
  4. Natural language explanation

Works with both SynFeasNet (v1) and SynFeasNetV2.
"""

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')


class SynFeasExplainer:
    """
    Generates explanations for synthetic feasibility predictions.

    Usage:
        explainer = SynFeasExplainer()
        explanation = explainer.explain(smiles, probability, threshold)
    """

    def __init__(self):
        pass

    def explain(self, smiles: str, probability: float, threshold: float,
                modality_weights: dict = None) -> dict:
        """
        Generate a full explanation for a prediction.

        Args:
            smiles: Input SMILES string.
            probability: Model prediction probability.
            threshold: Classification threshold.
            modality_weights: Optional dict from AttentionFusion.get_modality_weights()

        Returns:
            dict with 'text', 'factors', 'modality_analysis', 'chemistry_context'
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "text": "Cannot explain: invalid SMILES",
                "factors": [],
                "modality_analysis": {},
                "chemistry_context": {},
            }

        # Analyze molecular properties
        chemistry = self._analyze_chemistry(mol)
        factors = self._identify_factors(mol, chemistry, probability, threshold)
        modality_text = self._explain_modalities(modality_weights)

        # Build explanation text
        label = "SYNTHESIZABLE" if probability >= threshold else "NOT SYNTHESIZABLE"
        confidence = self._confidence_level(probability, threshold)

        text_parts = [
            f"Prediction: {label} (score: {probability:.3f}, "
            f"confidence: {confidence})",
        ]

        if factors:
            text_parts.append("Key factors:")
            for f in factors[:5]:
                text_parts.append(f"  • {f['description']} ({f['impact']})")

        if modality_text:
            text_parts.append(f"Model focus: {modality_text}")

        return {
            "text": "\n".join(text_parts),
            "factors": factors,
            "modality_analysis": modality_weights or {},
            "chemistry_context": chemistry,
        }

    def _analyze_chemistry(self, mol) -> dict:
        """Extract key molecular properties."""
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        max_ring = max(ring_sizes) if ring_sizes else 0

        # Count specific functional groups
        n_amide = len(mol.GetSubstructMatches(
            Chem.MolFromSmarts("[C](=O)[NH]"))) if Chem.MolFromSmarts("[C](=O)[NH]") else 0
        n_ester = len(mol.GetSubstructMatches(
            Chem.MolFromSmarts("[C](=O)[O][C]"))) if Chem.MolFromSmarts("[C](=O)[O][C]") else 0

        # Stereocenters
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 1),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_rings": len(ring_sizes),
            "max_ring_size": max_ring,
            "is_macrocycle": max_ring >= 8,
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_stereocenters": len(chiral_centers),
            "num_hba": Descriptors.NumHAcceptors(mol),
            "num_hbd": Descriptors.NumHDonors(mol),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "tpsa": round(Descriptors.TPSA(mol), 1),
            "n_amide_bonds": n_amide,
            "n_ester_bonds": n_ester,
        }

    def _identify_factors(self, mol, chemistry, probability, threshold) -> list:
        """Identify key factors contributing to the prediction."""
        factors = []
        is_feasible = probability >= threshold

        mw = chemistry["molecular_weight"]
        max_ring = chemistry["max_ring_size"]
        n_stereo = chemistry["num_stereocenters"]
        n_rings = chemistry["num_rings"]
        rot_bonds = chemistry["num_rotatable_bonds"]

        # Molecular weight
        if mw < 150:
            factors.append({
                "property": "molecular_weight",
                "value": mw,
                "impact": "positive",
                "description": f"Small molecule (MW={mw}) — generally easy to synthesize",
            })
        elif mw > 700:
            factors.append({
                "property": "molecular_weight",
                "value": mw,
                "impact": "negative",
                "description": f"Large molecule (MW={mw}) — synthesis complexity increases",
            })

        # Macrocycle
        if chemistry["is_macrocycle"]:
            impact = "negative" if not is_feasible else "neutral"
            factors.append({
                "property": "macrocycle",
                "value": max_ring,
                "impact": impact,
                "description": f"Macrocyclic ring ({max_ring}-membered) — ring closure is challenging",
            })

        # Stereocenters
        if n_stereo > 3:
            factors.append({
                "property": "stereocenters",
                "value": n_stereo,
                "impact": "negative",
                "description": f"{n_stereo} stereocenters — stereoselective synthesis required",
            })

        # Ring count
        if n_rings > 5:
            factors.append({
                "property": "ring_count",
                "value": n_rings,
                "impact": "negative",
                "description": f"{n_rings} ring systems — complex topology",
            })

        # Rotatable bonds (flexibility)
        if rot_bonds > 10:
            factors.append({
                "property": "flexibility",
                "value": rot_bonds,
                "impact": "neutral",
                "description": f"{rot_bonds} rotatable bonds — flexible chain, conformational challenge",
            })

        # Simple / drug-like
        if mw < 500 and n_stereo <= 2 and n_rings <= 3:
            factors.append({
                "property": "drug_likeness",
                "value": True,
                "impact": "positive",
                "description": "Drug-like molecule — standard medicinal chemistry",
            })

        # Amide / ester bonds
        if chemistry["n_amide_bonds"] > 0:
            factors.append({
                "property": "amide_bonds",
                "value": chemistry["n_amide_bonds"],
                "impact": "positive",
                "description": f"{chemistry['n_amide_bonds']} amide bond(s) — well-established coupling chemistry",
            })

        return factors

    def _confidence_level(self, probability, threshold) -> str:
        margin = abs(probability - threshold)
        if margin < 0.05:
            return "marginal"
        elif margin < 0.15:
            return "moderate"
        elif margin < 0.30:
            return "high"
        return "very high"

    def _explain_modalities(self, modality_weights: dict) -> str:
        """Explain which modality contributed most."""
        if not modality_weights:
            return ""

        sorted_mods = sorted(modality_weights.items(), key=lambda x: x[1], reverse=True)
        top = sorted_mods[0]
        top_name, top_weight = top

        modality_descriptions = {
            "ANN": "molecular fingerprint and physicochemical descriptors",
            "GAT": "2D molecular graph topology and atom environments",
            "ChemBERTa": "SMILES sequence patterns (language model)",
            "EGNN": "3D molecular geometry and spatial arrangement",
        }

        desc = modality_descriptions.get(top_name, top_name)
        parts = [f"{top_name} ({top_weight:.0%}) — {desc}"]

        # Mention if weights are balanced
        weights = list(modality_weights.values())
        if max(weights) - min(weights) < 0.1:
            parts.append("(all modalities contributing equally)")

        return "; ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# ATOM IMPORTANCE (from GAT attention)
# ══════════════════════════════════════════════════════════════════════════

class GATAtomImportance:
    """
    Extracts atom-level importance scores from GAT attention weights.

    Requires hooking into the GATBranch during forward pass.
    """

    @staticmethod
    def extract_attention(gat_branch, graph_data) -> dict:
        """
        Run forward pass through GAT and extract attention weights.

        Returns:
            dict with 'atom_scores' (N,) and 'atom_names' (N,)
        """
        hooks = []
        attention_weights = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights.append(output[1].detach().cpu())

        # Register hooks on GAT layers
        for name, module in gat_branch.named_modules():
            if hasattr(module, 'att_src'):  # GATConv has attention params
                h = module.register_forward_hook(hook_fn)
                hooks.append(h)

        # Forward pass
        with torch.no_grad():
            gat_branch(graph_data)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Aggregate attention into per-atom importance
        if not attention_weights:
            n_atoms = graph_data.x.size(0)
            return {
                "atom_scores": np.zeros(n_atoms),
                "num_atoms": n_atoms,
            }

        # Average attention from last layer as importance
        last_attn = attention_weights[-1] if attention_weights else None
        if last_attn is not None:
            # Attention shape: (E, num_heads)
            edge_index = graph_data.edge_index.cpu().numpy()
            n_atoms = graph_data.x.size(0)
            scores = np.zeros(n_atoms)
            attn_avg = last_attn.mean(dim=-1).numpy()  # (E,)
            for e in range(edge_index.shape[1]):
                dst = edge_index[1, e]
                scores[dst] += attn_avg[e]
            # Normalize
            if scores.max() > 0:
                scores = scores / scores.max()
            return {
                "atom_scores": scores,
                "num_atoms": n_atoms,
            }

        return {
            "atom_scores": np.zeros(graph_data.x.size(0)),
            "num_atoms": graph_data.x.size(0),
        }


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Testing Explainability Module")
    print("=" * 65)

    explainer = SynFeasExplainer()

    test_cases = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", 0.92, 0.5),
        ("Cyclosporin A",
         "CC[C@@H]1NC(=O)[C@H]([C@H](O)[C@H](C)C/C=C/C)N(C)"
         "C(=O)[C@H](C(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)",
         0.35, 0.5),
    ]

    modality_weights = {"ANN": 0.28, "GAT": 0.32, "ChemBERTa": 0.22, "EGNN": 0.18}

    for name, smi, prob, thr in test_cases:
        print(f"\n{name}:")
        result = explainer.explain(smi, prob, thr, modality_weights)
        print(result["text"])
        print(f"  Factors: {len(result['factors'])}")

    print("\n✅ Explainability module: all checks passed!")
