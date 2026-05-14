"""
explainer.py — SynPractIQ v3
================================
Provides human-readable explanations for Synthetic Practicality Index (SPI)
predictions.  Covers:

  1. Per-dimension SPI explanation with chemical rationale
  2. Bottleneck identification (which dimension limits the score most)
  3. Atom-level importance from GAT attention weights
  4. Actionable suggestions to improve synthesizability
  5. Modality importance (which branch drove the prediction)

Usage:
    from explainability.explainer import SynPractIQExplainer
    explainer = SynPractIQExplainer()
    result    = explainer.explain(smiles, spi_result)
    print(result["text"])
"""

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Dimension display names
DIM_DISPLAY = {
    "synthetic_complexity":   "Synthetic Complexity",
    "route_practicality":     "Route Practicality",
    "precursor_availability": "Precursor Availability",
    "scalability":            "Scalability",
    "retro_confidence":       "Retrosynthesis Confidence",
    "medchem_realism":        "Medicinal Chemistry Realism",
}

# Human-readable thresholds per dimension
DIM_THRESHOLDS = {
    "high":   0.65,
    "medium": 0.40,
    "low":    0.00,
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════

class SynPractIQExplainer:
    """
    Generates structured explanations for SPI predictions.

    Accepts the output dict from predict.predict() directly.
    """

    def __init__(self):
        pass

    def explain(self, smiles: str, spi_result: dict,
                modality_weights: dict = None) -> dict:
        """
        Generate a full explanation for an SPI prediction.

        Args:
            smiles        : Input SMILES string.
            spi_result    : Output dict from predict.predict().
            modality_weights: Optional from model.fusion.get_modality_weights().

        Returns dict with:
            text          : str — full narrative
            factors       : list[dict] — per-factor analysis
            suggestions   : list[str] — actionable improvements
            bottleneck    : str — the weakest SPI dimension
            modality_info : str — which branch drove prediction
            dim_analysis  : dict — per-dimension explanation
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"text": "Cannot explain: invalid SMILES",
                    "factors": [], "suggestions": [], "bottleneck": "",
                    "modality_info": "", "dim_analysis": {}}

        chemistry     = spi_result.get("chemistry", {})
        dimensions    = spi_result.get("spi_dimensions", {})
        spi_score     = spi_result.get("spi_score", 0.5)
        spi_label     = spi_result.get("spi_label", "unknown")
        stage1_pass   = spi_result.get("stage1_pass", True)

        mol_props     = self._analyze_mol(mol)
        factors       = self._identify_factors(mol, mol_props, dimensions)
        dim_analysis  = self._explain_dimensions(dimensions)
        bottleneck    = min(dimensions, key=dimensions.get) if dimensions else ""
        suggestions   = self._generate_suggestions(mol, mol_props, dimensions, bottleneck)
        modality_info = self._explain_modalities(modality_weights)

        # ── Narrative ─────────────────────────────────────────────────────────
        parts = [
            f"═══ SPI Explanation ═══",
            f"Overall: {spi_label.upper()} (SPI = {spi_score:.3f})",
            f"Stage 1 Gate: {'✓ Passes synthesizability check' if stage1_pass else '✗ Fails synthesizability check'}",
            "",
        ]

        if dimensions:
            parts.append("Dimension Breakdown:")
            for dim, score in sorted(dimensions.items(), key=lambda x: x[1]):
                tier  = self._tier(score)
                dname = DIM_DISPLAY.get(dim, dim)
                parts.append(f"  [{tier:6s}] {dname}: {score:.3f}")
            parts.append("")

        if bottleneck:
            b_score = dimensions.get(bottleneck, 0.0)
            b_name  = DIM_DISPLAY.get(bottleneck, bottleneck)
            parts.append(f"⚠  Bottleneck: {b_name} ({b_score:.3f}) — see suggestions below.")
            parts.append("")

        if factors:
            parts.append("Key Chemical Factors:")
            for f in factors[:6]:
                icon = "+" if f["impact"] == "positive" else ("−" if f["impact"] == "negative" else "~")
                parts.append(f"  [{icon}] {f['description']}")
            parts.append("")

        if suggestions:
            parts.append("Actionable Suggestions:")
            for s in suggestions:
                parts.append(f"  → {s}")
            parts.append("")

        if modality_info:
            parts.append(f"Model Focus: {modality_info}")

        return {
            "text":         "\n".join(parts),
            "factors":      factors,
            "suggestions":  suggestions,
            "bottleneck":   bottleneck,
            "modality_info": modality_info,
            "dim_analysis": dim_analysis,
        }

    # ── Chemistry analysis ────────────────────────────────────────────────────

    def _analyze_mol(self, mol) -> dict:
        ring_info  = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        max_ring   = max(ring_sizes) if ring_sizes else 0
        chiral     = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        n_amide = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[C](=O)[NH]") or Chem.MolFromSmarts("CC")))
        n_ester = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[C](=O)[O][C]") or Chem.MolFromSmarts("CC")))

        # Detect known difficult fragments
        _azide   = Chem.MolFromSmarts("[N-]=[N+]=[N,n]")
        _diazo   = Chem.MolFromSmarts("[C]=[N+]=[N-]")
        _isocyan = Chem.MolFromSmarts("[N]=[C]=[O]")
        _perox   = Chem.MolFromSmarts("[O]-[O]")

        has_azide   = mol.HasSubstructMatch(_azide)   if _azide   else False
        has_diazo   = mol.HasSubstructMatch(_diazo)   if _diazo   else False
        has_isocyan = mol.HasSubstructMatch(_isocyan) if _isocyan else False
        has_perox   = mol.HasSubstructMatch(_perox)   if _perox   else False

        return {
            "mw":            Descriptors.MolWt(mol),
            "heavy_atoms":   mol.GetNumHeavyAtoms(),
            "rings":         len(ring_sizes),
            "max_ring":      max_ring,
            "stereocenters": len(chiral),
            "rot_bonds":     Descriptors.NumRotatableBonds(mol),
            "hba":           Descriptors.NumHAcceptors(mol),
            "hbd":           Descriptors.NumHDonors(mol),
            "logp":          Descriptors.MolLogP(mol),
            "tpsa":          Descriptors.TPSA(mol),
            "n_amide":       n_amide,
            "n_ester":       n_ester,
            "is_macrocycle": max_ring >= 8,
            "has_azide":     has_azide,
            "has_diazo":     has_diazo,
            "has_isocyan":   has_isocyan,
            "has_perox":     has_perox,
        }

    # ── Factor identification ─────────────────────────────────────────────────

    def _identify_factors(self, mol, props, dimensions) -> list:
        factors = []

        # Molecular weight
        mw = props["mw"]
        if mw < 200:
            factors.append({"property": "mw", "value": mw, "impact": "positive",
                             "description": f"Small molecule (MW={mw:.0f}) — typically straightforward synthesis"})
        elif mw > 700:
            factors.append({"property": "mw", "value": mw, "impact": "negative",
                             "description": f"Large molecule (MW={mw:.0f}) — multi-step route, high convergent complexity"})

        # Stereochemistry
        n_ster = props["stereocenters"]
        if n_ster == 0:
            factors.append({"property": "stereo", "value": n_ster, "impact": "positive",
                             "description": "No stereocenters — no asymmetric synthesis required"})
        elif n_ster > 3:
            factors.append({"property": "stereo", "value": n_ster, "impact": "negative",
                             "description": f"{n_ster} stereocenters — stereoselective synthesis required (exponentially harder)"})

        # Macrocycle
        if props["is_macrocycle"]:
            factors.append({"property": "macrocycle", "value": props["max_ring"], "impact": "negative",
                             "description": f"Macrocyclic ring ({props['max_ring']}-membered) — challenging ring closure"})

        # Ring count
        n_rings = props["rings"]
        if n_rings > 5:
            factors.append({"property": "rings", "value": n_rings, "impact": "negative",
                             "description": f"{n_rings} ring systems — polycyclic complexity increases route length"})
        elif n_rings == 0:
            factors.append({"property": "rings", "value": 0, "impact": "positive",
                             "description": "Acyclic molecule — no ring-forming steps needed"})

        # Drug-likeness proxy
        if props["mw"] < 500 and n_ster <= 2 and n_rings <= 3 and props["logp"] < 5:
            factors.append({"property": "drug_like", "value": True, "impact": "positive",
                             "description": "Drug-like (Lipinski-compliant) — favors established medicinal chemistry routes"})

        # Dangerous functional groups
        for fg, name in [("has_azide","Azide"), ("has_diazo","Diazo"),
                          ("has_perox","Peroxide"), ("has_isocyan","Isocyanate")]:
            if props.get(fg):
                factors.append({"property": fg, "value": True, "impact": "negative",
                                 "description": f"{name} group present — explosive/highly reactive; severe safety concerns"})

        # Amide bonds (positive for coupling chemistry)
        if props["n_amide"] > 0:
            factors.append({"property": "amide", "value": props["n_amide"], "impact": "positive",
                             "description": f"{props['n_amide']} amide bond(s) — well-established HATU/HBTU coupling chemistry"})

        # Flexibility
        rb = props["rot_bonds"]
        if rb > 12:
            factors.append({"property": "rot_bonds", "value": rb, "impact": "neutral",
                             "description": f"{rb} rotatable bonds — flexible chain may complicate purification"})

        return factors

    # ── Dimension explanation ─────────────────────────────────────────────────

    def _explain_dimensions(self, dimensions: dict) -> dict:
        """Returns per-dimension textual analysis."""
        explanations = {}
        for dim, score in dimensions.items():
            tier  = self._tier(score)
            dname = DIM_DISPLAY.get(dim, dim)

            if dim == "synthetic_complexity":
                if score > 0.65:
                    text = f"{dname} is LOW ({score:.3f}) — molecule is relatively straightforward to build."
                elif score > 0.40:
                    text = f"{dname} is MODERATE ({score:.3f}) — several synthetic steps expected."
                else:
                    text = f"{dname} is HIGH ({score:.3f}) — complex stereocenters, rings, or protecting groups."

            elif dim == "route_practicality":
                if score > 0.65:
                    text = f"{dname} is HIGH ({score:.3f}) — robust, high-yield routes predicted."
                elif score > 0.40:
                    text = f"{dname} is MODERATE ({score:.3f}) — some steps may have yield/selectivity issues."
                else:
                    text = f"{dname} is LOW ({score:.3f}) — retrosynthesis found low-confidence or divergent routes."

            elif dim == "precursor_availability":
                if score > 0.65:
                    text = f"{dname} is HIGH ({score:.3f}) — building blocks commercially available."
                elif score > 0.40:
                    text = f"{dname} is MODERATE ({score:.3f}) — some rare or expensive reagents required."
                else:
                    text = f"{dname} is LOW ({score:.3f}) — key precursors unavailable or very expensive."

            elif dim == "scalability":
                if score > 0.65:
                    text = f"{dname} is HIGH ({score:.3f}) — suitable for gram to kilogram scale production."
                elif score > 0.40:
                    text = f"{dname} is MODERATE ({score:.3f}) — some cryogenic or hazardous steps."
                else:
                    text = f"{dname} is LOW ({score:.3f}) — serious industrial-scale concerns (hazardous intermediates, poor yield)."

            elif dim == "retro_confidence":
                if score > 0.65:
                    text = f"{dname} is HIGH ({score:.3f}) — AI retrosynthesis found confident, well-precedented routes."
                elif score > 0.40:
                    text = f"{dname} is MODERATE ({score:.3f}) — some disconnections lack literature precedent."
                else:
                    text = f"{dname} is LOW ({score:.3f}) — no reliable retrosynthetic pathway found."

            elif dim == "medchem_realism":
                if score > 0.65:
                    text = f"{dname} is HIGH ({score:.3f}) — clean structure, no PAINS or unstable motifs."
                elif score > 0.40:
                    text = f"{dname} is MODERATE ({score:.3f}) — some functional group incompatibilities."
                else:
                    text = f"{dname} is LOW ({score:.3f}) — problematic/reactive/dangerous functional groups present."
            else:
                text = f"{dname}: {score:.3f} ({tier})"

            explanations[dim] = text
        return explanations

    # ── Suggestions ───────────────────────────────────────────────────────────

    def _generate_suggestions(self, mol, props, dimensions, bottleneck) -> list:
        """Returns actionable improvement suggestions."""
        suggestions = []
        if not bottleneck:
            return suggestions

        score = dimensions.get(bottleneck, 0.5)
        if score > 0.55:
            return []  # No suggestions needed

        if bottleneck == "synthetic_complexity":
            if props["stereocenters"] > 3:
                suggestions.append("Consider reducing stereocenters or using a chiral pool approach.")
            if props["is_macrocycle"]:
                suggestions.append("Explore ring-opening macrolactonization or RCM strategies for macrocycle formation.")
            suggestions.append("Simplify the scaffold using bioisosteric replacements where possible.")

        elif bottleneck == "route_practicality":
            suggestions.append("Use convergent synthesis strategy to reduce longest linear sequence (LLS).")
            suggestions.append("Screen alternative disconnection points with ASKCOS or IBM RXN.")
            suggestions.append("Consider protecting group strategies to improve chemoselectivity.")

        elif bottleneck == "precursor_availability":
            suggestions.append("Search Sigma-Aldrich, Enamine, and Combi-Blocks for alternative building blocks.")
            suggestions.append("Consider de novo synthesis of key fragments from commodity chemicals.")
            if props["is_macrocycle"]:
                suggestions.append("Use linear precursor strategy and close the ring at the end.")

        elif bottleneck == "scalability":
            if props.get("has_azide") or props.get("has_diazo"):
                suggestions.append("Replace azide/diazo groups with safer surrogates for scale-up.")
            suggestions.append("Optimize reaction conditions for continuous flow chemistry where possible.")
            suggestions.append("Identify and replace any cryogenic or high-pressure steps early.")

        elif bottleneck == "retro_confidence":
            suggestions.append("Manually examine retrosynthetic tree with a medicinal chemist.")
            suggestions.append("Search SciFinder/Reaxys for analogous transformations.")
            suggestions.append("Consider fragment-based approach to reduce disconnection complexity.")

        elif bottleneck == "medchem_realism":
            suggestions.append("Run PAINS filter (Baell & Holloway) to identify assay interference.")
            suggestions.append("Check unstable functional groups (nitroso, peroxide) for stability under process conditions.")
            suggestions.append("Consider bioisosteric replacement of problematic groups.")

        return suggestions

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _tier(self, score: float) -> str:
        if score >= 0.65: return "HIGH"
        if score >= 0.40: return "MED"
        return "LOW"

    def _explain_modalities(self, modality_weights: dict) -> str:
        if not modality_weights:
            return ""
        top_name = max(modality_weights, key=modality_weights.get)
        top_val  = modality_weights[top_name]
        desc_map = {
            "ANN":      "fingerprint and physicochemical descriptors",
            "GAT":      "2D molecular graph topology",
            "ChemBERTa":"SMILES sequence (language model)",
            "EGNN":     "3D geometry and stereochemistry",
        }
        desc = desc_map.get(top_name, top_name)
        weights_str = " | ".join(
            f"{k}: {v:.2f}" for k, v in
            sorted(modality_weights.items(), key=lambda x: -x[1])
        )
        return f"{top_name} ({top_val:.0%}, {desc}) — [{weights_str}]"


# ══════════════════════════════════════════════════════════════════════════════
# GAT ATOM IMPORTANCE  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

class GATAtomImportance:
    """Extracts atom-level importance from GAT attention weights."""

    @staticmethod
    def extract_attention(gat_branch, graph_data) -> dict:
        hooks, attention_weights = [], []

        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights.append(output[1].detach().cpu())

        for name, module in gat_branch.named_modules():
            if hasattr(module, "att_src"):
                h = module.register_forward_hook(hook_fn)
                hooks.append(h)

        with torch.no_grad():
            gat_branch(graph_data)

        for h in hooks:
            h.remove()

        if not attention_weights:
            n = graph_data.x.size(0)
            return {"atom_scores": np.zeros(n), "num_atoms": n}

        last_attn = attention_weights[-1]
        edge_index = graph_data.edge_index.cpu().numpy()
        n_atoms    = graph_data.x.size(0)
        scores     = np.zeros(n_atoms)
        attn_avg   = last_attn.mean(dim=-1).numpy()

        for e in range(edge_index.shape[1]):
            scores[edge_index[1, e]] += attn_avg[e]

        if scores.max() > 0:
            scores = scores / scores.max()

        return {"atom_scores": scores, "num_atoms": n_atoms}


# ══════════════════════════════════════════════════════════════════════════════
# Backward-compat alias
# ══════════════════════════════════════════════════════════════════════════════
SynFeasExplainer = SynPractIQExplainer


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("SynPractIQ Explainer — smoke test")
    print("=" * 65)

    explainer = SynPractIQExplainer()

    # Simulate a predict() output for Aspirin
    mock_result = {
        "stage1_pass":   True,
        "stage1_prob":   0.95,
        "spi_score":     0.72,
        "spi_class":     3,
        "spi_label":     "practical",
        "spi_dimensions": {
            "synthetic_complexity":   0.85,
            "route_practicality":     0.78,
            "precursor_availability": 0.91,
            "scalability":            0.80,
            "retro_confidence":       0.65,
            "medchem_realism":        0.70,
        },
        "chemistry": {
            "molecular_weight": 180.2,
            "num_heavy_atoms":  13,
            "max_ring_size":    6,
            "num_stereocenters": 0,
        },
        "warning": "",
    }

    result = explainer.explain(
        "CC(=O)Oc1ccccc1C(=O)O",
        mock_result,
        {"ANN": 0.28, "GAT": 0.35, "ChemBERTa": 0.22, "EGNN": 0.15},
    )

    print(result["text"])
    print(f"\nFactors found: {len(result['factors'])}")
    print(f"Suggestions  : {result['suggestions']}")
    print(f"Bottleneck   : {result['bottleneck']}")
    print("\n✅ Explainer smoke test passed!")