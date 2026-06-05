"""
test_smoke.py — SynPractIQ v3
================================
Smoke Tests for all modules of the Synthetic Practicality Intelligence System.

Run:  python -m pytest tests/test_smoke.py -v
Or:   python tests/test_smoke.py
"""

import sys, os
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_import_ann():
    from models.ann_branch import ANNBranch, ANNFeatureExtractor
    print("[OK] ANNBranch imports")


def test_import_gat():
    from models.gat_branch import GATBranch, GraphBuilder
    print("[OK] GATBranch imports")


def test_import_chemberta():
    from models.chemBERTa_branch import ChemBERTaBranch, SMILESTokenizer
    print("[OK] ChemBERTaBranch imports")


def test_import_egnn():
    from models.egnn_branch import EGNNBranch, Graph3DBuilder, ConformerGenerator
    print("[OK] EGNNBranch imports")


def test_import_fusion():
    from models.attention_fusion import (
        SynPractIQModel, AttentionFusion, SPIOutputHead,
        SPI_DIMENSION_NAMES, SynFeasNetV2,   # backward-compat alias
    )
    assert len(SPI_DIMENSION_NAMES) == 6, f"Expected 6 SPI dims, got {len(SPI_DIMENSION_NAMES)}"
    print(f"[OK] SynPractIQModel imports | {len(SPI_DIMENSION_NAMES)} SPI dimensions")


def test_import_spi_labels():
    from spi_labels import SPILabelGenerator, SPI_WEIGHTS, FEASIBILITY_GATE_THRESHOLD
    assert abs(sum(SPI_WEIGHTS.values()) - 1.0) < 1e-6, "SPI weights must sum to 1.0"
    print(f"[OK] SPILabelGenerator imports | gate_threshold={FEASIBILITY_GATE_THRESHOLD}")


def test_import_calibration():
    from models.calibration import TemperatureScaling, compute_ece
    print("[OK] TemperatureScaling imports")


def test_import_explainability():
    from explainability.explainer import SynPractIQExplainer, GATAtomImportance
    print("[OK] SynPractIQExplainer imports")


def test_import_retrosynthesis():
    from retrosynthesis.providers import (
        RetrosynthesisRouter, MockRetrosynthesisProvider,
        IBMRXNProvider, ASKCOSProvider,
    )
    print("[OK] Retrosynthesis imports")


# ══════════════════════════════════════════════════════════════════════════════
# SPI LABEL GENERATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_spi_label_generation():
    """Test SPILabelGenerator on a mini synthetic DataFrame."""
    import pandas as pd
    from spi_labels import SPILabelGenerator, SPI_DIMENSION_NAMES

    # Build a minimal DataFrame mimicking dataset.csv columns
    df = pd.DataFrame({
        "smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin — practical
            "CCO",                             # Ethanol — trivial
            "CC1NC(=O)C(C(O)CC)N(C)C(=O)C", # Complex — harder
        ],
        "split": ["train", "train", "val"],
        "sa_score_proxy":              [3.2,  1.5,  8.5],
        "stereo_burden":               [0.0,  0.0,  1.0],
        "ring_complexity":             [0.6,  0.0,  0.8],
        "bertz_complexity_proxy":      [200,  20,   600],
        "pg_steps_estimate":           [1,    0,    5],
        "fg_difficulty_score":         [0.2,  0.0,  0.7],
        "route_success_prob":          [0.5,  0.9,  0.1],
        "route_convergence":           [0.3,  0.8,  0.0],
        "route_confidence_mean":       [0.6,  0.9,  0.2],
        "route_confidence_std":        [0.2,  0.1,  0.5],
        "chemoselectivity_burden":     [0.3,  0.0,  0.8],
        "precursor_buyability":        [0.8,  1.0,  0.1],
        "reaction_template_coverage":  [0.7,  0.9,  0.3],
        "precursor_risk_score":        [0.3,  0.0,  0.9],
        "manufacturability_score":     [0.7,  0.9,  0.3],
        "scale_up_risk":               [0.2,  0.0,  0.7],
        "intermediate_instability_score": [0.2, 0.0, 0.6],
        "purification_complexity":     [0.6,  0.2,  0.9],
        "n_routes_found":              [3,    5,    0],
        "best_route_depth":            [4,    1,    30],
        "retro_branching_factor":      [1.5,  1.0,  4.0],
        "epistemic_uncertainty":       [0.3,  0.1,  0.8],
        "aleatoric_uncertainty":       [0.2,  0.1,  0.7],
        "fg_incompatibility_v2":       [0.1,  0.0,  0.5],
        "fg_complexity_count":         [2,    0,    6],
        "catalyst_dependence":         [0.1,  0.0,  0.3],
        "label_uncertainty_v2":        [0.2,  0.05, 0.6],
        "sample_weight":               [1.0,  1.0,  0.5],
        "uncertainty_downweight":      [0.0,  0.0,  0.3],
        # Dangerous FG flags
        "fg_azide": [0, 0, 0], "fg_diazo": [0, 0, 0],
        "fg_peroxide": [0, 0, 0], "fg_nitroso": [0, 0, 0],
        "fg_isocyanate": [0, 0, 0], "fg_isothiocyanate": [0, 0, 0],
        "fg_acyl_halide": [0, 0, 0],
    })

    gen    = SPILabelGenerator()
    df_out = gen.generate(df)

    # Check all expected columns present
    for dim in SPI_DIMENSION_NAMES:
        col = f"spi_{dim}"
        assert col in df_out.columns, f"Missing column: {col}"
        assert df_out[col].between(0, 1).all(), f"{col} out of [0,1]"

    assert "spi_score"        in df_out.columns
    assert "spi_class"        in df_out.columns
    assert "spi_label"        in df_out.columns
    assert "stage1_pass"      in df_out.columns
    assert "spi_sample_weight" in df_out.columns

    # Sanity: Ethanol (trivial) should score higher than complex molecule
    ethanol_spi  = df_out.loc[df_out["smiles"] == "CCO", "spi_score"].values[0]
    complex_spi  = df_out.loc[df_out["smiles"].str.startswith("CC1NC"), "spi_score"].values[0]
    assert ethanol_spi > complex_spi, (
        f"Expected Ethanol SPI ({ethanol_spi:.3f}) > complex ({complex_spi:.3f})"
    )
    print(f"[OK] SPILabelGenerator: Aspirin={df_out['spi_score'].iloc[0]:.3f} "
          f"Ethanol={ethanol_spi:.3f} Complex={complex_spi:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# BRANCH FORWARD PASS TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_ann_forward():
    from models.ann_branch import ANNBranch
    model = ANNBranch()
    x = torch.randn(2, 2256)
    out = model(x)
    assert out.shape == (2, 256), f"ANNBranch shape: {out.shape}"
    print("[OK] ANNBranch forward: (2, 2256) → (2, 256)")


def test_gat_forward():
    from models.gat_branch import GATBranch, GraphBuilder
    from torch_geometric.data import Batch
    builder = GraphBuilder()
    graphs  = [builder.build("CCO"), builder.build("c1ccccc1")]
    batch   = Batch.from_data_list(graphs)
    model   = GATBranch()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 256), f"GATBranch shape: {out.shape}"
    print("[OK] GATBranch forward: 2 graphs → (2, 256)")


def test_egnn_forward():
    from models.gat_branch  import GraphBuilder
    from models.egnn_branch import EGNNBranch, Graph3DBuilder
    from torch_geometric.data import Batch

    gb  = GraphBuilder()
    g3d = Graph3DBuilder()
    graphs = []
    for smi in ["CCO", "c1ccccc1"]:
        g = gb.build(smi)
        g = g3d.add_coords(g, smi)
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    model = EGNNBranch()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 256), f"EGNNBranch shape: {out.shape}"
    print("[OK] EGNNBranch forward: 2 graphs+coords → (2, 256)")


# ══════════════════════════════════════════════════════════════════════════════
# FUSION + SPI OUTPUT HEAD TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_spi_output_head():
    from models.attention_fusion import SPIOutputHead, SPI_DIMENSION_NAMES
    head = SPIOutputHead(input_dim=256, hidden_dim=128)
    head.eval()
    x = torch.randn(4, 256)
    with torch.no_grad():
        out = head(x)
    assert out["sub_scores"].shape   == (4, len(SPI_DIMENSION_NAMES)), \
        f"sub_scores shape: {out['sub_scores'].shape}"
    assert out["spi_score"].shape    == (4, 1)
    assert out["stage1_logit"].shape == (4, 1)
    # Sub-scores must be in [0, 1] (sigmoid applied)
    assert out["sub_scores"].min() >= 0.0 and out["sub_scores"].max() <= 1.0
    print(f"[OK] SPIOutputHead: sub_scores{out['sub_scores'].shape} "
          f"spi{out['spi_score'].shape} gate{out['stage1_logit'].shape}")


def test_synpractiq_forward():
    """Full forward pass through SynPractIQModel."""
    from models.ann_branch       import ANNFeatureExtractor
    from models.gat_branch       import GraphBuilder
    from models.chemBERTa_branch import SMILESTokenizer
    from models.egnn_branch      import Graph3DBuilder
    from models.attention_fusion import SynPractIQModel, SPI_DIMENSION_NAMES
    from torch_geometric.data    import Batch

    test_smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCO"]

    ann_ext = ANNFeatureExtractor()
    ann_ext.fit_descriptors(test_smiles)
    gb  = GraphBuilder()
    tok = SMILESTokenizer(max_length=128)
    g3d = Graph3DBuilder()

    ann_feats = torch.tensor(ann_ext.compute_batch(test_smiles), dtype=torch.float32)
    tokens    = tok(test_smiles)
    graphs    = [g3d.add_coords(gb.build(s), s) for s in test_smiles]
    batch_g   = Batch.from_data_list(graphs)

    model = SynPractIQModel()
    model.eval()
    with torch.no_grad():
        out = model(ann_feats, batch_g, tokens["input_ids"], tokens["attention_mask"])

    assert out["sub_scores"].shape   == (2, len(SPI_DIMENSION_NAMES))
    assert out["spi_score"].shape    == (2, 1)
    assert out["stage1_logit"].shape == (2, 1)

    print(f"[OK] SynPractIQModel forward: sub_scores{out['sub_scores'].shape} "
          f"spi{out['spi_score'].shape}")
    print(f"     Modality weights: {model.fusion.get_modality_weights()}")


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_calibration():
    from models.calibration import TemperatureScaling, compute_ece
    cal    = TemperatureScaling()
    logits = torch.randn(100)
    labels = (torch.rand(100) > 0.5).float()
    temp   = cal.calibrate(logits, labels)
    assert temp > 0, "Temperature must be positive"

    probs = torch.sigmoid(logits).numpy()
    ece   = compute_ece(probs, labels.numpy())
    assert 0 <= ece <= 1, f"ECE out of range: {ece}"
    print(f"[OK] Calibration: T={temp:.3f}, ECE={ece:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINER TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_explainer():
    from explainability.explainer import SynPractIQExplainer

    mock_result = {
        "stage1_pass": True,
        "stage1_prob": 0.92,
        "spi_score":   0.72,
        "spi_class":   3,
        "spi_label":   "practical",
        "spi_dimensions": {
            "synthetic_complexity":   0.85,
            "route_practicality":     0.78,
            "precursor_availability": 0.91,
            "scalability":            0.80,
            "retro_confidence":       0.65,
            "medchem_realism":        0.20,   # intentionally low
        },
        "chemistry": {"molecular_weight": 180.2, "num_heavy_atoms": 13,
                       "max_ring_size": 6, "num_stereocenters": 0},
        "warning": "",
    }

    explainer = SynPractIQExplainer()
    result = explainer.explain("CC(=O)Oc1ccccc1C(=O)O", mock_result,
                               {"ANN": 0.28, "GAT": 0.35, "ChemBERTa": 0.22, "EGNN": 0.15})

    assert "text"       in result
    assert "factors"    in result
    assert "suggestions" in result
    assert "bottleneck" in result
    assert result["bottleneck"] == "medchem_realism", \
        f"Expected bottleneck=medchem_realism, got {result['bottleneck']}"
    print(f"[OK] Explainer: bottleneck={result['bottleneck']} "
          f"factors={len(result['factors'])} suggestions={len(result['suggestions'])}")


# ══════════════════════════════════════════════════════════════════════════════
# RETROSYNTHESIS TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_retrosynthesis():
    from retrosynthesis.providers import RetrosynthesisRouter
    router = RetrosynthesisRouter()
    result = router.analyze("CC(=O)Oc1ccccc1C(=O)O", max_steps=3)
    d = result.to_dict()
    assert d["provider"] == "mock"
    assert d["success"]  is True
    print(f"[OK] Retrosynthesis: {d['num_steps']} steps via {d['provider']}")


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SynPractIQ v3 — Smoke Tests")
    print("=" * 70)

    tests = [
        # Import tests
        ("Import: ANNBranch",          test_import_ann),
        ("Import: GATBranch",          test_import_gat),
        ("Import: ChemBERTa",          test_import_chemberta),
        ("Import: EGNN",               test_import_egnn),
        ("Import: SynPractIQModel",    test_import_fusion),
        ("Import: SPILabelGenerator",  test_import_spi_labels),
        ("Import: Calibration",        test_import_calibration),
        ("Import: Explainability",     test_import_explainability),
        ("Import: Retrosynthesis",     test_import_retrosynthesis),
        # Label generation
        ("SPI Labels",                 test_spi_label_generation),
        # Branch forward passes
        ("Forward: ANN",               test_ann_forward),
        ("Forward: GAT",               test_gat_forward),
        ("Forward: EGNN",              test_egnn_forward),
        # Fusion + output head
        ("SPIOutputHead",              test_spi_output_head),
        ("Forward: SynPractIQModel",   test_synpractiq_forward),
        # Other modules
        ("Calibration",                test_calibration),
        ("Explainer",                  test_explainer),
        ("Retrosynthesis",             test_retrosynthesis),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            print(f"\n--- {name} ---")
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)