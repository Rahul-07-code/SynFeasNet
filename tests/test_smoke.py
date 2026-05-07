"""
Smoke Tests — SynFeasNet v2
==============================
Verifies all imports work and model forward passes produce correct shapes.

Run: python -m pytest tests/test_smoke.py -v
Or:  python tests/test_smoke.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ══════════════════════════════════════════════════════════════════════════

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
    from models.attention_fusion import SynFeasNetV2, AttentionFusion, OutputHead
    print("[OK] SynFeasNetV2 imports")


def test_import_calibration():
    from models.calibration import TemperatureScaling, compute_ece
    print("[OK] TemperatureScaling imports")


def test_import_explainability():
    from explainability.explainer import SynFeasExplainer, GATAtomImportance
    print("[OK] Explainability imports")


def test_import_retrosynthesis():
    from retrosynthesis.providers import (
        RetrosynthesisRouter, MockRetrosynthesisProvider,
        IBMRXNProvider, ASKCOSProvider,
    )
    print("[OK] Retrosynthesis imports")


def test_import_api():
    from api.fastapi_app import app
    print("[OK] FastAPI app imports")


# ══════════════════════════════════════════════════════════════════════════
# SHAPE TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_ann_forward():
    import torch
    from models.ann_branch import ANNBranch
    model = ANNBranch()
    x = torch.randn(2, 2256)
    out = model(x)
    assert out.shape == (2, 256), f"ANNBranch shape: {out.shape}"
    print("[OK] ANNBranch forward: (2, 2256) -> (2, 256)")


def test_gat_forward():
    import torch
    from models.gat_branch import GATBranch, GraphBuilder
    from torch_geometric.data import Batch

    builder = GraphBuilder()
    graphs = [builder.build("CCO"), builder.build("c1ccccc1")]
    batch = Batch.from_data_list(graphs)

    model = GATBranch()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 256), f"GATBranch shape: {out.shape}"
    print("[OK] GATBranch forward: 2 graphs -> (2, 256)")


def test_egnn_forward():
    import torch
    from models.gat_branch import GraphBuilder
    from models.egnn_branch import EGNNBranch, Graph3DBuilder
    from torch_geometric.data import Batch

    gb = GraphBuilder()
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
    print("[OK] EGNNBranch forward: 2 graphs+coords -> (2, 256)")


def test_calibration():
    import torch
    from models.calibration import TemperatureScaling, compute_ece
    import numpy as np

    cal = TemperatureScaling()
    logits = torch.randn(100)
    labels = (torch.rand(100) > 0.5).float()
    temp = cal.calibrate(logits, labels)
    assert temp > 0, "Temperature must be positive"

    probs = torch.sigmoid(logits).numpy()
    ece = compute_ece(probs, labels.numpy())
    assert 0 <= ece <= 1, f"ECE out of range: {ece}"
    print(f"[OK] Calibration: T={temp:.3f}, ECE={ece:.4f}")


def test_explainer():
    from explainability.explainer import SynFeasExplainer
    explainer = SynFeasExplainer()
    result = explainer.explain("CC(=O)Oc1ccccc1C(=O)O", 0.92, 0.5)
    assert "text" in result
    assert "factors" in result
    print(f"[OK] Explainer: {len(result['factors'])} factors")


def test_retrosynthesis():
    from retrosynthesis.providers import RetrosynthesisRouter
    router = RetrosynthesisRouter()
    result = router.analyze("CC(=O)Oc1ccccc1C(=O)O", max_steps=3)
    d = result.to_dict()
    assert d["provider"] == "mock"
    assert d["success"] is True
    print(f"[OK] Retrosynthesis: {d['num_steps']} steps via {d['provider']}")


# ══════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("SynFeasNet v2 — Smoke Tests")
    print("=" * 65)

    tests = [
        ("Import: ANN", test_import_ann),
        ("Import: GAT", test_import_gat),
        ("Import: ChemBERTa", test_import_chemberta),
        ("Import: EGNN", test_import_egnn),
        ("Import: Fusion", test_import_fusion),
        ("Import: Calibration", test_import_calibration),
        ("Import: Explainability", test_import_explainability),
        ("Import: Retrosynthesis", test_import_retrosynthesis),
        ("Forward: ANN", test_ann_forward),
        ("Forward: GAT", test_gat_forward),
        ("Forward: EGNN", test_egnn_forward),
        ("Calibration", test_calibration),
        ("Explainer", test_explainer),
        ("Retrosynthesis", test_retrosynthesis),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            print(f"\n--- {name} ---")
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 65}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 65)
