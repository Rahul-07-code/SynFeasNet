"""
ANN Branch — SynFeasNet
=======================
Handles the numerical/fingerprint representation of molecules.

Input  : SMILES string
Process: ECFP4 Morgan Fingerprint (2048-bit) +
         208 RDKit physicochemical descriptors
         → concatenated 2256-dim vector
         → 3-layer Fully Connected Network
Output : 256-dimensional embedding vector

Why ANN here:
  - Input is a flat fixed-size numerical vector (2256 numbers)
  - No spatial structure (rules out CNN)
  - No sequence dependency (rules out RNN)
  - ANN learns non-linear combinations of chemical features
  - Fast to compute, stable to train
  - Captures global molecular properties effectively

KEY FIX vs original:
  - DescriptorExtractor now uses DATASET-WIDE mean/std normalization
    (fit on training data, applied consistently to val/test/inference)
  - Per-molecule z-score was wrong: it destroyed absolute scale info
    (MW=150 and MW=800 became indistinguishable after per-mol z-score)
  - Scaler can be saved/loaded so inference uses the same stats as training

SRS Reference: FR-2, Module 2, Section 2.6
"""

import os
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ══════════════════════════════════════════════════════════════════════════
# FINGERPRINT EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════

class FingerprintExtractor:
    """
    Computes ECFP4 Morgan Fingerprints (radius=2, 2048 bits).

    ECFP4 encodes the presence/absence of circular substructures
    up to radius 2 (i.e., the atom + up to 2 bonds away).
    These are binary vectors — no normalization needed.
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius  = radius
        self.n_bits  = n_bits
        self.generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    def compute(self, smiles: str) -> np.ndarray:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.n_bits, dtype=np.float32)
            fp = self.generator.GetFingerprint(mol)
            return np.array(fp, dtype=np.float32)
        except Exception:
            return np.zeros(self.n_bits, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# DESCRIPTOR EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════

class DescriptorExtractor:
    """
    Computes 208 RDKit physicochemical descriptors per molecule.

    IMPORTANT — Normalization strategy:
    ─────────────────────────────────────
    The original code normalized each molecule against ITSELF
    (subtract molecule's own mean, divide by molecule's own std).
    This was WRONG because:
      - It destroys absolute scale: MW=150 and MW=800 become identical
      - LogP=1.0 and LogP=8.0 become identical
      - Molecular properties that are informative BECAUSE of their
        absolute value (MW, ring count, HBD, HBA) lose that information

    The correct approach:
      1. Compute raw descriptors for the entire TRAINING set
      2. Compute mean + std across all training molecules (per descriptor)
      3. Use those training statistics to normalize every split
         (including validation, test, and inference)

    This is standard tabular ML pipeline practice. It ensures the model
    sees consistent scales across train/val/test and at inference time.

    Usage:
      extractor = DescriptorExtractor()
      extractor.fit(train_smiles_list)       # once, on training data only
      extractor.save_scaler("path/scaler.npz")  # persist for inference
      # --- later / inference ---
      extractor.load_scaler("path/scaler.npz")
      features = extractor.compute(smiles)   # uses loaded stats
    """

    def __init__(self):
        all_descriptors          = Descriptors.descList
        self.descriptor_names    = [name for name, _ in all_descriptors][:208]
        self.descriptor_fns      = [fn   for _, fn   in all_descriptors][:208]
        self.n_descriptors       = len(self.descriptor_names)

        # Dataset-wide normalization parameters — set by fit()
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None

        print(f"  DescriptorExtractor: using {self.n_descriptors} descriptors")

    # ──────────────────────────────────────────────────────────────────
    # RAW COMPUTE (no normalization)
    # ──────────────────────────────────────────────────────────────────

    def _raw_compute(self, smiles: str) -> np.ndarray:
        """
        Returns raw (un-normalized) descriptor values for one molecule.
        Replaces None / NaN / Inf with 0.0.
        Clips to [-1e6, 1e6] to handle extreme outliers before fitting.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.n_descriptors, dtype=np.float32)

            values = []
            for fn in self.descriptor_fns:
                try:
                    val = fn(mol)
                    if val is None or (isinstance(val, float) and
                                       (np.isnan(val) or np.isinf(val))):
                        val = 0.0
                    values.append(float(val))
                except Exception:
                    values.append(0.0)

            arr = np.array(values, dtype=np.float64)
            arr = np.clip(arr, -1e6, 1e6)
            return arr.astype(np.float32)

        except Exception:
            return np.zeros(self.n_descriptors, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # FIT  — call ONCE on the training set
    # ──────────────────────────────────────────────────────────────────

    def fit(self, smiles_list: list, verbose: bool = True) -> None:
        """
        Compute per-descriptor mean and std across the training set.

        Args:
            smiles_list : list of SMILES strings (training split only)
            verbose     : print progress every 5000 molecules
        """
        if verbose:
            print(f"  DescriptorExtractor.fit() — computing stats "
                  f"over {len(smiles_list)} training molecules...")

        all_raw = []
        for i, smi in enumerate(smiles_list):
            all_raw.append(self._raw_compute(smi))
            if verbose and (i + 1) % 5000 == 0:
                print(f"    {i+1}/{len(smiles_list)} molecules processed")

        all_raw = np.stack(all_raw, axis=0)        # (N, 208)
        self._mean = all_raw.mean(axis=0)          # (208,)
        self._std  = all_raw.std(axis=0) + 1e-6    # (208,) — avoid div/0

        if verbose:
            print(f"  DescriptorExtractor.fit() done. "
                  f"Mean range: [{self._mean.min():.3f}, {self._mean.max():.3f}]")

    # ──────────────────────────────────────────────────────────────────
    # SAVE / LOAD scaler stats
    # ──────────────────────────────────────────────────────────────────

    def save_scaler(self, path: str) -> None:
        """Save mean/std to disk so inference doesn't need to refit."""
        if self._mean is None:
            raise RuntimeError("Call fit() before save_scaler()")
        np.savez(path, mean=self._mean, std=self._std)
        print(f"  DescriptorExtractor scaler saved → {path}")

    def load_scaler(self, path: str) -> None:
        """Load mean/std from disk (for inference / val / test)."""
        data       = np.load(path)
        self._mean = data["mean"]
        self._std  = data["std"]
        print(f"  DescriptorExtractor scaler loaded ← {path}")

    # ──────────────────────────────────────────────────────────────────
    # COMPUTE  — normalized output
    # ──────────────────────────────────────────────────────────────────

    def compute(self, smiles: str) -> np.ndarray:
        """
        Returns dataset-normalized descriptor vector for one molecule.
        Requires fit() or load_scaler() to have been called first.
        Falls back to raw values if scaler not available (warns once).
        """
        raw = self._raw_compute(smiles)

        if self._mean is not None:
            return ((raw - self._mean) / self._std).astype(np.float32)

        # Scaler not fitted — return raw with a warning
        import warnings
        warnings.warn(
            "DescriptorExtractor.compute() called without fit() or "
            "load_scaler(). Returning raw (un-normalized) values. "
            "Call fit() on training data or load_scaler() for inference.",
            RuntimeWarning,
            stacklevel=2,
        )
        return raw


# ══════════════════════════════════════════════════════════════════════════
# ANN BRANCH
# ══════════════════════════════════════════════════════════════════════════

class ANNBranch(nn.Module):
    """
    3-layer Fully Connected Network over ECFP4 + descriptor features.

    Input : 2256-dim (2048 fingerprint + 208 descriptors)
    Output: 256-dim embedding vector
    """

    def __init__(self,
                 fp_dim:   int   = 2048,
                 desc_dim: int   = 208,
                 dropout:  float = 0.3):
        super(ANNBranch, self).__init__()

        input_dim = fp_dim + desc_dim   # 2256

        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                        nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR  (combines fingerprint + descriptors)
# ══════════════════════════════════════════════════════════════════════════

class ANNFeatureExtractor:
    """
    Combines FingerprintExtractor + DescriptorExtractor into one object.

    IMPORTANT: Call fit_descriptors() on training SMILES before use,
    or load_descriptor_scaler() for inference.
    """

    def __init__(self):
        self.fp_extractor   = FingerprintExtractor(radius=2, n_bits=2048)
        self.desc_extractor = DescriptorExtractor()
        self.input_dim      = (self.fp_extractor.n_bits +
                               self.desc_extractor.n_descriptors)
        print(f"  ANNFeatureExtractor ready | input_dim={self.input_dim}")

    # ── Scaler delegation ─────────────────────────────────────────────

    def fit_descriptors(self, smiles_list: list) -> None:
        """Fit descriptor scaler on training data. Call once."""
        self.desc_extractor.fit(smiles_list)

    def save_descriptor_scaler(self, path: str) -> None:
        self.desc_extractor.save_scaler(path)

    def load_descriptor_scaler(self, path: str) -> None:
        self.desc_extractor.load_scaler(path)

    # ── Feature computation ───────────────────────────────────────────

    def compute(self, smiles: str) -> np.ndarray:
        fp   = self.fp_extractor.compute(smiles)
        desc = self.desc_extractor.compute(smiles)
        return np.concatenate([fp, desc], axis=0)

    def compute_batch(self, smiles_list: list) -> np.ndarray:
        return np.stack([self.compute(s) for s in smiles_list])


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Testing ANN Branch (with dataset-wide normalization fix)")
    print("=" * 65)

    train_smiles = [
        "CC1NC(=O)C(C(O)CC)N(C)C(=O)C(CC(C)C)NC(=O)c2csc(n2)C(C)C",
        "CC(=O)NC1C(O)C(OC2C(NC(C)=O)C(O)C(OC3OC(CO)C(O)C(O)C3O)OC2CO)OC(CO)C1O",
        "O=C1CCCCCCCCCCCCC(=O)NCCCCN1",
        "CC1(C)CCC(=O)NCC(=O)NCCCCC(=O)NCC(=O)O1",
    ]

    test_smiles = train_smiles[:2]

    print("\n1. Fitting descriptor scaler on 'training' SMILES...")
    extractor = ANNFeatureExtractor()
    extractor.fit_descriptors(train_smiles)

    print("\n2. Testing compute() with fitted scaler...")
    for smi in test_smiles:
        feat = extractor.compute(smi)
        print(f"   Shape: {feat.shape} | FP non-zero: {feat[:2048].sum():.0f} "
              f"| Desc mean: {feat[2048:].mean():.3f} "
              f"| Desc std: {feat[2048:].std():.3f}")

    print("\n3. Testing save/load scaler...")
    extractor.save_descriptor_scaler("/tmp/test_scaler.npz")
    extractor2 = ANNFeatureExtractor()
    extractor2.load_descriptor_scaler("/tmp/test_scaler.npz")
    feat_orig    = extractor.compute(test_smiles[0])
    feat_loaded  = extractor2.compute(test_smiles[0])
    assert np.allclose(feat_orig, feat_loaded, atol=1e-6), \
        "Save/load produced different values!"
    print("   Save/load round-trip: PASS")

    print("\n4. Testing ANNBranch forward pass...")
    model    = ANNBranch()
    features = extractor.compute_batch(test_smiles)
    x        = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    print(f"   Output shape: {out.shape}   (expected [2, 256])")
    assert out.shape == (2, 256)

    print("\n✅ ANN Branch with dataset-wide normalization: all checks passed!")