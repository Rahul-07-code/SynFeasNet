"""
spi_labels.py — SynPractIQ (Synthetic Practicality Intelligence System)
=========================================================================
Derives multi-dimensional Synthetic Practicality Index (SPI) targets
directly from the rich dataset columns already present in dataset.csv.

SPI Dimensions (6 sub-scores, each in [0, 1]):
  1. synthetic_complexity    — lower = simpler to synthesize
  2. route_practicality      — higher = more robust routes
  3. precursor_availability  — higher = easier to source
  4. scalability             — higher = more industrially viable
  5. retro_confidence        — higher = AI retrosynthesis is confident
  6. medchem_realism         — higher = more chemically sound

Final SPI = weighted average of the 6 sub-scores.

All sub-scores are oriented so that HIGHER = MORE PRACTICAL.

Usage:
    from spi_labels import SPILabelGenerator
    gen = SPILabelGenerator()
    df_with_spi = gen.generate(df)
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# ── Dimension weights (sum to 1.0) ───────────────────────────────────────────
SPI_WEIGHTS = {
    "synthetic_complexity":   0.20,
    "route_practicality":     0.20,
    "precursor_availability": 0.18,
    "scalability":            0.17,
    "retro_confidence":       0.15,
    "medchem_realism":        0.10,
}

# SPI thresholds for categorical labels
SPI_CLASS_THRESHOLDS = {
    "trivial":    (0.75, 1.01),   # class 4
    "practical":  (0.55, 0.75),   # class 3
    "challenging":(0.35, 0.55),   # class 2
    "difficult":  (0.15, 0.35),   # class 1
    "intractable":(0.00, 0.15),   # class 0
}

FEASIBILITY_GATE_THRESHOLD = 0.25   # Stage 1 gate: SPI < this → filtered out


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _safe(series: pd.Series) -> pd.Series:
    """Replace NaN/Inf with column median (or 0 if all NaN)."""
    s = series.copy().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    median = s.median()
    if np.isnan(median):
        median = 0.0
    return s.fillna(median)


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(0.0, 1.0)


def _invert(series: pd.Series) -> pd.Series:
    """Invert a [0,1] score so that lower-is-harder becomes higher-is-better."""
    return 1.0 - _clip01(series)


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return _clip01((series - mn) / (mx - mn))


# ══════════════════════════════════════════════════════════════════════════════
# SPI LABEL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class SPILabelGenerator:
    """
    Derives six SPI sub-scores and the composite SPI from dataset columns.

    The dataset.csv already contains all necessary computed features;
    this class just assembles and re-weights them correctly.
    """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add SPI columns to df. Returns a copy.

        New columns added:
          spi_synthetic_complexity    float [0,1]  higher=simpler
          spi_route_practicality      float [0,1]
          spi_precursor_availability  float [0,1]
          spi_scalability             float [0,1]
          spi_retro_confidence        float [0,1]
          spi_medchem_realism         float [0,1]
          spi_score                   float [0,1]  composite SPI
          spi_class                   int   {0,1,2,3,4}
          spi_label                   str
          spi_sample_weight           float [0,1]  training reliability weight
          stage1_pass                 bool  synthesizability gate
        """
        df = df.copy()
        df = self._validate_smiles(df)

        df["spi_synthetic_complexity"]   = self._dim_synthetic_complexity(df)
        df["spi_route_practicality"]     = self._dim_route_practicality(df)
        df["spi_precursor_availability"] = self._dim_precursor_availability(df)
        df["spi_scalability"]            = self._dim_scalability(df)
        df["spi_retro_confidence"]       = self._dim_retro_confidence(df)
        df["spi_medchem_realism"]        = self._dim_medchem_realism(df)

        df["spi_score"] = self._composite_spi(df)
        df["spi_class"] = df["spi_score"].apply(self._classify)
        df["spi_label"] = df["spi_score"].apply(self._label)
        df["spi_sample_weight"] = self._sample_weights(df)

        # Stage 1 gate: only molecules above FEASIBILITY_GATE_THRESHOLD pass
        df["stage1_pass"] = (df["spi_score"] >= FEASIBILITY_GATE_THRESHOLD) & df["valid_smiles"]

        self._print_stats(df)
        return df

    # ── Stage 1: SMILES validation ────────────────────────────────────────────

    def _validate_smiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic chemical validity check using RDKit."""
        smiles_col = "smiles" if "smiles" in df.columns else "smiles_canonical"
        valid = []
        for smi in df[smiles_col].astype(str):
            try:
                mol = Chem.MolFromSmiles(smi)
                valid.append(mol is not None and mol.GetNumAtoms() > 0)
            except Exception:
                valid.append(False)
        df["valid_smiles"] = valid
        n_invalid = sum(1 for v in valid if not v)
        if n_invalid > 0:
            print(f"  [SPILabelGenerator] {n_invalid} invalid SMILES found → stage1_pass=False")
        return df

    # ── Dimension 1: Synthetic Complexity (higher = simpler) ─────────────────

    def _dim_synthetic_complexity(self, df: pd.DataFrame) -> pd.Series:
        """
        Combines SA score, stereochemical burden, ring complexity,
        Bertz complexity, and protecting-group burden.
        All inverted so higher = simpler.
        """
        # sa_score_proxy: 1=easy, 10=hard → normalize and invert
        sa = _safe(df.get("sa_score_proxy", pd.Series(5.0, index=df.index)))
        sa_norm = _invert(_clip01((sa - 1.0) / 9.0))  # 1→1.0, 10→0.0

        stereo = _invert(_clip01(_safe(df.get("stereo_burden", pd.Series(0, index=df.index)))))
        ring = _invert(_clip01(_safe(df.get("ring_complexity", pd.Series(0.5, index=df.index)))))
        bertz = _invert(_normalize(_safe(df.get("bertz_complexity_proxy", pd.Series(100, index=df.index)))))
        pg = _invert(_clip01(_safe(df.get("pg_steps_estimate", pd.Series(0, index=df.index))) / 10.0))
        fg_diff = _invert(_clip01(_safe(df.get("fg_difficulty_score", pd.Series(0, index=df.index)))))

        score = (sa_norm * 0.30 + stereo * 0.20 + ring * 0.15 +
                 bertz * 0.15 + pg * 0.10 + fg_diff * 0.10)
        return _clip01(score)

    # ── Dimension 2: Route Practicality (higher = more robust) ───────────────

    def _dim_route_practicality(self, df: pd.DataFrame) -> pd.Series:
        route_succ = _clip01(_safe(df.get("route_success_prob", pd.Series(0.1, index=df.index))))
        convergence = _clip01(_safe(df.get("route_convergence", pd.Series(0, index=df.index))))
        conf_mean = _clip01(_safe(df.get("route_confidence_mean", pd.Series(0.3, index=df.index))))
        # chemoselectivity_burden: higher = harder → invert
        chemo = _invert(_clip01(_safe(df.get("chemoselectivity_burden", pd.Series(0.5, index=df.index)))))
        # conf_std: higher = more uncertain → invert
        conf_std = _invert(_clip01(_safe(df.get("route_confidence_std", pd.Series(0.3, index=df.index)))))

        score = (route_succ * 0.35 + convergence * 0.20 + conf_mean * 0.25 +
                 chemo * 0.10 + conf_std * 0.10)
        return _clip01(score)

    # ── Dimension 3: Precursor Availability (higher = easier to source) ───────

    def _dim_precursor_availability(self, df: pd.DataFrame) -> pd.Series:
        buyability = _clip01(_safe(df.get("precursor_buyability", pd.Series(0.2, index=df.index))))
        template_cov = _clip01(_safe(df.get("reaction_template_coverage", pd.Series(0.5, index=df.index))))
        # precursor_risk_score: higher risk = harder → invert
        risk = _invert(_clip01(_safe(df.get("precursor_risk_score", pd.Series(0.7, index=df.index)))))

        score = buyability * 0.45 + template_cov * 0.30 + risk * 0.25
        return _clip01(score)

    # ── Dimension 4: Scalability (higher = more industrially viable) ──────────

    def _dim_scalability(self, df: pd.DataFrame) -> pd.Series:
        mfg = _clip01(_safe(df.get("manufacturability_score", pd.Series(0.5, index=df.index))))
        # scale_up_risk: higher = harder → invert
        scaleup = _invert(_clip01(_safe(df.get("scale_up_risk", pd.Series(0.2, index=df.index)))))
        # intermediate_instability: higher = harder → invert
        instability = _invert(_clip01(_safe(df.get("intermediate_instability_score", pd.Series(0.5, index=df.index)))))
        # purification_complexity: higher = harder → invert
        purif = _invert(_clip01(_safe(df.get("purification_complexity", pd.Series(0.8, index=df.index)))))

        score = mfg * 0.35 + scaleup * 0.25 + instability * 0.20 + purif * 0.20
        return _clip01(score)

    # ── Dimension 5: Retrosynthesis Confidence (higher = AI is confident) ─────

    def _dim_retro_confidence(self, df: pd.DataFrame) -> pd.Series:
        # n_routes_found: more routes = easier; normalize to [0,1]
        n_routes = _safe(df.get("n_routes_found", pd.Series(0, index=df.index)))
        routes_norm = _clip01(n_routes / 5.0)  # 5+ routes = max confidence

        # best_route_depth: fewer steps = easier; 1=best(1.0), 30=worst(0.0)
        depth = _safe(df.get("best_route_depth", pd.Series(30, index=df.index)))
        depth_score = _clip01(1.0 - (depth - 1.0) / 29.0)

        # retro_branching_factor: higher = more complex → invert
        branching = _invert(_clip01(_safe(df.get("retro_branching_factor", pd.Series(2, index=df.index))) / 5.0))

        # epistemic_uncertainty: higher = less confident → invert
        epist = _invert(_clip01(_safe(df.get("epistemic_uncertainty", pd.Series(0.5, index=df.index)))))

        # aleatoric_uncertainty: higher = less confident → invert
        aleat = _invert(_clip01(_safe(df.get("aleatoric_uncertainty", pd.Series(0.3, index=df.index)))))

        score = (routes_norm * 0.30 + depth_score * 0.25 + branching * 0.15 +
                 epist * 0.15 + aleat * 0.15)
        return _clip01(score)

    # ── Dimension 6: Medicinal Chemistry Realism (higher = cleaner) ───────────

    def _dim_medchem_realism(self, df: pd.DataFrame) -> pd.Series:
        # fg_incompatibility_v2: higher = worse → invert
        fg_incompat = _invert(_clip01(_safe(df.get("fg_incompatibility_v2", pd.Series(0, index=df.index)))))
        # fg_complexity_count: more complex FGs = harder → invert, normalize
        fg_count = _invert(_clip01(_safe(df.get("fg_complexity_count", pd.Series(2, index=df.index))) / 10.0))
        # catalyst_dependence: higher = harder → invert
        cat = _invert(_clip01(_safe(df.get("catalyst_dependence", pd.Series(0, index=df.index)))))
        # label_uncertainty_v2: higher = less reliable → invert
        uncert = _invert(_clip01(_safe(df.get("label_uncertainty_v2", pd.Series(0.3, index=df.index)))))

        # Penalize dangerous functional groups (all are binary 0/1 in dataset)
        danger_fgs = ["fg_azide", "fg_diazo", "fg_peroxide", "fg_nitroso",
                      "fg_isocyanate", "fg_isothiocyanate", "fg_acyl_halide"]
        danger_score = pd.Series(0.0, index=df.index)
        for fg in danger_fgs:
            if fg in df.columns:
                danger_score += _safe(df[fg])
        danger_penalty = _invert(_clip01(danger_score / max(len(danger_fgs), 1)))

        score = (fg_incompat * 0.25 + fg_count * 0.20 + cat * 0.15 +
                 uncert * 0.15 + danger_penalty * 0.25)
        return _clip01(score)

    # ── Composite SPI ─────────────────────────────────────────────────────────

    def _composite_spi(self, df: pd.DataFrame) -> pd.Series:
        spi = (
            df["spi_synthetic_complexity"]   * SPI_WEIGHTS["synthetic_complexity"] +
            df["spi_route_practicality"]     * SPI_WEIGHTS["route_practicality"] +
            df["spi_precursor_availability"] * SPI_WEIGHTS["precursor_availability"] +
            df["spi_scalability"]            * SPI_WEIGHTS["scalability"] +
            df["spi_retro_confidence"]       * SPI_WEIGHTS["retro_confidence"] +
            df["spi_medchem_realism"]        * SPI_WEIGHTS["medchem_realism"]
        )
        return _clip01(spi)

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, spi: float) -> int:
        """Returns 0 (intractable) to 4 (trivial)."""
        for cls_int, (label, (lo, hi)) in enumerate(SPI_CLASS_THRESHOLDS.items()):
            if lo <= spi < hi:
                return 4 - cls_int  # reverse so 4=trivial
        return 0

    def _label(self, spi: float) -> str:
        for name, (lo, hi) in SPI_CLASS_THRESHOLDS.items():
            if lo <= spi < hi:
                return name
        return "intractable"

    # ── Sample weights ────────────────────────────────────────────────────────

    def _sample_weights(self, df: pd.DataFrame) -> pd.Series:
        """
        Training reliability weight:
          - source_weight from dataset (ChEMBL=1.0, heuristic=0.5 etc.)
          - downweighted by label_uncertainty_v2
          - downweighted further if invalid SMILES
        """
        base = _clip01(_safe(df.get("sample_weight", pd.Series(1.0, index=df.index))))
        uncert_down = _clip01(_safe(df.get("uncertainty_downweight", pd.Series(0.0, index=df.index))))
        weight = base * (1.0 - 0.5 * uncert_down)
        weight = weight * df["valid_smiles"].astype(float)
        return _clip01(weight)

    # ── Statistics ────────────────────────────────────────────────────────────

    def _print_stats(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("SPI Label Generation Statistics")
        print("=" * 60)
        print(f"  Total molecules   : {len(df):,}")
        print(f"  Valid SMILES      : {df['valid_smiles'].sum():,}")
        print(f"  Stage 1 pass      : {df['stage1_pass'].sum():,}")
        print(f"\n  SPI Score distribution:")
        print(f"    mean = {df['spi_score'].mean():.4f}")
        print(f"    std  = {df['spi_score'].std():.4f}")
        print(f"    min  = {df['spi_score'].min():.4f}")
        print(f"    max  = {df['spi_score'].max():.4f}")
        print(f"\n  SPI Class distribution:")
        for cls, count in df["spi_class"].value_counts().sort_index().items():
            label = df[df["spi_class"] == cls]["spi_label"].iloc[0]
            pct = 100 * count / len(df)
            print(f"    Class {cls} ({label:12s}): {count:6,} ({pct:.1f}%)")
        print(f"\n  Sub-score means:")
        for dim in SPI_WEIGHTS:
            col = f"spi_{dim}"
            if col in df.columns:
                print(f"    {dim:30s}: {df[col].mean():.4f}")
        print("=" * 60)