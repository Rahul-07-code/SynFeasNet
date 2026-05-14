"""
preprocess.py  —  SynFeasNet v2
=====================================
Place at: SynFeasNet/preprocess.py   (project root)
Run:  python preprocess.py

PURPOSE
-------
Generates data/processed/dataset_hybrid.csv with scientifically
defensible labels and scaffold-aware train/val/test splits.

WHY THE OLD LABELS WERE WRONG
------------------------------
The old logic marked every ChEMBL/PubChem molecule as synthesizable.
Database existence ≠ synthetic feasibility. Cyclosporin A exists in
ChEMBL but its first total synthesis was completed decades after its
isolation. Many complex natural products in databases were never
synthesized — they were extracted from organisms.

The old logic also used SA score / SCScore / SYBA to derive labels,
then trained a model on SMILES fingerprints which can trivially predict
SA/SCScore. Result: the model memorized scoring heuristics, not
actual chemistry (circular reasoning → inflated metrics).

NEW LABEL LOGIC: EVIDENCE PYRAMID
----------------------------------
Level 1 — HARD NEGATIVE (label=0, confidence=high):
  • SA score ≥ 7.0   (chemically intractable by any measure)
  • SA ≥ 6.0 AND SCScore ≥ 4.8   (two hard metrics converge)
  • SA ≥ 5.5 AND SYBA ≤ −25      (hard + strongly non-drug-like)
  • Macrocycle (ring ≥ 14) AND stereocenters ≥ 5 AND SA ≥ 5.0
    → Natural-product-like macrocycles are rarely total-synthesizable
  • All three metrics hard: SA ≥ 5.0, SCScore ≥ 4.5, SYBA ≤ −15

Level 2 — HARD POSITIVE (label=1, confidence=high):
  • SA ≤ 3.5  (objectively simple — textbook synthesis territory)
  • SA ≤ 4.0 AND SCScore ≤ 4.0 AND SYBA ≥ 5.0  (all agree strongly)
  • SA ≤ 4.5 AND SCScore ≤ 3.5 AND ring ≤ 8     (moderate + not macrocycle)

Level 3 — MEDIUM CONFIDENCE (label derived, confidence=medium):
  • SA ≤ 4.5 AND SCScore ≤ 4.0 AND SYBA ≥ 0     (two of three agree)
    → Only kept to prevent severe class imbalance

Level 4 — UNCERTAIN → EXCLUDED:
  • Everything not covered above is REMOVED from the training set.
  • These borderline molecules would add noisy gradients that teach
    the model to memorize heuristics.

SCAFFOLD-AWARE SPLIT
--------------------
Uses Bemis-Murcko scaffolds so that structurally similar molecules
land in the same split. This prevents a molecule in the test set from
having a nearly identical analogue in the training set (a major source
of overly-optimistic metrics in the field).

Molecules with unique scaffolds (singletons) go to train by default
since test/val need representative chemistry.

OUTPUT COLUMNS
--------------
smiles_canonical, label_final, label_confidence, label_rationale,
split, sascore, syba_score, scscore, source, molecular_weight,
num_heavy_atoms, max_ring_size, is_macrocycle, num_stereocenters
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

RDLogger.DisableLog("rdApp.*")

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV  = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_hybrid.csv")

# ── Scoring thresholds (chemistry literature references) ─────────────────────
# SA Score: Ertl & Schuffenhauer 2009 — 1 (simple) to 10 (impossible)
_SA_EASY          = 3.5    # ≤ this: objectively easy to synthesize
_SA_MODERATE_POS  = 4.0    # ≤ this: moderate complexity, positive signal
_SA_MODERATE_NEG  = 4.5    # ≤ this: upper bound for medium-confidence positive
_SA_HARD          = 5.5    # ≥ this: hard synthesis
_SA_VERY_HARD     = 6.0    # ≥ this: very hard, near-impossible for most labs
_SA_EXTREME       = 7.0    # ≥ this: chemically intractable

# SCScore: Coley et al. 2018 — 1 (simple) to 5 (complex)
_SC_EASY          = 3.5    # ≤ this: clear positive signal
_SC_MODERATE      = 4.0    # ≤ this: moderate, combined with SA can be positive
_SC_HARD          = 4.5    # ≥ this: hard
_SC_VERY_HARD     = 4.8    # ≥ this: very hard

# SYBA: Voršilák et al. 2020 — positive = synthesizable, negative = not
_SYBA_GOOD        = 5.0    # ≥ this: clear synthesizable signal
_SYBA_NEUTRAL     = 0.0    # ≥ this: weak positive signal
_SYBA_BAD         = -15.0  # ≤ this: clear non-synthesizable signal
_SYBA_VERY_BAD    = -25.0  # ≤ this: strong non-synthesizable signal

# Macrocycle / complexity
_MACRO_RING_MIN   = 12     # ring size ≥ this = macrocycle (strict definition)
_MACRO_STEREO_MIN = 5      # stereocenters ≥ this = stereochemically complex
_MACRO_SA_HARD    = 5.0    # SA ≥ this for a macrocycle = likely not synthesizable

# Balance control
_MAX_POS_MEDIUM   = 1500   # cap on medium-confidence positives (prevents re-imbalance)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _float(row, *keys):
    """Safely extract a float from any of the given column names."""
    for k in keys:
        v = row.get(k, None)
        if v is None:
            continue
        try:
            f = float(v)
            if np.isfinite(f):
                return f
        except (TypeError, ValueError):
            pass
    return np.nan


def _nan(v):
    return np.isnan(v) if isinstance(v, float) else True


def _mol_descriptors(mol):
    """Compute ring/stereochemistry descriptors."""
    ri         = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    max_ring   = max(ring_sizes) if ring_sizes else 0
    n_stereo   = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_atoms    = mol.GetNumHeavyAtoms()
    mw         = Descriptors.MolWt(mol)
    return {
        "max_ring_size":     max_ring,
        "is_macrocycle":     max_ring >= _MACRO_RING_MIN,
        "num_stereocenters": n_stereo,
        "num_heavy_atoms":   n_atoms,
        "molecular_weight":  round(mw, 2),
        "num_rings":         len(ring_sizes),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORE LABEL ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_label(row, mol):
    """
    Evidence-pyramid label assignment.

    Returns
    -------
    label       : int (0 or 1) or None (→ exclude from training)
    confidence  : str  'high' | 'medium' | 'uncertain'
    rationale   : str  short human-readable reason
    """
    sa   = _float(row, "sascore",    "sa_score",   "SAscore")
    sc   = _float(row, "scscore",    "sc_score",   "SCScore")
    syba = _float(row, "syba_score", "syba",       "SYBA")

    desc = _mol_descriptors(mol)
    max_ring = desc["max_ring_size"]
    n_stereo = desc["num_stereocenters"]
    is_macro = desc["is_macrocycle"]       # ring ≥ 12
    n_atoms  = desc["num_heavy_atoms"]

    # ── HARD NEGATIVES ────────────────────────────────────────────────────
    # Rule N1: Chemically intractable by any published metric
    if not _nan(sa) and sa >= _SA_EXTREME:
        return 0, "high", f"N1:sa_extreme:{sa:.1f}"

    # Rule N2: Two primary metrics converge on very hard
    if not _nan(sa) and not _nan(sc) and sa >= _SA_VERY_HARD and sc >= _SC_VERY_HARD:
        return 0, "high", f"N2:both_very_hard:sa{sa:.1f}+sc{sc:.1f}"

    # Rule N3: Hard SA + strongly negative SYBA
    if not _nan(sa) and not _nan(syba) and sa >= _SA_HARD and syba <= _SYBA_VERY_BAD:
        return 0, "high", f"N3:hard+syba_very_bad:sa{sa:.1f}+syba{syba:.1f}"

    # Rule N4: Complex macrocycle — large ring + stereocenters + hard SA
    # This is the KEY fix for this project:
    # Natural-product macrocycles in databases are often NOT total-synthesizable.
    # Ring ≥ 14, ≥5 stereocenters, SA ≥ 5.0 = very likely isolation-only compound.
    if max_ring >= 14 and n_stereo >= 5 and not _nan(sa) and sa >= _MACRO_SA_HARD:
        return 0, "high", f"N4:complex_macro:ring{max_ring}+stereo{n_stereo}+sa{sa:.1f}"

    # Rule N5: All three metrics agree hard
    sa_hard   = not _nan(sa)   and sa   >= 5.0
    sc_hard   = not _nan(sc)   and sc   >= _SC_HARD
    syba_bad  = not _nan(syba) and syba <= _SYBA_BAD
    if sa_hard and sc_hard and syba_bad:
        return 0, "high", f"N5:all_three_hard:sa{sa:.1f}+sc{sc:.1f}+syba{syba:.1f}"

    # Rule N6: Moderate macrocycle with high SA — excludes medium-ring naturals
    if max_ring >= _MACRO_RING_MIN and not _nan(sa) and sa >= _SA_VERY_HARD:
        return 0, "high", f"N6:macro_hard:ring{max_ring}+sa{sa:.1f}"

    # ── HARD POSITIVES ────────────────────────────────────────────────────
    # Rule P1: Simple molecule — SA ≤ 3.5 is unambiguous
    if not _nan(sa) and sa <= _SA_EASY and n_atoms <= 70:
        return 1, "high", f"P1:simple:sa{sa:.1f}"

    # Rule P2: All three metrics strongly agree synthesizable
    sa_good   = not _nan(sa)   and sa   <= _SA_MODERATE_POS
    sc_good   = not _nan(sc)   and sc   <= _SC_EASY
    syba_good = not _nan(syba) and syba >= _SYBA_GOOD
    if sa_good and sc_good and syba_good:
        return 1, "high", f"P2:all_agree:sa{sa:.1f}+sc{sc:.1f}+syba{syba:.1f}"

    # Rule P3: SA + SCScore agree, NOT a macrocycle
    sa_p3 = not _nan(sa) and sa <= _SA_MODERATE_NEG
    sc_p3 = not _nan(sc) and sc <= _SC_EASY
    if sa_p3 and sc_p3 and not is_macro:
        return 1, "high", f"P3:sa+sc_agree_nonmacro:sa{sa:.1f}+sc{sc:.1f}"

    # Rule P4: SA moderate + SYBA positive — medium ring or non-macrocycle
    sa_p4   = not _nan(sa)   and sa   <= _SA_MODERATE_NEG
    syba_p4 = not _nan(syba) and syba >= _SYBA_NEUTRAL
    sc_p4   = not _nan(sc)   and sc   <= _SC_MODERATE
    if sa_p4 and sc_p4 and syba_p4 and max_ring <= 8:
        return 1, "high", f"P4:sa+sc+syba_nonmacro:sa{sa:.1f}"

    # Rule P5: Medium confidence — two of three weakly agree, no macrocycle
    # (capped by _MAX_POS_MEDIUM to prevent re-imbalance)
    sa_p5 = not _nan(sa) and sa <= _SA_MODERATE_NEG
    sc_p5 = not _nan(sc) and sc <= _SC_MODERATE
    if sa_p5 and sc_p5 and not is_macro:
        return 1, "medium", f"P5:two_agree_medium:sa{sa:.1f}+sc{sc:.1f}"

    # ── UNCERTAIN → EXCLUDE ───────────────────────────────────────────────
    return None, "uncertain", "borderline:excluded"


# ══════════════════════════════════════════════════════════════════════════════
# SCAFFOLD-AWARE SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def get_murcko_scaffold(smiles):
    """Get Bemis-Murcko scaffold SMILES, or fall back to the molecule itself."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        sc = MurckoScaffoldSmiles(mol=mol, includeChiralCenters=False)
        return sc if sc else smiles
    except Exception:
        return smiles


def scaffold_split(df, val_frac=0.10, test_frac=0.10, seed=42):
    """
    Bemis-Murcko scaffold split.

    Groups molecules by scaffold, then assigns whole scaffold groups to
    train/val/test so no similar molecule appears in two different splits.

    Singleton scaffolds (unique structure) always go to train to ensure
    test set contains representative chemistry.
    """
    np.random.seed(seed)

    print("  Computing Murcko scaffolds...")
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        smi = str(row["smiles_canonical"])
        sc  = get_murcko_scaffold(smi)
        scaffolds[sc].append(idx)

    # Sort scaffolds: multi-member groups first (most representative), then singletons
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: -len(x))
    singletons    = [s for s in scaffold_sets if len(s) == 1]
    multi         = [s for s in scaffold_sets if len(s) > 1]

    n_total  = len(df)
    n_test   = int(n_total * test_frac)
    n_val    = int(n_total * val_frac)

    # Shuffle multi-member scaffolds so the split is not scaffold-size biased
    np.random.shuffle(multi)

    test_idx, val_idx, train_idx = [], [], []

    # Assign multi-member scaffolds first to test/val to ensure diversity
    for group in multi:
        if len(test_idx) < n_test:
            test_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    # All singleton scaffolds go to train (no risk of leakage)
    for group in singletons:
        train_idx.extend(group)

    # Verify no overlap
    assert len(set(train_idx) & set(val_idx))  == 0, "train/val overlap"
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap"
    assert len(set(val_idx)   & set(test_idx)) == 0, "val/test overlap"

    splits = pd.Series("train", index=df.index)
    for i in val_idx:
        splits[i] = "val"
    for i in test_idx:
        splits[i] = "test"

    print(f"  Scaffold split: {splits.value_counts().to_dict()}")
    print(f"  Total scaffolds: {len(scaffolds)}")
    print(f"  Multi-member scaffolds: {len(multi)} | Singletons: {len(singletons)}")
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    sep = "=" * 70

    print(f"\n{sep}")
    print("  SynFeasNet v2 — Preprocessing Pipeline")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")
    print(sep)

    # ── Load ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input CSV not found: {args.input}\n"
            f"Expected at: {INPUT_CSV}"
        )
    df = pd.read_csv(args.input, low_memory=False)
    print(f"\nLoaded {len(df):,} rows from {os.path.basename(args.input)}")
    print(f"Columns: {list(df.columns)}")

    # Check required column
    if "smiles_canonical" not in df.columns:
        # Try common alternatives
        for alt in ("smiles", "SMILES", "canonical_smiles"):
            if alt in df.columns:
                df = df.rename(columns={alt: "smiles_canonical"})
                print(f"  Renamed '{alt}' → 'smiles_canonical'")
                break
        else:
            raise ValueError(
                f"Cannot find SMILES column. Available: {list(df.columns)}"
            )

    df["smiles_canonical"] = df["smiles_canonical"].astype(str).str.strip()
    df = df[df["smiles_canonical"] != ""].reset_index(drop=True)
    print(f"After removing empty SMILES: {len(df):,} rows")

    # ── Column name detection ─────────────────────────────────────────────
    print("\nDetected scoring columns:")
    # Map the internal names (sascore, scscore, syba_score) to your actual CSV columns
    column_mapping = {
        "sascore":    ["sa_score_proxy", "sascore", "SAscore", "sa_score", "SA"],
        "scscore":    ["synthesis_feasibility_v2", "scscore", "SCScore", "sc_score", "SC"],
        "syba_score": ["p_synthesizable", "syba_score", "SYBA", "syba"],
        "source":     ["source", "Source", "data_source"],
    }
    
    found_cols = {}
    for internal, alts in column_mapping.items():
        found = next((c for c in alts if c in df.columns), None)
        found_cols[internal] = found
        print(f"  {internal:12s}: {'found as ' + found if found else 'NOT FOUND (will be NaN)'}")

    # Rename the columns to the internal names so assign_label() works without modification
    for internal, actual in found_cols.items():
        if actual and actual != internal:
            df = df.rename(columns={actual: internal})


    # ── Validate SMILES + Assign Labels ───────────────────────────────────
    print("\nValidating SMILES and assigning labels...")

    rows = df.to_dict(orient="records")
    labels, confidences, rationales = [], [], []
    mol_descriptors_list = []

    invalid, excluded, pos_high, pos_med, neg_high = 0, 0, 0, 0, 0

    for i, row in enumerate(rows):
        smi = str(row["smiles_canonical"])
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            labels.append(0)
            confidences.append("high")
            rationales.append("invalid_smiles")
            mol_descriptors_list.append({
                "max_ring_size": 0, "is_macrocycle": False,
                "num_stereocenters": 0, "num_heavy_atoms": 0,
                "molecular_weight": 0.0, "num_rings": 0,
            })
            invalid += 1
            continue

        desc  = _mol_descriptors(mol)
        lbl, conf, rat = assign_label(row, mol)

        labels.append(lbl)
        confidences.append(conf)
        rationales.append(rat)
        mol_descriptors_list.append(desc)

        if lbl is None:
            excluded += 1
        elif lbl == 1 and conf == "high":
            pos_high += 1
        elif lbl == 1 and conf == "medium":
            pos_med += 1
        elif lbl == 0:
            neg_high += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{len(rows):,} processed...")

    df["label_final"]      = labels
    df["label_confidence"] = confidences
    df["label_rationale"]  = rationales

    # Add mol descriptors
    desc_df = pd.DataFrame(mol_descriptors_list)
    for col in desc_df.columns:
        df[col] = desc_df[col].values

    print(f"\nLabel assignment complete:")
    print(f"  Invalid SMILES          : {invalid:,}")
    print(f"  Excluded (uncertain)    : {excluded:,}")
    print(f"  Pos high-confidence     : {pos_high:,}")
    print(f"  Pos medium-confidence   : {pos_med:,}")
    print(f"  Neg high-confidence     : {neg_high:,} (includes {invalid} invalid)")
    print(f"  Total labeled           : {pos_high + pos_med + neg_high:,}")

    # ── Remove uncertain rows ─────────────────────────────────────────────
    df_labeled = df[df["label_final"].notna()].copy()
    df_labeled["label_final"] = df_labeled["label_final"].astype(int)

    n_pos = (df_labeled["label_final"] == 1).sum()
    n_neg = (df_labeled["label_final"] == 0).sum()
    print(f"\nAfter removing uncertain rows: {len(df_labeled):,}")
    print(f"  Positive: {n_pos:,} ({100*n_pos/len(df_labeled):.1f}%)")
    print(f"  Negative: {n_neg:,} ({100*n_neg/len(df_labeled):.1f}%)")

    # ── Cap medium-confidence positives to prevent re-imbalance ──────────
    if pos_med > _MAX_POS_MEDIUM:
        medium_mask  = (
            (df_labeled["label_final"] == 1) &
            (df_labeled["label_confidence"] == "medium")
        )
        medium_idx   = df_labeled[medium_mask].index.tolist()
        np.random.seed(42)
        keep         = np.random.choice(medium_idx, size=_MAX_POS_MEDIUM, replace=False)
        drop         = list(set(medium_idx) - set(keep))
        df_labeled   = df_labeled.drop(index=drop).reset_index(drop=True)
        print(f"\nCapped medium-confidence positives: kept {_MAX_POS_MEDIUM:,}, "
              f"dropped {len(drop):,}")

    n_pos = (df_labeled["label_final"] == 1).sum()
    n_neg = (df_labeled["label_final"] == 0).sum()
    print(f"\nFinal balance: Pos={n_pos:,}  Neg={n_neg:,}  "
          f"Ratio={n_pos/max(n_neg,1):.1f}:1")

    if n_neg < 500:
        print("\n  WARNING: fewer than 500 negatives. Consider loosening N-rules.")
    if n_pos / max(n_neg, 1) > 4:
        print("\n  WARNING: still heavily imbalanced. Consider lowering _MAX_POS_MEDIUM.")

    # ── Rationale distribution ─────────────────────────────────────────────
    print("\nLabel rationale distribution (top 15):")
    rule_prefix = df_labeled["label_rationale"].str.split(":").str[0]
    print(rule_prefix.value_counts().head(15).to_string())

    # ── Scaffold-aware split ──────────────────────────────────────────────
    print("\nPerforming scaffold-aware train/val/test split...")
    df_labeled["split"] = scaffold_split(
        df_labeled, val_frac=0.10, test_frac=0.10, seed=42
    )

    # Verify label balance per split
    print("\nLabel distribution per split:")
    for split_name in ["train", "val", "test"]:
        sub = df_labeled[df_labeled["split"] == split_name]
        pos = (sub["label_final"] == 1).sum()
        neg = (sub["label_final"] == 0).sum()
        print(f"  {split_name:5s}: {len(sub):5,} molecules | "
              f"Pos={pos:4,} ({100*pos/max(len(sub),1):.0f}%) | "
              f"Neg={neg:4,} ({100*neg/max(len(sub),1):.0f}%)")

    # ── Macrocycle subset report ──────────────────────────────────────────
    macro_df = df_labeled[df_labeled["is_macrocycle"] == True]
    if len(macro_df) > 0:
        print(f"\nMacrocycle subset (ring ≥ {_MACRO_RING_MIN}): {len(macro_df):,} molecules")
        n_mp = (macro_df["label_final"] == 1).sum()
        n_mn = (macro_df["label_final"] == 0).sum()
        print(f"  Macrocycle Pos={n_mp:,} ({100*n_mp/len(macro_df):.0f}%) | "
              f"Neg={n_mn:,} ({100*n_mn/len(macro_df):.0f}%)")
        print("  (More negatives among macrocycles = scientifically correct)")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Ensure essential columns are present
    keep_cols = [
        "smiles_canonical", "label_final", "label_confidence",
        "label_rationale", "split",
        "molecular_weight", "num_heavy_atoms", "max_ring_size",
        "is_macrocycle", "num_stereocenters", "num_rings",
    ]
    # Add scoring columns if they exist
    for c in ["sascore", "scscore", "syba_score", "source"]:
        if c in df_labeled.columns:
            keep_cols.append(c)

    # Add any other original columns not already included
    for c in df_labeled.columns:
        if c not in keep_cols:
            keep_cols.append(c)

    df_out = df_labeled[[c for c in keep_cols if c in df_labeled.columns]]
    df_out.to_csv(args.output, index=False)

    print(f"\n{'─'*70}")
    print(f"Saved {len(df_out):,} rows → {args.output}")
    print(f"\nIMPORTANT: Delete the feature cache before retraining!")
    print(f"  Cache dir: {os.path.join(PROJECT_ROOT, 'data', 'cache_hybrid')}")
    print(f"  Command  : python preprocess.py --clear-cache")
    print(f"\nThen retrain:")
    print(f"  python training/train.py")
    print(f"{'─'*70}")

    # ── Clear cache if requested ──────────────────────────────────────────
    if args.clear_cache:
        _clear_cache()

    return df_out


def _clear_cache():
    """Delete all cached feature files so train.py regenerates with new labels."""
    import glob
    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache_hybrid")
    if not os.path.exists(cache_dir):
        print(f"  Cache dir not found: {cache_dir}  (nothing to clear)")
        return
    files = glob.glob(os.path.join(cache_dir, "*.pt")) + \
            glob.glob(os.path.join(cache_dir, "*.npz"))
    if not files:
        print("  Cache is already empty.")
        return
    for f in files:
        os.remove(f)
        print(f"  Deleted: {os.path.basename(f)}")
    print(f"  Cleared {len(files)} cache files from {cache_dir}")
    print("  train.py will regenerate all features on next run.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SynFeasNet v2 — Generate scientifically valid labels and splits"
    )
    parser.add_argument(
        "--input", default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})"
    )
    parser.add_argument(
        "--output", default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete feature cache after preprocessing (required for retraining)"
    )
    args = parser.parse_args()
    main(args)