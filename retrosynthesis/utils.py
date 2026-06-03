from rdkit import Chem


def validate_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return ""

    return Chem.MolToSmiles(mol)


def is_macrocycle(smiles: str) -> bool:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return False

    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        if len(ring) >= 12:
            return True

    return False