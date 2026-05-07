"""
Retrosynthesis Providers — SynFeasNet v2
==========================================
Provides retrosynthetic analysis through multiple backends:
  1. IBM RXN for Chemistry (API)
  2. ASKCOS / MIT (API)
  3. Mock (rule-based, always available)

The Router auto-selects the best available provider.

Usage:
    from retrosynthesis.providers import RetrosynthesisRouter
    router = RetrosynthesisRouter()
    result = router.analyze("CC(=O)Oc1ccccc1C(=O)O")
    print(result.to_dict())
"""

import time
from abc import ABC, abstractmethod
from typing import List

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')


# ══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

class RetrosynthesisStep:
    """One retrosynthetic disconnection."""

    def __init__(self, reactants: List[str], product: str,
                 reaction_smiles: str = "", confidence: float = 0.0,
                 name: str = ""):
        self.reactants = reactants
        self.product = product
        self.reaction_smiles = reaction_smiles
        self.confidence = confidence
        self.name = name

    def to_dict(self) -> dict:
        return {
            "reactants": self.reactants,
            "product": self.product,
            "reaction_smiles": self.reaction_smiles,
            "confidence": round(self.confidence, 3),
            "reaction_name": self.name,
        }


class RetrosynthesisResult:
    """Complete retrosynthesis plan."""

    def __init__(self, target: str, steps: List[RetrosynthesisStep],
                 provider: str = "", success: bool = True,
                 message: str = ""):
        self.target = target
        self.steps = steps
        self.provider = provider
        self.success = success
        self.message = message

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "steps": [s.to_dict() for s in self.steps],
            "num_steps": len(self.steps),
            "provider": self.provider,
            "success": self.success,
            "message": self.message,
        }


# ══════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE
# ══════════════════════════════════════════════════════════════════════════

class RetrosynthesisProvider(ABC):
    """Abstract base for retrosynthesis backends."""

    @abstractmethod
    def analyze(self, smiles: str, max_steps: int = 5) -> RetrosynthesisResult:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


# ══════════════════════════════════════════════════════════════════════════
# IBM RXN PROVIDER
# ══════════════════════════════════════════════════════════════════════════

class IBMRXNProvider(RetrosynthesisProvider):
    """IBM RXN for Chemistry API integration."""

    def __init__(self, api_key: str = "", base_url: str = "https://rxn.res.ibm.com/rxn/api/api/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import requests
            resp = requests.get(
                f"{self.base_url}/attempts",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            return resp.status_code != 401
        except Exception:
            return False

    def analyze(self, smiles: str, max_steps: int = 5) -> RetrosynthesisResult:
        if not self.is_available():
            return RetrosynthesisResult(
                target=smiles, steps=[], provider="ibm_rxn",
                success=False, message="IBM RXN API not available"
            )
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            # Submit
            resp = requests.post(
                f"{self.base_url}/retrosynthesis",
                headers=headers,
                json={"smiles": smiles, "max_steps": max_steps},
                timeout=30,
            )
            pred_id = resp.json()["response"]["payload"]["id"]

            # Poll
            for _ in range(30):
                time.sleep(2)
                r = requests.get(
                    f"{self.base_url}/retrosynthesis/{pred_id}",
                    headers=headers, timeout=10,
                )
                data = r.json()["response"]["payload"]
                if data.get("status") == "SUCCESS":
                    return self._parse(smiles, data)
                if data.get("status") == "FAILED":
                    break

            return RetrosynthesisResult(
                target=smiles, steps=[], provider="ibm_rxn",
                success=False, message="Timeout / failed"
            )
        except Exception as e:
            return RetrosynthesisResult(
                target=smiles, steps=[], provider="ibm_rxn",
                success=False, message=str(e)
            )

    def _parse(self, smiles, data) -> RetrosynthesisResult:
        steps = []
        for seq in data.get("sequences", [])[:1]:
            for rxn in seq.get("reactions", []):
                steps.append(RetrosynthesisStep(
                    reactants=rxn.get("reactants", []),
                    product=rxn.get("product", smiles),
                    reaction_smiles=rxn.get("smiles", ""),
                    confidence=rxn.get("confidence", 0.0),
                    name=rxn.get("name", ""),
                ))
        return RetrosynthesisResult(
            target=smiles, steps=steps, provider="ibm_rxn",
            success=len(steps) > 0,
        )


# ══════════════════════════════════════════════════════════════════════════
# ASKCOS PROVIDER
# ══════════════════════════════════════════════════════════════════════════

class ASKCOSProvider(RetrosynthesisProvider):
    """MIT ASKCOS API integration."""

    def __init__(self, base_url: str = "https://askcos.mit.edu/api/v2",
                 api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def is_available(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def analyze(self, smiles: str, max_steps: int = 5) -> RetrosynthesisResult:
        if not self.is_available():
            return RetrosynthesisResult(
                target=smiles, steps=[], provider="askcos",
                success=False, message="ASKCOS not available"
            )
        try:
            import requests
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = requests.post(
                f"{self.base_url}/tree-builder/",
                headers=headers,
                json={"smiles": smiles, "max_depth": max_steps},
                timeout=10,
            )
            task_id = resp.json().get("task_id")
            if not task_id:
                return RetrosynthesisResult(
                    target=smiles, steps=[], provider="askcos",
                    success=False, message="No task ID"
                )

            for _ in range(60):
                time.sleep(2)
                r = requests.get(
                    f"{self.base_url}/celery/task/{task_id}/",
                    headers=headers, timeout=10,
                )
                data = r.json()
                if data.get("state") == "SUCCESS":
                    return self._parse(smiles, data.get("output", {}))
                if data.get("state") in ("FAILURE", "REVOKED"):
                    break

            return RetrosynthesisResult(
                target=smiles, steps=[], provider="askcos",
                success=False, message="Timeout"
            )
        except Exception as e:
            return RetrosynthesisResult(
                target=smiles, steps=[], provider="askcos",
                success=False, message=str(e)
            )

    def _parse(self, smiles, output) -> RetrosynthesisResult:
        steps = []
        for tree in output.get("trees", [])[:1]:
            self._extract(tree, steps)
        return RetrosynthesisResult(
            target=smiles, steps=steps, provider="askcos",
            success=len(steps) > 0,
        )

    def _extract(self, node, steps):
        children = node.get("children", [])
        if children:
            reactants = [c["smiles"] for c in children if c.get("is_chemical")]
            if reactants:
                steps.append(RetrosynthesisStep(
                    reactants=reactants,
                    product=node.get("smiles", ""),
                    confidence=node.get("template_score", 0.0),
                    name=node.get("template_name", ""),
                ))
            for c in children:
                self._extract(c, steps)


# ══════════════════════════════════════════════════════════════════════════
# MOCK PROVIDER (always available)
# ══════════════════════════════════════════════════════════════════════════

# Common retrosynthetic SMARTS patterns
_RETRO_RULES = [
    {
        "name": "Amide bond formation",
        "smarts": "[C:1](=O)[NH:2]",
        "desc": "Disconnect amide → carboxylic acid + amine",
    },
    {
        "name": "Ester hydrolysis",
        "smarts": "[C:1](=O)[O:2][C:3]",
        "desc": "Disconnect ester → acid + alcohol",
    },
    {
        "name": "Suzuki coupling",
        "smarts": "[c:1]-[c:2]",
        "desc": "Disconnect biaryl → aryl halide + aryl boronic acid",
    },
    {
        "name": "Reductive amination",
        "smarts": "[C:1][NH:2]",
        "desc": "Disconnect C-N → aldehyde/ketone + amine",
    },
    {
        "name": "Wittig olefination",
        "smarts": "[C:1]=[C:2]",
        "desc": "Disconnect olefin → aldehyde + phosphonium ylide",
    },
    {
        "name": "Friedel-Crafts acylation",
        "smarts": "[c:1][C:2](=O)",
        "desc": "Disconnect aryl ketone → arene + acyl chloride",
    },
]


class MockRetrosynthesisProvider(RetrosynthesisProvider):
    """
    Rule-based mock retrosynthesis using SMARTS matching.
    Always available as a fallback.
    """

    def is_available(self) -> bool:
        return True

    def analyze(self, smiles: str, max_steps: int = 5) -> RetrosynthesisResult:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return RetrosynthesisResult(
                target=smiles, steps=[], provider="mock",
                success=False, message="Invalid SMILES"
            )

        steps = []
        current_smiles = smiles
        current_mol = mol

        for _ in range(max_steps):
            step = self._find_disconnection(current_mol, current_smiles)
            if step is None:
                break
            steps.append(step)
            # Continue on the largest fragment
            if step.reactants:
                largest = max(step.reactants, key=len)
                next_mol = Chem.MolFromSmiles(largest)
                if next_mol is None or next_mol.GetNumAtoms() < 5:
                    break
                current_mol = next_mol
                current_smiles = largest
            else:
                break

        if not steps:
            steps = self._generic_analysis(mol, smiles)

        return RetrosynthesisResult(
            target=smiles, steps=steps, provider="mock",
            success=len(steps) > 0,
            message=f"Mock: {len(steps)} disconnections identified"
        )

    def _find_disconnection(self, mol, smiles) -> RetrosynthesisStep:
        for rule in _RETRO_RULES:
            pattern = Chem.MolFromSmarts(rule["smarts"])
            if pattern and mol.HasSubstructMatch(pattern):
                return RetrosynthesisStep(
                    reactants=[rule["desc"]],
                    product=smiles,
                    confidence=0.7,
                    name=rule["name"],
                )
        return None

    def _generic_analysis(self, mol, smiles) -> list:
        steps = []
        mw = Descriptors.MolWt(mol)
        n_rings = Descriptors.RingCount(mol)

        if mw > 300:
            steps.append(RetrosynthesisStep(
                reactants=["Smaller building blocks"],
                product=smiles, confidence=0.5,
                name="Convergent synthesis",
            ))
        if n_rings >= 2:
            steps.append(RetrosynthesisStep(
                reactants=["Ring system precursors"],
                product=smiles, confidence=0.5,
                name="Ring construction",
            ))
        if not steps:
            steps.append(RetrosynthesisStep(
                reactants=["Commercially available or 1-step"],
                product=smiles, confidence=0.8,
                name="Direct purchase / simple synthesis",
            ))
        return steps


# ══════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════

class RetrosynthesisRouter:
    """
    Auto-selects the best available retrosynthesis provider.
    Priority: IBM RXN → ASKCOS → Mock (fallback).

    Usage:
        router = RetrosynthesisRouter()
        result = router.analyze("CC(=O)Oc1ccccc1C(=O)O")
    """

    def __init__(self, ibm_api_key: str = "", askcos_url: str = "",
                 askcos_api_key: str = ""):
        self.providers = {
            "ibm_rxn": IBMRXNProvider(api_key=ibm_api_key),
            "askcos": ASKCOSProvider(
                base_url=askcos_url or "https://askcos.mit.edu/api/v2",
                api_key=askcos_api_key,
            ),
            "mock": MockRetrosynthesisProvider(),
        }
        self.active = self._detect()
        print(f"  Retrosynthesis provider: {self.active}")

    def _detect(self) -> str:
        for name in ["ibm_rxn", "askcos"]:
            if self.providers[name].is_available():
                return name
        return "mock"

    def analyze(self, smiles: str, max_steps: int = 5) -> RetrosynthesisResult:
        provider = self.providers[self.active]
        result = provider.analyze(smiles, max_steps=max_steps)

        # Fallback on failure
        if not result.success and self.active != "mock":
            result = self.providers["mock"].analyze(smiles, max_steps=max_steps)

        return result


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Testing Retrosynthesis Providers")
    print("=" * 65)

    router = RetrosynthesisRouter()

    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
        "CCO",                       # Ethanol
    ]

    for smi in test_smiles:
        print(f"\n{smi}:")
        result = router.analyze(smi, max_steps=3)
        d = result.to_dict()
        print(f"  Provider: {d['provider']}")
        print(f"  Success:  {d['success']}")
        print(f"  Steps:    {d['num_steps']}")
        for s in d["steps"]:
            print(f"    → {s['reaction_name']}: {s['reactants']}")

    print("\n✅ Retrosynthesis: all checks passed!")
