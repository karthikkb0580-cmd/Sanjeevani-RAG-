"""
app/chemistry/rdkit_utils.py

RDKit-based molecular descriptor computation.

Used by:
  - DrugLikenessAgent (Agent 7)
  - NoveltyAgent (Agent 8)

Wraps all RDKit operations in try/except to gracefully handle:
  - Invalid SMILES
  - Missing RDKit installation (falls back to None values)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MolecularDescriptors:
    """Computed molecular descriptors."""
    smiles_valid:           bool            = False
    molecular_weight:       Optional[float] = None
    exact_mw:               Optional[float] = None
    logp:                   Optional[float] = None
    tpsa:                   Optional[float] = None
    hbd:                    Optional[int]   = None   # H-bond donors
    hba:                    Optional[int]   = None   # H-bond acceptors
    rotatable_bonds:        Optional[int]   = None
    aromatic_rings:         Optional[int]   = None
    qed:                    Optional[float] = None
    lipinski_violations:    Optional[int]   = None
    passes_lipinski:        Optional[bool]  = None
    molecular_formula:      Optional[str]   = None
    smiles_canonical:       Optional[str]   = None
    inchi:                  Optional[str]   = None
    inchi_key:              Optional[str]   = None


def compute_descriptors(smiles: str) -> MolecularDescriptors:
    """
    Compute drug-likeness descriptors from a SMILES string using RDKit.

    Returns MolecularDescriptors with None values for any failed computation.
    Never raises — all errors are caught and logged.
    """
    result = MolecularDescriptors()

    if not smiles or not smiles.strip():
        logger.warning("Empty SMILES string provided")
        return result

    try:
        from rdkit import Chem  # type: ignore[import-untyped]
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors, AllChem  # type: ignore[import-untyped]
        from rdkit.Chem.rdMolDescriptors import CalcTPSA  # type: ignore[import-untyped]

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            logger.warning("RDKit could not parse SMILES: %s", smiles[:100])
            return result

        result.smiles_valid       = True
        result.smiles_canonical   = Chem.MolToSmiles(mol)
        result.molecular_formula  = rdMolDescriptors.CalcMolFormula(mol)
        result.molecular_weight   = round(Descriptors.MolWt(mol), 3)
        result.exact_mw           = round(Descriptors.ExactMolWt(mol), 5)
        result.logp               = round(Descriptors.MolLogP(mol), 3)
        result.tpsa               = round(CalcTPSA(mol), 3)
        result.hbd                = rdMolDescriptors.CalcNumHBD(mol)
        result.hba                = rdMolDescriptors.CalcNumHBA(mol)
        result.rotatable_bonds    = rdMolDescriptors.CalcNumRotatableBonds(mol)
        result.aromatic_rings     = rdMolDescriptors.CalcNumAromaticRings(mol)

        # QED
        try:
            result.qed = round(QED.qed(mol), 4)
        except Exception as e:
            logger.debug("QED computation failed: %s", e)

        # InChI
        try:
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey  # type: ignore[import-untyped]
            inchi = MolToInchi(mol)
            result.inchi     = inchi
            result.inchi_key = InchiToInchiKey(inchi) if inchi else None
        except Exception as e:
            logger.debug("InChI computation failed: %s", e)

        # Lipinski Ro5
        violations = 0
        if result.molecular_weight and result.molecular_weight > 500:
            violations += 1
        if result.logp and result.logp > 5:
            violations += 1
        if result.hbd and result.hbd > 5:
            violations += 1
        if result.hba and result.hba > 10:
            violations += 1
        result.lipinski_violations = violations
        result.passes_lipinski     = violations <= 1

        logger.debug(
            "Descriptors computed: MW=%.2f, LogP=%.2f, QED=%s, Lipinski=%s",
            result.molecular_weight or 0,
            result.logp or 0,
            result.qed,
            result.passes_lipinski,
        )

    except ImportError:
        logger.warning(
            "RDKit is not installed. Install with: pip install rdkit-pypi. "
            "Descriptor computation skipped."
        )
    except Exception as exc:
        logger.error("Unexpected error in descriptor computation: %s", exc)

    return result


def compute_similarity(smiles1: str, smiles2: str) -> float:
    """
    Compute Tanimoto similarity between two SMILES strings using Morgan fingerprints.

    Returns 0.0 on any error.
    """
    if not smiles1 or not smiles2:
        return 0.0
    try:
        from rdkit import Chem, DataStructs  # type: ignore[import-untyped]
        from rdkit.Chem import AllChem  # type: ignore[import-untyped]

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)

    except ImportError:
        logger.warning("RDKit not installed – similarity computation unavailable")
        return 0.0
    except Exception as exc:
        logger.error("Similarity computation error: %s", exc)
        return 0.0


def extract_scaffold(smiles: str) -> str:
    """
    Extract the Murcko scaffold from a SMILES string.

    Returns the scaffold SMILES or empty string on failure.
    """
    if not smiles:
        return ""
    try:
        from rdkit import Chem  # type: ignore[import-untyped]
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore[import-untyped]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else ""

    except ImportError:
        return ""
    except Exception as exc:
        logger.debug("Scaffold extraction failed: %s", exc)
        return ""
