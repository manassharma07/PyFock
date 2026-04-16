from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyfock import Basis, Mol


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"
REFS_PATH = DATA_DIR / "pyfock_h2o_refs.npz"
H2O_XYZ = ROOT / "examples" / "h2o.xyz"

BASIS_NAMES = ("def2-SVP", "def2-TZVP", "def2-QZVP")
AUX_BASIS_NAME = "def2-universal-jfit"
RYS_4C2E_SLICE = (0, 4, 0, 4, 0, 4, 0, 4)


def build_h2o_mol() -> Mol:
    return Mol(coordfile=str(H2O_XYZ))


def build_basis(mol: Mol, basis_name: str) -> Basis:
    return Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})


def load_references() -> np.lib.npyio.NpzFile:
    if not REFS_PATH.exists():
        pytest.fail(
            f"Missing reference data at {REFS_PATH}. "
            f"Generate it with: python3 tests/generate_references.py"
        )
    return np.load(REFS_PATH, allow_pickle=False)


@pytest.fixture(scope="session")
def refs():
    refs_obj = load_references()
    try:
        yield refs_obj
    finally:
        refs_obj.close()
