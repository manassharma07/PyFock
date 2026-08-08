"""Discovery and execution support for directory-based PyFock tests."""

import ast
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


TESTS_ROOT = Path(__file__).resolve().parent
SUITES = ("short", "long", "gpu")


@dataclass(frozen=True)
class RegressionCase:
    directory: Path
    suite: str
    metadata: Dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.metadata["name"])

    @property
    def identifier(self) -> str:
        return self.suite + "/" + self.directory.name

    @property
    def test_path(self) -> Path:
        return self.directory / "test"

    @property
    def input_path(self) -> Path:
        return self.directory / str(self.metadata.get("input", "input.py"))

    @property
    def reference_path(self) -> Path:
        return self.directory / str(self.metadata.get("reference", "output.ref.txt"))

    @property
    def output_path(self) -> Path:
        return self.directory / str(self.metadata.get("output", "output.test.txt"))

    @property
    def report_path(self) -> Path:
        return self.directory / str(self.metadata.get("report", "test.report.txt"))


def _read_literal(path: Path, variable: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("{} must define a literal {} value".format(path, variable))


def load_case(directory: Path, suite: Optional[str] = None) -> RegressionCase:
    directory = Path(directory).resolve()
    suite = suite or directory.parent.name
    test_path = directory / "test"
    metadata = _read_literal(test_path, "TEST")
    if not isinstance(metadata, dict):
        raise ValueError("{} TEST must be a dictionary".format(test_path))

    required = (
        "name",
        "description",
        "basis",
        "auxbasis",
        "xc_name",
        "xc_ids",
        "ao_basis",
        "xc_algo",
        "device",
        "use_pyscf_grids",
        "use_libxc",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError("{} is missing {}".format(test_path, ", ".join(missing)))
    if suite not in SUITES:
        raise ValueError("Unknown suite {!r}".format(suite))
    if suite == "gpu" and metadata["device"] != "gpu":
        raise ValueError("GPU cases must set device='gpu'")
    if suite != "gpu" and metadata["device"] != "cpu":
        raise ValueError("Short and long cases must set device='cpu'")
    if metadata["device"] == "cpu" and int(metadata["xc_algo"]) not in (1, 2):
        raise ValueError("CPU XC_algo must be 1 or 2")
    if str(metadata["ao_basis"]).upper() not in ("CAO", "SAO"):
        raise ValueError("ao_basis must be CAO or SAO")
    if metadata["use_pyscf_grids"] is not True:
        raise ValueError("Every test must use PySCF grids")
    if metadata["use_libxc"] is not False:
        raise ValueError("These tests must exercise native PyFock XC")

    case = RegressionCase(directory, suite, metadata)
    if not case.input_path.is_file():
        raise ValueError("Missing required case file: {}".format(case.input_path))

    # Geometry filenames belong to the calculation input.  A test may declare
    # one explicitly for early validation, but the suite must not assume a
    # particular name such as molecule.xyz.
    geometry = metadata.get("geometry")
    if geometry is not None:
        geometry_path = directory / str(geometry)
        if not geometry_path.is_file():
            raise ValueError("Missing declared geometry file: {}".format(geometry_path))
    return case


def discover_cases(
    suites: Sequence[str] = ("short",), root: Path = TESTS_ROOT
) -> List[RegressionCase]:
    cases = []
    for suite in suites:
        suite_path = Path(root) / suite
        if not suite_path.is_dir():
            continue
        for test_path in sorted(suite_path.glob("*/test")):
            cases.append(load_case(test_path.parent, suite))
    return cases


def matches(case: RegressionCase, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    return any(
        fnmatch.fnmatch(case.identifier, pattern)
        or fnmatch.fnmatch(case.name, pattern)
        for pattern in patterns
    )


def run_case(
    case: RegressionCase,
    with_pyscf: bool = False,
    ncores_override: Optional[int] = None,
    output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    update_reference: bool = False,
) -> Path:
    command = [sys.executable, str(case.test_path)]
    if with_pyscf:
        command.append("--with-pyscf")
    if ncores_override is not None:
        command.extend(("--ncores", str(ncores_override)))
    if output_path is not None:
        command.extend(("--output", str(Path(output_path).resolve())))
    if report_path is not None:
        command.extend(("--report", str(Path(report_path).resolve())))
    if update_reference:
        command.append("--update-reference")

    completed = subprocess.run(
        command,
        cwd=str(case.directory),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    final_report = Path(report_path or case.report_path)
    if completed.returncode != 0:
        details = completed.stdout.strip()
        if final_report.is_file():
            details = final_report.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            "{} failed with status {}\n{}".format(
                case.identifier, completed.returncode, details
            )
        )
    return final_report


def format_case_summary(case: RegressionCase) -> str:
    metadata = case.metadata
    return "{:<38} {:<9} {:<4} XC#{:<1} {:<8} {}".format(
        case.identifier,
        metadata["basis"],
        metadata["ao_basis"],
        metadata["xc_algo"],
        metadata["xc_name"],
        str(metadata["device"]).upper(),
    )
