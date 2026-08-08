import pytest

from pyfock import XC
from tests.regression import discover_cases, run_case


SHORT_CASES = discover_cases(("short",))
LONG_CASES = discover_cases(("long",))


def test_regression_matrix_contract():
    cases = SHORT_CASES + LONG_CASES
    assert cases, "no regression test directories were discovered"
    native_ids = set(XC.get_implemented_ids())
    for case in cases:
        assert case.metadata["use_pyscf_grids"] is True
        if case.metadata["use_libxc"] is False:
            assert set(case.metadata["xc_ids"]) <= native_ids
        assert case.input_path.is_file()
        assert case.reference_path.is_file()
        assert case.test_path.is_file()


def _run_and_check(case, tmp_path, pytestconfig):
    run_case(
        case,
        with_pyscf=pytestconfig.getoption("--run-pyscf"),
        ncores_override=pytestconfig.getoption("--regression-ncores"),
        output_path=tmp_path / "output.test.txt",
        report_path=tmp_path / "test.report.txt",
    )


@pytest.mark.regression
@pytest.mark.parametrize("case", SHORT_CASES, ids=lambda case: case.directory.name)
def test_short_regression(case, tmp_path, pytestconfig):
    _run_and_check(case, tmp_path, pytestconfig)


@pytest.mark.regression
@pytest.mark.long
@pytest.mark.parametrize("case", LONG_CASES, ids=lambda case: case.directory.name)
def test_long_regression(case, tmp_path, pytestconfig):
    _run_and_check(case, tmp_path, pytestconfig)
