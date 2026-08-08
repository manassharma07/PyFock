# Full-calculation regression tests

The directory layout follows the TURBOMOLE `TURBOTEST/riper` model. A PyFock
case separates the normal calculation input from the test definition:

- `molecule.xyz` is the geometry;
- `input.py` is a standalone user-style PyFock calculation;
- `test` is the executable test definition, analogous to TURBOTEST's `CRIT`;
- `output.ref.txt` is the complete saved reference output.

Running `test` creates two ignored files in the same directory:

- `output.test.txt` contains the complete new calculation output;
- `test.report.txt` lists every selected quantity, reference value, test value,
  absolute difference, tolerance, and PASS/FAIL result.

The local `TEST` dictionary in each `test` file defines the input, output,
reference, and report files. Its local `CHECKS` list defines exactly which
quantities are extracted and the tolerance for each one. New checks can be
added without changing the calculation input or the shared driver.

All ten initial CPU inputs use density fitting, native PyFock XC, PySCF grids,
and the MINAO starting density from `benchmark_DFT_LDA_DF.py`. The matrix covers
CAO and SAO, CPU `XC_algo` 1 and 2, several native LDA/GGA functionals, and
`def2-SVP`, `def2-TZVP`, and `def2-QZVP` in both short and long suites.

## Run one directory

Enter any test directory and run the test with Python:

```text
python3 test
```

Include the matching density-fitted PySCF calculation and checks with:

```text
python3 test --with-pyscf
```

On systems that preserve the executable bit, `./test` and
`./test --with-pyscf` are equivalent shortcuts.

The calculation input is independently usable as a normal example:

```text
python3 input.py
python3 input.py --with-pyscf
```

## Run several directories

From the `tests` directory, run all short tests with:

```text
./test
./test --with-pyscf
```

The top-level `test` executable forwards all suite-runner options. For example:

```text
./test --list
./test --case 'short/ch4_spzmod_cao_xc1_qzvp' --with-pyscf
./test --ncores 4
```

Alternatively, from the repository root:

```text
python3 tests/run_tests.py short
python3 tests/run_tests.py long
python3 tests/run_tests.py short long --with-pyscf
python3 tests/run_tests.py all --case '*water*' --ncores 2
python3 tests/run_tests.py short long --list
```

The same cases are exposed through pytest:

```text
pytest tests/test_regression.py -m 'not long'
pytest tests/test_regression.py -m long
pytest tests/test_regression.py --run-pyscf
```

## Update references

Reference replacement is explicit and requires PySCF to be run as well:

```text
python3 test --with-pyscf --update-reference
python3 tests/run_tests.py short --with-pyscf --update-references
```

Review `output.ref.txt` and `test.report.txt` after regenerating a reference.
