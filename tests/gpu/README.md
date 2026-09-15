# GPU regression tests

GPU-only full-calculation cases belong here. They use the same `molecule.xyz`,
`input.py`, executable `test`, and `output.ref.txt` layout as the CPU suites. Their
local `test` definition sets `device` to `"gpu"`.

Every case runs `XC_algo = 3`, the GPU default and the only algorithm the GPU
branch should be tested with; algorithm 2 is obsolete on the GPU. The references
are CPU outputs of the same calculation, and the CPU has only its own XC driver,
so a reference is unaffected by that choice: the case asserts that the GPU-native
XC algorithm reproduces the CPU to the stated tolerance.

The CH4/def2-QZVP cases test algorithm 11 in CAO and SAO, plus
algorithm 11 with no cached integrals. The total-energy tolerance is 1e-8
Hartree. They require PySCF for the same level-3 grids and initial density used
by the CPU references.

The H2O/def2-SVP cases compare GPU algorithms 10 and 11 (full/zero cache)
against CPU algorithm 11 on native level-3 grids, in CAO and SAO. These need no
PySCF: `python tests/test gpu --case 'h2o_*' --ncores 4`. The runner accepts
`use_pyscf_grids: False` only in this suite, because GPU hosts often lack PySCF.
On Windows machines where CuPy cannot compile its CUB kernels (no `cl.exe` on
`PATH`), set `CUPY_ACCELERATORS=` (empty) before running; see
`docs/df_algo11_gpu.md`.

The H2O/def2-SVP PBE case (`h2o_pbe_cao_svp_xc3_df_algo_11`) is the only GGA
case; it guards the GPU AO-gradient kernel and the gradient-dependent XC
functionals.

Algorithm 10's fixed CUDA buffers do not support orbital f/g shells, so it is
not used as a QZVP reference.

Run all cases with `python tests/test gpu --ncores 4`. Integral-only hardware coverage is in
`python -m pytest tests/test_df_algo11_cupy.py` and does not require PySCF.
