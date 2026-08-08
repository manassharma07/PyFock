# GPU regression tests

GPU-only full-calculation cases belong here. They use the same `molecule.xyz`,
`input.py`, executable `test`, and `output.ref.txt` layout as the CPU suites. Their
local `test` definition sets `device` to `"gpu"`. GPU cases will be added after
the initial ten CPU cases have been reviewed.
