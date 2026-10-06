# RI-K before/after (HF/def2-SVP/def2-universal-jkfit, SAO, 8 threads, i9-12900K)

Times in seconds. Energies agree to 1e-11 Ha and the iteration counts are identical.

| molecule | atoms | nao (CAO) | naux (sph.) | pair rows | iters | K per iter: before | after | speedup | K total: before | after | orthonormalize rows: before | after | SCF total: before | after | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Benzene | 12 | 120 | 558 | 6945 | 8 | 0.019 | 0.011 | 1.67x | 0.15 | 0.09 | 0.12 | 0.06 | 1.06 | 0.89 | 1.19x |
| Serotonin | 25 | 255 | 1197 | 23172 | 10 | 0.138 | 0.062 | 2.23x | 1.38 | 0.62 | 0.24 | 0.25 | 3.16 | 2.43 | 1.30x |
| Caffeine | 24 | 260 | 1242 | 24332 | 12 | 0.152 | 0.068 | 2.23x | 1.83 | 0.82 | 0.26 | 0.28 | 3.79 | 2.84 | 1.33x |
| Decane_C10H22 | 32 | 260 | 1146 | 20144 | 8 | 0.119 | 0.055 | 2.16x | 0.95 | 0.44 | 0.19 | 0.21 | 2.46 | 1.95 | 1.26x |
| Icosane_C20H42 | 62 | 510 | 2256 | 45308 | 8 | 0.839 | 0.379 | 2.21x | 6.71 | 3.03 | 1.05 | 1.11 | 11.17 | 8.07 | 1.38x |
