"""Full DFT SCF timing/energy comparison; positional arguments match the CPU script.

python benchmarks_tests/benchmark_DF_algos_scf.py XYZ BASIS sao|cao strict|nostrict
    [threshold] [DF_algo] [max_memory_gb|none] [ncores] [--gpu] [--native-grids]

Default grids are PySCF level 3. --native-grids explicitly uses PyFock grids on
machines without PySCF; compare runs using the same grid option. Run twice for
warm compilation-cache timings. A final RESULT JSON includes the total energy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    # PyFock prints a logo with block characters; keep redirected logs from failing on Windows code pages.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xyz')
    parser.add_argument('basis')
    parser.add_argument('ao', choices=['sao', 'cao'])
    parser.add_argument('strict', choices=['strict', 'nostrict'])
    parser.add_argument('threshold', nargs='?', type=float, default=1e-9)
    parser.add_argument('algo', nargs='?', type=int, default=11)
    parser.add_argument('memory', nargs='?', default='none')
    parser.add_argument('ncores', nargs='?', type=int, default=4)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--native-grids', action='store_true')
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = str(args.ncores)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pyfock import Mol, Basis, DFT
    from benchmark_df_memory import device_memory_tracker

    mem = None if args.memory.lower() == 'none' else float(args.memory)
    mol = Mol(coordfile=args.xyz)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.basis)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
    dft = DFT(mol, basis, aux, xc=[1, 7], conv_crit=1e-7, gridsLevel=3,
              use_pyscf_grids=not args.native_grids, blocksize=5000, save_ao_values=True,
              use_gpu=args.gpu, ncores=args.ncores)
    dft.max_itr = 35
    dft.XC_algo = 2
    dft.DF_algo = args.algo
    dft.max_memory_ints3c2e = mem
    dft.threshold_schwarz = args.threshold
    dft.strict_schwarz = args.strict == 'strict'
    dft.sao = args.ao == 'sao'
    dft.use_libxc = False
    with device_memory_tracker(args.gpu) as memory:
        start = time.perf_counter()
        energy, _ = dft.scf()
        elapsed = time.perf_counter() - start
    result = dict(xyz=Path(args.xyz).name, basis=args.basis, ao=args.ao,
                  strict=dft.strict_schwarz, threshold=args.threshold, algo=args.algo,
                  memory_gb=mem, ncores=args.ncores, gpu=args.gpu,
                  grids='native' if args.native_grids else 'pyscf', E=float(energy),
                  converged=bool(dft.converged), scf_seconds=elapsed, device_memory=memory)
    print('RESULT ' + json.dumps(result), flush=True)
    if not dft.converged:
        raise SystemExit('SCF did not converge; this is not a valid converged-energy benchmark.')


if __name__ == '__main__':
    main()
