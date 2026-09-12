"""Warm build/gamma/J/direct timings of algorithms 10 and 11 on CPU or CUDA.

python benchmarks_tests/benchmark_DF_algo11_kernels.py XYZ BASIS [nthreads ...]
    [--gpu] [--sao] [--nostrict] [--threshold 1e-9] [--repeats 3] [--classes]

GPU times synchronize each operation; setup and device input copies are outside
contraction timings. Build timings include plan/metadata setup for algorithm 11.
Memory peaks include allocator-tracked CuPy allocations. --classes also prints
kernel time, registers/thread and local/shared bytes from compiled CUDA kernels.
"""
import argparse
import gc
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
    parser.add_argument('threads', nargs='*', type=int, default=[1, 2, 4])
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--sao', action='store_true')
    parser.add_argument('--nostrict', action='store_true')
    parser.add_argument('--threshold', type=float, default=1e-9)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--classes', action='store_true')
    parser.add_argument('--cpu-reference', action='store_true',
                        help='Also time CPU algorithm 11 and compare every stored GPU value')
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import numba
    from pyfock import Mol, Basis, Integrals
    from pyfock.Integrals import df_algo11_helpers as cpu11, df_algo10_helpers as cpu10
    from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag
    from benchmark_df_memory import device_memory_tracker

    if args.gpu:
        import cupy as cp
        from pyfock.Integrals import df_algo11_helpers_cupy as gpu11, df_algo10_helpers_cupy as gpu10
    mol = Mol(coordfile=args.xyz)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.basis)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
    nao, naux = basis.bfs_nao, aux.bfs_nao
    sq4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    metric = Integrals.rys_2c2e_symm(aux)
    if args.sao:
        proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
        metric = proj @ metric @ proj.T + 1e-12 * np.eye(naux)
    sq2 = np.sqrt(np.abs(np.diag(metric)))
    bounds10 = cpu10.aux_shell_max_bounds(sq2, aux) if args.sao else sq2
    iA, iB = np.tril_indices(nao)
    S = Integrals.overlap_mat_symm(basis)
    dmat = 0.1 * (S + 0.3 * S @ S)
    coeff = np.cos(np.arange(naux, dtype=float))
    tri = (2 * dmat - np.diag(np.diag(dmat)))[iA, iB]
    strict = not args.nostrict
    offsets, nsig = cpu10.calc_offsets_3c2e_schwarz(sq4, bounds10, args.threshold, strict, iA, iB)
    print(f'{Path(args.xyz).name} {args.basis}: nao={nao}, naux={naux}, gpu={args.gpu}', flush=True)

    def sync():
        if args.gpu:
            cp.cuda.get_current_stream().synchronize()

    def best(fn):
        out = fn()  # compile/warm up, excluded
        sync()
        del out
        timing = float('inf')
        for _ in range(args.repeats):
            gc.collect()
            sync()
            start = time.perf_counter()
            out = fn()
            sync()
            timing = min(timing, time.perf_counter() - start)
            # Release the previous build before allocating the next plan.
            if _ + 1 != args.repeats:
                del out
        return timing, out

    for nt in args.threads:
        numba.set_num_threads(nt)
        print(f'Benchmarking {nt} host threads (warming kernels)...', flush=True)
        if args.gpu:
            sq4d, sq2d, offd, dd, cd, trid = map(cp.asarray, (sq4, bounds10, offsets, dmat, coeff, tri))
            build11 = lambda mem=None: gpu11.build_plan_cupy(basis, aux, sq4, sq2, args.threshold, strict,
                                                             sao=args.sao, max_memory_gb=mem)
            build10 = lambda: gpu10.rys_3c2e_tri_schwarz_sparse_algo10_cupy(
                basis, aux, iA, iB, offd, sq4d, sq2d, args.threshold, strict, nsig, sao=args.sao)
            gamma11 = lambda p: gpu11.gamma_from_plan_cupy(p, dd)
            j11 = lambda p: gpu11.J_from_plan_cupy(p, cd)
            gamma10 = lambda v: gpu10.df_coeff_calculator_algo10_cupy(
                v, trid, nao, offd, naux, sq4d, sq2d, args.threshold, strict)
            j10 = lambda v: gpu10.J_tri_calculator_algo10_cupy(
                v, cd, iA.size, nao, offd, sq4d, sq2d, args.threshold, naux, strict)
            host = cp.asnumpy
        else:
            build11 = lambda mem=None: cpu11.build_plan(basis, aux, sq4, sq2, args.threshold, strict,
                                                       sao=args.sao, max_memory_gb=mem, ncores=nt)
            build10 = lambda: cpu10.rys_3c2e_tri_schwarz_sparse_algo10(
                basis, aux, iA, iB, offsets, sq4, bounds10, args.threshold, strict, nsig, sao=args.sao)
            gamma11 = lambda p: cpu11.gamma_from_plan(p, dmat)
            j11 = lambda p: cpu11.J_from_plan(p, coeff)
            gamma10 = lambda v: cpu10.df_coeff_calculator_algo10(
                v, tri, iA, iB, offsets, naux, sq4, bounds10, args.threshold, strict, nt)
            j10 = lambda v: cpu10.J_tri_calculator_algo10(
                v, coeff, iA, iB, offsets, iA.size, sq4, bounds10, args.threshold, strict)
            host = np.asarray
        cpu_reference = {}
        ref_plan = None
        if args.gpu and args.cpu_reference:
            tbc, ref_plan = best(lambda: cpu11.build_plan(basis, aux, sq4, sq2, args.threshold,
                                    strict, sao=args.sao, ncores=nt))
            tgc, ref_g = best(lambda: cpu11.gamma_from_plan(ref_plan, dmat))
            tjc, ref_j = best(lambda: cpu11.J_from_plan(ref_plan, coeff))
            cpu_reference.update(build_s=tbc, gamma_s=tgc, J_s=tjc)
        with device_memory_tracker(args.gpu) as memory11:
            tb, plan = best(build11)
            tg, g = best(lambda: gamma11(plan))
            tj, j = best(lambda: j11(plan))
            stored11 = plan.values.nbytes
            if ref_plan is not None:
                np.testing.assert_array_equal(plan.pair_offset, ref_plan.pair_offset)
                error = 0.0
                for start in range(0, plan.values.size, 1000000):
                    error = max(error, float(np.max(np.abs(host(plan.values[start:start + 1000000])
                                                          - ref_plan.values[start:start + 1000000]))))
                cpu_reference.update(max_values_error=error,
                    max_gamma_error=float(np.max(np.abs(host(g) - ref_g))),
                    max_J_error=float(np.max(np.abs(host(j) - ref_j))), build_speedup=tbc / tb,
                    gamma_speedup=tgc / tg, J_speedup=tjc / tj)
                del ref_plan, ref_g, ref_j
            if args.gpu and args.classes:
                report_classes(plan, best)
            # Benchmark algorithm 10 without retaining algorithm 11's device values.
            gh, jh = host(g), host(j)
            del plan, g, j
        gc.collect()
        if args.gpu:
            cp.get_default_memory_pool().free_all_blocks()
        supported10 = (not args.gpu or (max(basis.shells) <= 3 and max(aux.shells) <= 5
                                        and max(aux.bfs_nprim) <= 7))
        t10 = t10g = t10j = errorg = errorj = None
        memory10 = {}
        if supported10:
            with device_memory_tracker(args.gpu) as memory10:
                t10, v = best(build10)
                t10g, g10 = best(lambda: gamma10(v))
                t10j, j10v = best(lambda: j10(v))
                errorg = float(np.max(np.abs(gh - host(g10))))
                errorj = float(np.max(np.abs(jh[iA, iB] - host(j10v))))
                del v, g10, j10v
        else:
            print('Algorithm 10 GPU omitted: fixed local buffers support orbital l<=2 only.', flush=True)
        direct = build11(0.0)
        with device_memory_tracker(args.gpu) as memory0:
            t0g, g0 = best(lambda: gamma11(direct))
            t0j, j0 = best(lambda: j11(direct))
        result = dict(xyz=Path(args.xyz).name, basis=args.basis, gpu=args.gpu, sao=args.sao,
                      strict=strict, threshold=args.threshold, threads=nt, build11_s=tb,
                      gamma11_s=tg, J11_s=tj, build10_s=t10, gamma10_s=t10g, J10_s=t10j,
                      direct_gamma_s=t0g, direct_J_s=t0j, build_speedup=t10 / tb if supported10 else None,
                      algo10_supported=supported10, cpu11=cpu_reference,
                      stored11_bytes=stored11, stored10_bytes=nsig * 8,
                      max_gamma_error=errorg, max_J_error=errorj,
                      max_direct_gamma_error=float(np.max(np.abs(gh - host(g0)))),
                      max_direct_J_error=float(np.max(np.abs(jh - host(j0)))),
                      memory11=memory11, memory10=memory10, memory_direct=memory0)
        print('RESULT ' + json.dumps(result), flush=True)
        del direct, g0, j0


def report_classes(plan, best):
    from pyfock.Integrals.df_algo11_cuda_kernels import KERNELS
    specialized_total = 0.0
    for cls, start, end in plan.cached.groups:
        kernel, threads, cooperative = KERNELS[cls]
        items = plan.cached.items[start:end]
        blocks = end - start if cooperative else (end - start + threads - 1) // threads
        def launch():
            kernel[blocks, threads, plan.nb_stream](
                plan.orbital, plan.auxiliary, plan.shells, plan.aux_shells, plan.pairs,
                items, plan.device['pair_offset'], plan.values, plan.data_x, plan.data_w)
            plan.stream.synchronize()
        elapsed, _ = best(launch)
        specialized_total += elapsed
        print('CLASS ' + json.dumps(dict(angular_momenta=cls, tier='B' if cooperative else 'A',
                  items=end - start, seconds=elapsed,
                  registers=list(kernel.get_regs_per_thread().values()),
                  local_bytes=list(kernel.get_local_mem_per_thread().values()),
                  shared_bytes=list(kernel.get_shared_mem_per_block().values()))), flush=True)
    if plan.dims[2] <= 2 and plan.dims[3] <= 4:
        # The d-d-g wrapper safely bounds every spd item. Using it for all
        # classes measures the cost of uniform scratch sizing on the same work.
        kernel, threads, _ = KERNELS[2, 2, 4]
        count = plan.cached.items.shape[0]
        def generic():
            kernel[(count + threads - 1) // threads, threads, plan.nb_stream](
                plan.orbital, plan.auxiliary, plan.shells, plan.aux_shells, plan.pairs,
                plan.cached.items, plan.device['pair_offset'], plan.values, plan.data_x, plan.data_w)
            plan.stream.synchronize()
        elapsed, _ = best(generic)
        print('SPECIALIZATION ' + json.dumps(dict(generic_spd_s=elapsed,
              specialized_sum_s=specialized_total, speedup=elapsed / specialized_total)), flush=True)


if __name__ == '__main__':
    main()
