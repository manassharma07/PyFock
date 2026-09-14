"""Skala, the neural exchange-correlation functional of Microsoft Research AI for Science.

Skala is a machine-learned semilocal exchange-correlation functional distributed as a TorchScript
checkpoint. It differs from every other functional in :mod:`pyfock.XC` in one way that shapes this whole
module: it is **not pointwise**. Where an LDA, GGA or meta-GGA returns an energy density and its
derivatives at each grid point independently, Skala consumes the density features of a whole atomic grid
at once -- its non-local layers aggregate over the points of each atom -- and returns the *total* XC
energy as a single number. The XC potential is therefore obtained by automatic differentiation of that
number rather than from closed-form derivatives.

What comes back out of the differentiation is, however, exactly what PyFock's meta-GGA machinery already
consumes. Writing ``E_xc = sum_p w_p e_p``, autograd hands back

    dE/drho_p ,  dE/d(grad rho)_p ,  dE/dtau_p

per grid point, with the quadrature weight ``w_p`` already folded in. Those three are precisely the
``weights*vrho``, ``2*weights*vsigma*grad_rho`` and ``weights*vtau`` intermediates that
:func:`pyfock.Integrals.eval_xc_2` builds before contracting them with the AO values, so the potential
assembly is unchanged and only the block that evaluates the functional is replaced.

Spin convention
---------------
Skala always takes spin-resolved features, shaped ``(2, G)`` for the density and kinetic-energy density
and ``(2, 3, G)`` for the density gradient, whereas PyFock's restricted code carries total quantities. The
two are bridged by feeding ``rho/2`` into both spin channels. Differentiating a function of
``d_alpha = d_beta = rho/2`` gives ``dE/drho = (c_alpha + c_beta)/2``, so the returned cotangents are
averaged over the two channels (they are equal by symmetry at a closed-shell density; averaging is exact
in general and immune to round-off asymmetry).

Dependencies
------------
Only PyTorch is needed at run time. The checkpoint is read with :func:`torch.jit.load`, so neither the
``skala`` package -- which would pull in PySCF and e3nn, neither of which PyFock uses and the first of
which has no Windows wheels -- nor any of its dependencies have to be installed. ``huggingface_hub`` is
used once to fetch the checkpoint and can be avoided entirely by pointing ``SKALA_LOCAL_MODEL_PATH`` at a
local copy.

Security
--------
TorchScript deserialization executes code from the file, so every checkpoint downloaded here is verified
against the SHA-256 digests published by Microsoft (:data:`_KNOWN_HASHES`, copied verbatim from
``skala/functional/_hashes.py``) before it is loaded. A file supplied through
``SKALA_LOCAL_MODEL_PATH`` is *not* verified; only point it at a file you trust.

References
----------
Microsoft Research AI for Science, "Accurate and scalable exchange-correlation with deep learning",
https://github.com/microsoft/skala (MIT licence). Skala 1.1 expects a DFT-D3 dispersion correction with
B3LYP5 parameters, which is additive and not part of the SCF; see :func:`d3_settings`.
"""

import hashlib
import json
import os

import numpy as np

__all__ = ['is_skala', 'canonical_name', 'load_skala', 'SkalaFunctional', 'SKALA_FUNCTIONALS']


# Published functional name -> (HuggingFace repo, CPU checkpoint, CUDA checkpoint). The CUDA checkpoints
# are traced for the device and are not interchangeable with the CPU ones.
SKALA_FUNCTIONALS = {
    'skala-1.1':      ('microsoft/skala-1.1', 'skala-1.1-rev1.fun', 'skala-1.1-rev1-cuda.fun'),
    'skala-1.1-rev1': ('microsoft/skala-1.1', 'skala-1.1-rev1.fun', 'skala-1.1-rev1-cuda.fun'),
    'skala-1.1-rev0': ('microsoft/skala-1.1', 'skala-1.1.fun',      'skala-1.1-cuda.fun'),
    'skala-1.0':      ('microsoft/skala-1.0', 'skala-1.0.fun',      'skala-1.0-cuda.fun'),
}

# SHA-256 digests of the published checkpoints, copied from skala/functional/_hashes.py.
_KNOWN_HASHES = {
    ('microsoft/skala-1.0', 'skala-1.0.fun'):           '08d94436995937eb57c451af7c92e2c7f9e1bff6b7da029a3887e9f9dd4581c0',
    ('microsoft/skala-1.0', 'skala-1.0-cuda.fun'):      '0b38e13237cec771fed331664aace42f8c0db8f15caca6a5c563085e61e2b1fd',
    ('microsoft/skala-1.1', 'skala-1.1.fun'):           '0c8432ac3f03c8f1276372df9aca5b7ee7f8939d47a8789eb158976e89aa0606',
    ('microsoft/skala-1.1', 'skala-1.1-cuda.fun'):      'f77be6002d873c0a2384b6df7850d32bbec519036344ff5fdde9730c6f9a4326',
    ('microsoft/skala-1.1', 'skala-1.1-rev1.fun'):      '7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd',
    ('microsoft/skala-1.1', 'skala-1.1-rev1-cuda.fun'): 'f848eae769dca91741a518ae7275d10caac398ab21db649f91bc1f136872f223',
}

_PROTOCOL_VERSION = 2

# Loaded checkpoints, keyed by (canonical name, device string).
_MODEL_CACHE = {}


def is_skala(xc):
    """Whether ``xc`` names a Skala functional (case-insensitive); False for anything else, including
    LibXC IDs and the LibXC-style names the rest of :mod:`pyfock.XC` uses."""
    return isinstance(xc, str) and xc.strip().lower() in SKALA_FUNCTIONALS


def canonical_name(xc):
    """The canonical Skala name for ``xc``, e.g. ``'Skala-1.1'`` -> ``'skala-1.1'``.

    Raises
    ------
    ValueError
        If ``xc`` is not a published Skala functional.
    """
    name = xc.strip().lower() if isinstance(xc, str) else xc
    if name not in SKALA_FUNCTIONALS:
        raise ValueError("Unknown Skala functional '" + str(xc) + "'. Available: "
                         + ', '.join(sorted(SKALA_FUNCTIONALS)) + '.')
    return name


def _resolve_checkpoint(name, device_type):
    """Local path and expected SHA-256 of the checkpoint of ``name`` for ``device_type`` ('cpu'/'cuda').

    Downloads it from HuggingFace on first use (cached under ``~/.cache/huggingface`` afterwards).
    ``SKALA_LOCAL_MODEL_PATH`` overrides the lookup and disables hash verification.
    """
    local = os.environ.get('SKALA_LOCAL_MODEL_PATH')
    if local:
        print('Skala: loading the model from SKALA_LOCAL_MODEL_PATH; SHA-256 verification is disabled.',
              flush=True)
        return local, None

    repo, cpu_file, cuda_file = SKALA_FUNCTIONALS[name]
    filename = cuda_file if device_type == 'cuda' else cpu_file
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "Skala needs the model checkpoint '" + filename + "'. Either install huggingface_hub "
            '(pip install huggingface_hub) so that PyFock can download it, or download it by hand from '
            'https://huggingface.co/' + repo + ' and point SKALA_LOCAL_MODEL_PATH at the file.')
    return hf_hub_download(repo_id=repo, filename=filename), _KNOWN_HASHES.get((repo, filename))


class SkalaFunctional:
    """A loaded Skala checkpoint, evaluated on PyFock's grids.

    Instances are created by :func:`load_skala`, which caches them per (functional, device); building one
    directly re-reads the checkpoint. The object is stateless apart from the model, so a single instance
    is reused across SCF iterations.

    Attributes
    ----------
    name : str
        Canonical functional name.
    features : list of str
        Feature names the checkpoint declares, e.g. ``['density', 'kin', 'grad', ...]``.
    metadata : dict
        Training metadata stored in the checkpoint.
    device : str
        ``'cpu'`` or ``'cuda'``.
    """

    def __init__(self, model, name, features, metadata, d3, device):
        self._model = model
        self.name = name
        self.features = features
        self.metadata = metadata
        self._d3 = d3
        self.device = device
        self._needs = set(features)

    # ------------------------------------------------------------------
    def d3_settings(self):
        """Name of the DFT-D3 parametrisation this functional expects, or None.

        Skala 1.1 returns ``'b3lyp5'``. The dispersion energy is a purely additive, geometry-dependent
        term that does not enter the SCF; PyFock does not add it automatically, so a total energy reported
        without it is the bare Skala energy.
        """
        return self._d3

    # ------------------------------------------------------------------
    def _layout(self, atom_sizes, max_points):
        """Grid-point layout for the model, and the evaluation chunks that tile it.

        The model's non-local layers aggregate over the points of each atom, so the energy is additive
        over atoms and chunking is exact. Atoms are visited in order of increasing atomic grid size, as
        Skala's own chunker does, so that each chunk holds atoms with the *same* number of grid points:
        the model pads a chunk to a rectangular (atoms, points per atom) layout, and mixing sizes would
        pad every small atom up to the largest one in the chunk.

        Returns ``(point_order, chunks)``, where ``point_order`` maps a position in that size-major
        layout to its position in the atom-major one, and each chunk is
        ``(atom_indices, start, stop)`` with ``start``/``stop`` slicing the size-major layout.
        """
        order = np.argsort(atom_sizes, kind='stable')
        sizes = atom_sizes[order]
        # Where each atom's points begin in the atom-major layout (atoms in index order).
        atom_major_start = np.cumsum(atom_sizes) - atom_sizes
        point_order = (np.concatenate([np.arange(atom_major_start[a], atom_major_start[a] + atom_sizes[a])
                                       for a in order])
                       if order.size else np.zeros(0, dtype=np.int64))

        starts = np.concatenate(([0], np.cumsum(sizes)))
        chunks = []
        i = 0
        while i < len(sizes):
            j = i + 1
            while j < len(sizes) and sizes[j] == sizes[i]:
                j += 1
            per_chunk = max(1, int(max_points // max(1, int(sizes[i]))))
            for k in range(i, j, per_chunk):
                stop = min(k + per_chunk, j)
                chunks.append((order[k:stop], int(starts[k]), int(starts[stop])))
            i = j
        return point_order, chunks

    # ------------------------------------------------------------------
    def exc_and_potential(self, rho, rho_grad, tau, coords, weights, atomic_weights, atom_idx,
                          atom_coords, max_points_per_chunk=250000, nuclear_terms=False):
        """Exchange-correlation energy and its grid-point derivatives for one density.

        All inputs are the *total* (spin-summed) quantities PyFock's restricted code works with, given on
        the whole molecular grid in whatever order the grid happens to be in; the reordering to the
        atom-major layout the model needs, and back, is handled here.

        Parameters
        ----------
        rho : (G,) ndarray
            Electron density at the grid points.
        rho_grad : (3, G) ndarray
            Cartesian gradient of the density.
        tau : (G,) ndarray
            Positive kinetic-energy density, ``0.5 * sum_i |grad psi_i|^2`` (the LibXC convention, which
            is what :func:`pyfock.Integrals.eval_xc_2` computes).
        coords : (G, 3) ndarray
            Grid point coordinates in Bohr.
        weights : (G,) ndarray
            Quadrature weights, Becke partitioning included.
        atomic_weights : (G,) ndarray
            Single-atom quadrature weights, i.e. ``weights`` before the Becke partitioning factor
            (:attr:`pyfock.Grids.Grids.atomic_weights`).
        atom_idx : (G,) int ndarray
            Index of the atom each grid point belongs to.
        atom_coords : (natm, 3) ndarray
            Nuclear coordinates in Bohr.
        max_points_per_chunk : int
            Upper bound on the grid points evaluated by the model at once. Chunking is exact and only
            trades speed for peak memory.
        nuclear_terms : bool
            Also return the part of ``dE/dR`` that does not go through the density. Unlike a semilocal
            functional, Skala reads the grid geometry directly -- it consumes ``grid_coords`` and
            ``coarse_0_atomic_coords`` -- so the energy depends on the nuclear positions explicitly.
            Both derivatives come out of the same reverse pass at negligible extra cost.

        Returns
        -------
        exc : float
            Total exchange-correlation energy in Hartree.
        vrho : (G,) ndarray
            ``dE/drho`` at each grid point, quadrature weight included.
        vgrad : (3, G) ndarray
            ``dE/d(grad rho)`` at each grid point, quadrature weight included.
        vtau : (G,) ndarray
            ``dE/dtau`` at each grid point, quadrature weight included.
        explicit_grad : (natm, 3) ndarray
            Only with ``nuclear_terms``: the explicit ``dE/dR``, i.e. the model's direct dependence on
            the nuclear coordinates plus the cotangent of every grid point gathered onto the atom whose
            grid it belongs to (grid points translate rigidly with their atom). This is *not* the whole
            XC gradient -- the density-mediated part comes from contracting ``vrho``/``vgrad``/``vtau``
            with the AO derivatives, and the grid-weight response is not included; see
            :func:`pyfock.Integrals.eval_xc_grad_skala`.
        """
        import torch

        device = torch.device(self.device)
        ngrids = rho.shape[0]
        natm = atom_coords.shape[0]

        # The model needs the points of each atom to be contiguous. A stable sort of atom_idx groups
        # them by atom whatever order the grid itself is in (PyFock groups points into spatial boxes by
        # default), and _layout then reorders the atoms by atomic grid size so that a chunk never mixes
        # sizes. The composition of the two maps a position in the model's layout to a grid index, and
        # scattering through it puts the cotangents back where they came from.
        atom_sizes = np.bincount(atom_idx, minlength=natm).astype(np.int64)
        if atom_sizes.sum() != ngrids:
            raise ValueError('Skala: the grid has ' + str(ngrids) + ' points but they are assigned to '
                             + str(int(atom_sizes.sum())) + ' atomic grids.')
        point_order, chunks = self._layout(atom_sizes, max_points_per_chunk)
        perm = np.argsort(atom_idx, kind='stable')[point_order]

        def to_torch(array, dtype=torch.float64):
            return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)

        # Views of everything the model consumes, in its own layout.
        rho_m = to_torch(rho[perm])
        grad_m = to_torch(rho_grad[:, perm])
        tau_m = to_torch(tau[perm])
        coords_m = to_torch(coords[perm])
        weights_m = to_torch(weights[perm])
        atomic_w_m = to_torch(atomic_weights[perm])
        atom_coords_t = to_torch(atom_coords)

        exc = 0.0
        vrho_m = torch.zeros_like(rho_m)
        vgrad_m = torch.zeros_like(grad_m)
        vtau_m = torch.zeros_like(tau_m)
        # Explicit nuclear dependence (only assembled when the caller asks for it): Skala reads the grid
        # geometry itself, so the energy depends on the nuclear positions beyond the density.
        dgrid_m = torch.zeros_like(coords_m) if nuclear_terms else None
        datom = torch.zeros_like(atom_coords_t) if nuclear_terms else None

        for atoms, start, stop in chunks:
            sel = slice(start, stop)
            # Spin-resolved inputs: a closed-shell density splits evenly between the two channels.
            density = torch.stack((rho_m[sel] * 0.5, rho_m[sel] * 0.5)).requires_grad_()
            grad = torch.stack((grad_m[:, sel] * 0.5, grad_m[:, sel] * 0.5)).requires_grad_()
            kin = torch.stack((tau_m[sel] * 0.5, tau_m[sel] * 0.5)).requires_grad_()

            chunk_atoms = torch.as_tensor(atoms, dtype=torch.long, device=device)
            chunk_sizes = torch.as_tensor(atom_sizes[atoms], dtype=torch.long, device=device)
            grid_coords = coords_m[sel]
            chunk_atom_coords = atom_coords_t[chunk_atoms]
            if nuclear_terms:
                grid_coords = grid_coords.detach().requires_grad_()
                chunk_atom_coords = chunk_atom_coords.detach().requires_grad_()
            features = {
                'density': density,
                'grad': grad,
                'kin': kin,
                'grid_coords': grid_coords,
                'grid_weights': weights_m[sel],
                'atomic_grid_weights': atomic_w_m[sel],
                'atomic_grid_sizes': chunk_sizes,
                'coarse_0_atomic_coords': chunk_atom_coords,
                'atomic_grid_size_bound_shape': torch.zeros(int(chunk_sizes.max()), 0, dtype=torch.long,
                                                            device=device),
            }
            features = {key: value for key, value in features.items() if key in self._needs}

            energy = self._model.get_exc(features)
            # One reverse pass gives every derivative needed, whether or not the nuclear terms were
            # asked for; adding the two geometry inputs costs nothing beyond their own storage.
            inputs = (density, grad, kin)
            if nuclear_terms:
                # allow_unused: a functional that does not declare grid_coords or
                # coarse_0_atomic_coords simply has no explicit nuclear dependence through it, and
                # autograd hands back None rather than raising.
                inputs = inputs + (grid_coords, chunk_atom_coords)
            cotangents = torch.autograd.grad(energy, inputs, allow_unused=nuclear_terms)
            c_rho, c_grad, c_kin = cotangents[0], cotangents[1], cotangents[2]

            exc += float(energy.detach())
            # d/d(total) = (d/d(alpha) + d/d(beta)) / 2, since each channel holds half the total.
            vrho_m[sel] = 0.5 * (c_rho[0] + c_rho[1])
            vgrad_m[:, sel] = 0.5 * (c_grad[0] + c_grad[1])
            vtau_m[sel] = 0.5 * (c_kin[0] + c_kin[1])
            if nuclear_terms:
                if cotangents[3] is not None:
                    dgrid_m[sel] = cotangents[3]
                if cotangents[4] is not None:
                    datom.index_add_(0, chunk_atoms, cotangents[4])

        vrho = np.empty(ngrids, dtype=np.float64)
        vtau = np.empty(ngrids, dtype=np.float64)
        vgrad = np.empty((3, ngrids), dtype=np.float64)
        vrho[perm] = vrho_m.cpu().numpy()
        vtau[perm] = vtau_m.cpu().numpy()
        vgrad[:, perm] = vgrad_m.cpu().numpy()
        if not nuclear_terms:
            return exc, vrho, vgrad, vtau

        # Every grid point translates rigidly with the atom it belongs to, so its cotangent lands on
        # that atom; the model's direct dependence on the nuclear coordinates adds on top.
        dgrid = np.empty((ngrids, 3), dtype=np.float64)
        dgrid[perm] = dgrid_m.cpu().numpy()
        explicit = np.ascontiguousarray(datom.cpu().numpy(), dtype=np.float64)
        # bincount rather than np.add.at: the same segmented sum, but without the latter's
        # element-by-element unbuffered path, which is an order of magnitude slower on a large grid.
        for direction in range(3):
            explicit[:, direction] += np.bincount(atom_idx, weights=dgrid[:, direction],
                                                  minlength=natm)
        return exc, vrho, vgrad, vtau, explicit


def load_skala(xc, use_gpu=False):
    """Load a Skala functional, reusing an already loaded one when possible.

    Parameters
    ----------
    xc : str
        Functional name, e.g. ``'skala-1.1'`` (case-insensitive).
    use_gpu : bool
        Load the CUDA checkpoint onto the GPU instead of the CPU one. The two are separate files traced
        for their device.

    Returns
    -------
    SkalaFunctional
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            'The Skala functionals need PyTorch (pip install torch). PyFock loads the published '
            'TorchScript checkpoint directly, so the skala package itself -- and with it PySCF -- is not '
            'required.')

    name = canonical_name(xc)
    device = 'cuda' if use_gpu else 'cpu'
    cached = _MODEL_CACHE.get((name, device))
    if cached is not None:
        return cached

    path, expected_hash = _resolve_checkpoint(name, device)
    if expected_hash is not None:
        with open(path, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected_hash:
            raise ValueError('Skala: the checkpoint ' + str(path) + ' has SHA-256 ' + digest
                             + ' but ' + expected_hash + ' was expected. Refusing to load it: '
                             'TorchScript files execute code when deserialized.')

    extra_files = {'metadata': b'', 'features': b'', 'expected_d3_settings': b'', 'protocol_version': b''}
    model = torch.jit.load(path, _extra_files=extra_files, map_location=torch.device(device))

    protocol = json.loads(extra_files['protocol_version'].decode('utf-8'))
    if protocol != _PROTOCOL_VERSION:
        raise RuntimeError('Skala: the checkpoint uses protocol version ' + str(protocol)
                           + ' but PyFock supports ' + str(_PROTOCOL_VERSION) + '.')

    functional = SkalaFunctional(
        model, name,
        features=json.loads(extra_files['features'].decode('utf-8')),
        metadata=json.loads(extra_files['metadata'].decode('utf-8')),
        d3=json.loads(extra_files['expected_d3_settings'].decode('utf-8')),
        device=device)
    _MODEL_CACHE[(name, device)] = functional
    return functional


def citation(name):
    """Citation string printed in the SCF output for a Skala functional."""
    return (SKALA_FUNCTIONALS[canonical_name(name)][0].split('/')[1].upper()
            + ': Microsoft Research AI for Science, "Accurate and scalable exchange-correlation with '
            'deep learning", https://github.com/microsoft/skala (MIT).')
