# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando

Optimized version: splits each solver into a "build" step (factorizes the
CRAM poles once for a given A*dt) and an "apply" step (reuses those
factorizations for any number of state vectors). The original cram16/cram48
functions are kept as thin wrappers so existing call sites keep working
unchanged; new code should prefer build_*/apply_* directly to actually get
the speedup.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

# OpenMC code.

# ---- CRAM16 coefficients ----
_C16_ALPHA = np.array([
            +5.464930576870210e+3 - 3.797983575308356e+4j,
            +9.045112476907548e+1 - 1.115537522430261e+3j,
            +2.344818070467641e+2 - 4.228020157070496e+2j,
            +9.453304067358312e+1 - 2.951294291446048e+2j,
            +7.283792954673409e+2 - 1.205646080220011e+5j,
            +3.648229059594851e+1 - 1.155509621409682e+2j,
            +2.547321630156819e+1 - 2.639500283021502e+1j,
            +2.394538338734709e+1 - 5.650522971778156e+0j],
            dtype=np.complex128)

_C16_THETA = np.array([
            +3.509103608414918 + 8.436198985884374j,
            +5.948152268951177 + 3.587457362018322j,
            -5.264971343442647 + 16.22022147316793j,
            +1.419375897185666 + 10.92536348449672j,
            +6.416177699099435 + 1.194122393370139j,
            +4.993174737717997 + 5.996881713603942j,
            -1.413928462488886 + 13.49772569889275j,
            -10.84391707869699 + 19.27744616718165j],
            dtype=np.complex128)

_C16_ALPHA0 = 2.124853710495224e-16

# ---- CRAM48 coefficients ----
_theta_r = np.array([
    -4.465731934165702e+1, -5.284616241568964e+0,
    -8.867715667624458e+0, +3.493013124279215e+0,
    +1.564102508858634e+1, +1.742097597385893e+1,
    -2.834466755180654e+1, +1.661569367939544e+1,
    +8.011836167974721e+0, -2.056267541998229e+0,
    +1.449208170441839e+1, +1.853807176907916e+1,
    +9.932562704505182e+0, -2.244223871767187e+1,
    +8.590014121680897e-1, -1.286192925744479e+1,
    +1.164596909542055e+1, +1.806076684783089e+1,
    +5.870672154659249e+0, -3.542938819659747e+1,
    +1.901323489060250e+1, +1.885508331552577e+1,
    -1.734689708174982e+1, +1.316284237125190e+1])

_theta_i = np.array([
    +6.233225190695437e+1, +4.057499381311059e+1,
    +4.325515754166724e+1, +3.281615453173585e+1,
    +1.558061616372237e+1, +1.076629305714420e+1,
    +5.492841024648724e+1, +1.316994930024688e+1,
    +2.780232111309410e+1, +3.794824788914354e+1,
    +1.799988210051809e+1, +5.974332563100539e+0,
    +2.532823409972962e+1, +5.179633600312162e+1,
    +3.536456194294350e+1, +4.600304902833652e+1,
    +2.287153304140217e+1, +8.368200580099821e+0,
    +3.029700159040121e+1, +5.834381701800013e+1,
    +1.194282058271408e+0, +3.583428564427879e+0,
    +4.883941101108207e+1, +2.042951874827759e+1])

_C48_THETA = np.array(_theta_r + _theta_i * 1j, dtype=np.complex128)

_alpha_r = np.array([
    +6.387380733878774e+2, +1.909896179065730e+2,
    +4.236195226571914e+2, +4.645770595258726e+2,
    +7.765163276752433e+2, +1.907115136768522e+3,
    +2.909892685603256e+3, +1.944772206620450e+2,
    +1.382799786972332e+5, +5.628442079602433e+3,
    +2.151681283794220e+2, +1.324720240514420e+3,
    +1.617548476343347e+4, +1.112729040439685e+2,
    +1.074624783191125e+2, +8.835727765158191e+1,
    +9.354078136054179e+1, +9.418142823531573e+1,
    +1.040012390717851e+2, +6.861882624343235e+1,
    +8.766654491283722e+1, +1.056007619389650e+2,
    +7.738987569039419e+1, +1.041366366475571e+2])

_alpha_i = np.array([
    -6.743912502859256e+2, -3.973203432721332e+2,
    -2.041233768918671e+3, -1.652917287299683e+3,
    -1.783617639907328e+4, -5.887068595142284e+4,
    -9.953255345514560e+3, -1.427131226068449e+3,
    -3.256885197214938e+6, -2.924284515884309e+4,
    -1.121774011188224e+3, -6.370088443140973e+4,
    -1.008798413156542e+6, -8.837109731680418e+1,
    -1.457246116408180e+2, -6.388286188419360e+1,
    -2.195424319460237e+2, -6.719055740098035e+2,
    -1.693747595553868e+2, -1.177598523430493e+1,
    -4.596464999363902e+3, -1.738294585524067e+3,
    -4.311715386228984e+1, -2.777743732451969e+2])

_C48_ALPHA = np.array(_alpha_r + _alpha_i * 1j, dtype=np.complex128)
_C48_ALPHA0 = 2.258038182743983e-47


# Per-process cache of already-built operators, keyed on the actual matrix
# contents (not just shape) so it's safe even if different Bateman matrices
# happen to appear in the same process. Each worker process (multiprocessing
# uses separate memory spaces) builds this independently, so it naturally
# gives "factorize once per worker, reuse for every package/timestep that
# worker handles" without any changes needed in sim.py's dispatch logic --
# it's also what makes repeated calculate_states() calls in the SAME process
# (e.g. several packages processed one after another) reuse work, not just
# the timestep loop inside one call. SuperLU factorizations can't be pickled,
# so this cache is deliberately process-local; nothing here crosses a
# multiprocessing boundary.
_operator_cache = {}
_CACHE_MAX_ENTRIES = 16


def _cache_key(solver_name, A, dt, rad_flag):
    # A.tobytes() is a cheap, exact fingerprint of the matrix contents --
    # O(n^2), negligible next to the O(n^3) factorization it lets us skip.
    return (solver_name, A.shape, A.tobytes(), float(dt), bool(rad_flag))


def clear_operator_cache():
    """Drop all cached CRAM operators in the current process. Not required
    for correctness (a changed matrix just misses the cache and rebuilds
    automatically) -- useful if you want to free factorization memory
    between separate simulation runs with different decay-chain data.
    """
    _operator_cache.clear()


def _build_operator(A, dt, rad_flag, theta, alpha, alpha0):
    """Factorizes (A*dt - theta_k*I) once per pole via a reusable sparse LU
    (scipy.sparse.linalg.splu), instead of the one-shot spsolve used before.
    Returns None when there's nothing to do, matching the original solvers'
    early-out (no decay / not radioactive) so apply_operator can just hand
    n0 back unchanged.
    """
    Adt = sp.csr_matrix(A * dt, dtype=np.float64)
    if Adt.count_nonzero() == 0 or not rad_flag:
        return None
    identity = sp.eye(Adt.shape[0], format='csc')
    Adt_csc = Adt.tocsc()
    lu_factors = [sla.splu((Adt_csc - theta_k * identity).tocsc()) for theta_k in theta]
    return {'lu_factors': lu_factors, 'alpha': alpha, 'alpha0': alpha0}


def _apply_operator(op, n0):
    if op is None:
        return n0
    y = n0.copy()
    reshape_back = (np.shape(y)[1] == 1)
    if reshape_back:
        y = y.reshape(-1)
    for lu, alpha_k in zip(op['lu_factors'], op['alpha']):
        y = y + 2 * np.real(alpha_k * lu.solve(y))
    if len(np.shape(y)) == 1:
        y = y.reshape(-1, 1)
    return y * op['alpha0']


def build_cram16_operator(A, dt, rad_flag):
    """Precompute the reusable CRAM16 operator for a fixed Bateman matrix A
    and timestep dt. Call once, then feed the result to apply_cram16_operator
    for every state vector / timestep that shares this same A and dt.
    Transparently cached per-process on (A, dt, rad_flag) -- calling this
    again with an identical matrix (e.g. for the next package) is nearly
    free instead of re-factorizing.
    """
    key = _cache_key('cram16', A, dt, rad_flag)
    if key in _operator_cache:
        return _operator_cache[key]
    op = _build_operator(A, dt, rad_flag, _C16_THETA, _C16_ALPHA, _C16_ALPHA0)
    if len(_operator_cache) >= _CACHE_MAX_ENTRIES:
        _operator_cache.pop(next(iter(_operator_cache)))
    _operator_cache[key] = op
    return op


def apply_cram16_operator(op, n0):
    """Apply a CRAM16 operator built by build_cram16_operator to a state
    vector n0. Cheap (no factorization work) -- safe to call many times.
    """
    return _apply_operator(op, n0)


def build_cram48_operator(A, dt, rad_flag):
    """Precompute the reusable CRAM48 operator for a fixed Bateman matrix A
    and timestep dt. Transparently cached per-process, same as
    build_cram16_operator.
    """
    key = _cache_key('cram48', A, dt, rad_flag)
    if key in _operator_cache:
        return _operator_cache[key]
    op = _build_operator(A, dt, rad_flag, _C48_THETA, _C48_ALPHA, _C48_ALPHA0)
    if len(_operator_cache) >= _CACHE_MAX_ENTRIES:
        _operator_cache.pop(next(iter(_operator_cache)))
    _operator_cache[key] = op
    return op


def apply_cram48_operator(op, n0):
    """Apply a CRAM48 operator built by build_cram48_operator to a state
    vector n0."""
    return _apply_operator(op, n0)


def cram16(A, n0, dt, rad_flag):
    """Backwards-compatible one-shot CRAM16 call. Builds and applies the
    operator immediately, same as before -- kept so existing call sites
    keep working unchanged. Prefer build_cram16_operator/apply_cram16_operator
    directly when calling repeatedly with the same A and dt (e.g. inside a
    timestep loop), since this wrapper still refactorizes every call.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        A sparse matrix representing the decay constant matrix (Bateman
        matrix). Pre-multiplied by the time step (dt) internally.
    n0 : np.array
        Initial nuclide inventory at the start of the time step.
    dt : float
        The time step in seconds.
    rad_flag : bool
        Whether the package is radioactive; if False, n0 is returned as-is.

    Returns
    -------
    np.array
        The inventory after the time step.
    """
    op = build_cram16_operator(A, dt, rad_flag)
    return apply_cram16_operator(op, n0)


def cram48(A, n0, dt, rad_flag):
    """Backwards-compatible one-shot CRAM48 call -- see cram16 for details.
    """
    op = build_cram48_operator(A, dt, rad_flag)
    return apply_cram48_operator(op, n0)
