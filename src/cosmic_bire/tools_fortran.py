"""Adapter exposing the tools_fast API with OpenMP Fortran likelihood kernels."""
from __future__ import annotations

import ctypes
from ctypes.util import find_library

import numpy as np

from . import tools_fast as _python


def _load_global(name: str, fallback: str):
    return ctypes.CDLL(find_library(name) or fallback, mode=ctypes.RTLD_GLOBAL)


_gomp = _load_global("gomp", "/usr/lib64/libgomp.so.1")
_load_global("lapack", "/usr/lib64/liblapack.so.3")

try:
    from . import _tools_fast_fortran as _compiled  # noqa: E402
except ImportError as error:
    raise ImportError(
        "The Fortran backend is not built for this Python environment. "
        "Activate cb and run ./scripts/build_fortran.sh, then reinstall with "
        "python -m pip install -e ."
    ) from error


def set_num_threads(threads: int):
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be positive")
    _gomp.omp_set_num_threads(ctypes.c_int(threads))


def likelihood_prob(c_l_o_bin_a, c_l_th_bin_a, obs_cov_bin_a, ln_det_cov,
                    bin_range, nob, alpha, beta):
    return float(_compiled.likelihood_prob(
        np.asarray(c_l_o_bin_a[:bin_range], dtype=np.float64, order="F"),
        np.asarray(c_l_th_bin_a[:bin_range], dtype=np.float64, order="F"),
        np.asarray(obs_cov_bin_a[:bin_range], dtype=np.float64, order="F"),
        bool(ln_det_cov), int(nob), np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64)))


def likelihood_prob_model_eb(c_l_o_bin_a, c_l_th_bin_a, obs_cov_bin_a,
                             ln_det_cov, bin_range, nob, alpha, beta, psi_l,
                             A, turn_off_for_lowest_indices):
    return float(_compiled.likelihood_prob_model_eb(
        np.asarray(c_l_o_bin_a[:bin_range], dtype=np.float64, order="F"),
        np.asarray(c_l_th_bin_a[:bin_range], dtype=np.float64, order="F"),
        np.asarray(obs_cov_bin_a[:bin_range], dtype=np.float64, order="F"),
        bool(ln_det_cov), int(nob), np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64), np.asarray(psi_l, dtype=np.float64),
        np.asarray(A, dtype=np.float64), int(turn_off_for_lowest_indices)))


# Covariance is precomputed once, while the likelihood is evaluated millions
# of times. Keep the exact reference implementations for the non-hot helpers.
get_covariance = _python.get_covariance
get_inverse_variance_mean_std = _python.get_inverse_variance_mean_std
get_A_B_unprimed = _python.get_A_B_unprimed
get_A_B_primed_k = _python.get_A_B_primed_k
do_not_count_this_eq = _python.do_not_count_this_eq
size = _python.size
covariance = _python.covariance
get_A_ell = _python.get_A_ell
