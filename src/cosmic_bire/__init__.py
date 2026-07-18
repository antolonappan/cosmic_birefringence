"""User-facing Planck HFI cosmic-birefringence API."""

# The reference likelihood uses Numba ``parallel=True`` kernels while NumPy
# links against pthreads OpenBLAS in the cb environment. Nested BLAS/OpenMP
# pools can deadlock, so retain the reference runner's single-thread policy.
# These must be set before importing NumPy/Numba through the modules below.
import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

from .powerspec import Spectra
from .spectratheory import SpectraTheory
from .likelihood import CBlike
from .relcal import RelCal

__all__ = ["Spectra", "SpectraTheory", "CBlike", "RelCal"]
