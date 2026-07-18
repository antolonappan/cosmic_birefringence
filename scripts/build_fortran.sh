#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CONDA_PREFIX:?Activate the cb conda environment first}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export FFLAGS="-O3 -fopenmp -ffree-line-length-none"
python -m numpy.f2py -c --backend meson \
  src/cosmic_bire/fortran/tools_fast_fortran.pyf \
  src/cosmic_bire/fortran/tools_fast_omp.f90
cp _tools_fast_fortran*.so src/cosmic_bire/
rm -f _tools_fast_fortran*.so
echo "Built src/cosmic_bire/_tools_fast_fortran*.so"
