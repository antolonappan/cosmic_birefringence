# Cosmic Birefringence — Planck HFI

An installable, configuration-driven Planck NPIPE HFI implementation of the
Eskilt & Komatsu likelihood, using NaMaster.

## Layout

```text
cosmic_birefringence/
├── configs/planck_hfi.yml       # paths and numerical settings
├── data/f.npy                   # lightweight static input only
├── notebooks/
│   ├── spectra.ipynb
│   └── mcmc.ipynb
├── src/cosmic_bire/
│   ├── powerspec.py             # Spectra
│   ├── spectratheory.py         # SpectraTheory
│   ├── likelihood.py            # CBlike
│   ├── _reference_likelihood.py # reference likelihood implementation
│   └── tools_fast.py            # unchanged reference numerical kernels
├── pyproject.toml
└── setup.py
```

Every generated product is stored below:

```text
/global/homes/l/lonappan/pscratch/CBDATA/
├── spectra/raw
├── spectra/binned
├── namaster_workspaces
├── theory
├── covariance
├── chains
├── plots
└── results
```

## Install

```bash
module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/cosmic_birefringence
python -m pip install -e .
```

## Spectra API

```python
from cosmic_bire import Spectra

config = "configs/planck_hfi.yml"
spectra = Spectra(config)
spectra.compute_raw_spectra()
spectra.compute_binned_spectra()  # defaults: ell 51–1490, width 20
spectra.plot_spectra(binned=True, freq=143, choose="EE", which="both")
spectra.plot_spectra_matrix(binned=True, choose="EB")
```

`choose` accepts `EE`, `BB`, or `EB`; `which` accepts `A`, `B`, or `both`.
Custom binning can be requested with
`compute_binned_spectra(bin_width=30, lmin=51, lmax=1491)`.

## Likelihood API

```python
from cosmic_bire import CBlike

lh = CBlike("configs/planck_hfi.yml", backend="fortran", threads=4)
lh.precompute()
lh.run_sampler(nwalkers=32, nstep=100000, progress=True, resume=True)
lh.plot_corner()
```

`precompute()` creates or loads the original QuickPol-smoothed LCDM theory,
dust ψℓ, and analytic likelihood covariance. `resume=True` appends samples to
the HDF chain instead of resetting it. Corner plots are generated with GetDist.

### Likelihood backends

Both numerical backends are retained and selectable either in YAML or in the
constructor:

```yaml
likelihood:
  backend: fortran  # fortran or python
  threads: 4
```

```python
lh = CBlike(config, backend="fortran", threads=4)  # OpenMP Fortran
lh = CBlike(config, backend="python", threads=4)   # Numba reference
```

The Fortran backend replaces the repeated likelihood matrix construction,
Cholesky solve, and log-determinant loop. It parallelizes independent ell bins
with OpenMP. Covariance precomputation and diagnostic helpers continue to use
the unchanged reference `tools_fast.py`, since they are not the MCMC hot path.
BLAS remains single-threaded to avoid nested OpenMP/BLAS pools.

The compiled extension is included for the NERSC `cb` environment. To rebuild:

```bash
module load python
conda activate cb
python -m pip install meson ninja
./scripts/build_fortran.sh
python -m pip install -e .
```

On the current node, one real 72-bin likelihood evaluation took approximately
0.110, 0.062, 0.027, and 0.025 seconds with 1, 2, 4, and 8 Fortran threads,
respectively. Numerical agreement with the Python equations was at the
approximately 1e-12 absolute level in tested Planck bins.

## Preserved numerical behavior

- NaMaster uses unit-width, mode-decoupled bands for the raw spectra.
- Instrumental beams and pixel windows remain in observed spectra and are
  applied to LCDM theory with QuickPol, as in the reference code.
- The likelihood equations, dust-EB model, priors, and covariance kernels are
  retained. `tools_fast.py` is unchanged from the supplied reference.
- YAML defaults retain `lmin=51`, `lmax=1491`, and likelihood bin width 20.
