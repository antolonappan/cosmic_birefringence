from __future__ import annotations

from pathlib import Path

import emcee
from getdist import MCSamples, plots
import matplotlib.pyplot as plt
import numpy as np

from ._reference_likelihood import cosmic_birefringence
from .configuration import ensure_output_paths, load_config
from .spectratheory import SpectraTheory


class CBlike:
    """Friendly wrapper around the unchanged reference likelihood equations."""

    def __init__(self, config: str | Path | dict, backend: str | None = None,
                 threads: int | None = None):
        self.config, self.config_path = load_config(config)
        if backend is not None:
            self.config["likelihood"]["backend"] = str(backend).lower()
        if threads is not None:
            self.config["likelihood"]["threads"] = int(threads)
        self.paths = ensure_output_paths(self.config)
        self.analysis = None
        self.sampler = None
        self.samples = None

    @property
    def backend(self):
        return str(self.config["likelihood"].get("backend", "python")).lower()

    def _configure_backend_threads(self):
        like = self.config["likelihood"]
        backend = str(like.get("backend", "python")).lower()
        threads = int(like.get("threads", 1))
        if backend == "fortran":
            from . import tools_fortran
            tools_fortran.set_num_threads(threads)
        elif backend == "python":
            import numba
            numba.set_num_threads(threads)
        else:
            raise ValueError("likelihood.backend must be 'python' or 'fortran'")
        return backend

    def _parameters(self):
        s, like = self.config["spectra"], self.config["likelihood"]
        frequencies = [str(x) for x in s["frequencies"]]
        mask = str(like.get("mask", "30"))
        masks = [str(x) for x in s.get("masks", ["0", "30"])]
        tag = "_".join(frequencies)
        backend = self._configure_backend_threads()
        return {
            "alpha_labels": frequencies,
            "initial_beta": 0.0,
            "nob": len(frequencies),
            "initial_alphas": np.zeros(len(frequencies)),
            "alpha_no_split_list": [],
            "l_min": int(like.get("lmin", 51)),
            "l_max": int(like.get("lmax", 1491)),
            "l_bin": int(like.get("bin_width", 20)),
            "data_set": "cl_hfi",
            "flat_beta_prior": float(like.get("flat_beta_prior", 5.0)),
            "flat_alpha_prior": float(like.get("flat_alpha_prior", 5.0)),
            "cl_folder": f"mask_percent_{mask}",
            "index": masks.index(mask),
            "with_ln_det_cov": bool(like.get("with_ln_det_cov", True)),
            "backend": True,
            "computation_backend": backend,
            "frequency_dependent_beta": False,
            "psi_l": str(self.paths["theory"] / "psi_l_sigma15.npy"),
            "number_of_A": int(like.get("number_of_A", 4)),
            "turn_off_for_lowest_indices": int(like.get("turn_off_for_lowest_indices", 0)),
            "upper_bound_A": float(like.get("upper_bound_A", 1.0)),
            "f_sky_file": str(Path(self.config["inputs"]["f_sky_file"])),
            "spectra_file": str(self.paths["raw"] / f"cl_mask_percent_{mask}_freq_{tag}.npy"),
            "lcdm_file": str(self.paths["theory"] / f"beam_corrected_lcdm_spectra_{tag}.npy"),
            "covariance_file": str(self.paths["covariance"] / f"cov_bin_cl_hfi_mask_percent_{mask}_{tag}_lmin{like.get('lmin',51)}_lmax{like.get('lmax',1491)}.npy"),
        }

    def _required_files(self):
        p = self._parameters()
        return [Path(p[key]) for key in ("f_sky_file", "spectra_file", "lcdm_file", "psi_l")]

    def precompute(self, overwrite=False):
        """Prepare theory/psi and initialize or load the analytic covariance."""
        theory = SpectraTheory(self.config)
        theory.precompute(overwrite=overwrite)
        missing = [path for path in self._required_files() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing prerequisites:\n  " + "\n  ".join(map(str, missing)))
        self.analysis = cosmic_birefringence(self._parameters())
        return self.analysis

    @property
    def parameter_names(self):
        frequencies = [str(x) for x in self.config["spectra"]["frequencies"]]
        n_a = int(self.config["likelihood"].get("number_of_A", 4))
        return ["beta"] + [f"alpha_{f}{split}" for split in ("A", "B") for f in frequencies] + [f"A_{i}" for i in range(n_a)]

    @property
    def parameter_labels(self):
        frequencies = [str(x) for x in self.config["spectra"]["frequencies"]]
        n_a = int(self.config["likelihood"].get("number_of_A", 4))
        return [r"\beta"] + [rf"\alpha_{{{f}{split}}}" for split in ("A", "B") for f in frequencies] + [rf"A_{{{i}}}" for i in range(n_a)]

    def _initial_state(self, nwalkers, rng):
        ndim = len(self.parameter_names)
        if nwalkers < 2 * ndim:
            raise ValueError(f"nwalkers must be at least 2*ndim={2*ndim}")
        state = rng.normal(0.0, 0.1 * np.pi / 180, size=(nwalkers, ndim))
        number_of_A = int(self.config["likelihood"].get("number_of_A", 4))
        if number_of_A:
            state[:, -number_of_A:] = rng.normal(0.05, 0.015, size=(nwalkers, number_of_A))
        return state

    def run_sampler(self, nwalkers=None, nstep=None, burnin=None, progress=True, resume=True):
        if self.analysis is None:
            self.precompute()
        settings = self.config.get("sampler", {})
        nwalkers = int(nwalkers or settings.get("nwalkers", 32))
        nstep = int(nstep or settings.get("nsteps", 100000))
        mask = str(self.config["likelihood"].get("mask", "30"))
        chain_path = self.paths["chains"] / f"planck_hfi_mask_{mask}.h5"
        backend = emcee.backends.HDFBackend(chain_path)
        ndim = len(self.parameter_names)
        continuing = resume and chain_path.exists() and backend.iteration > 0
        if continuing:
            if backend.shape != (nwalkers, ndim):
                raise ValueError(f"Existing chain shape {backend.shape} does not match {(nwalkers, ndim)}")
            initial_state = None
            print(f"Resuming {chain_path} from step {backend.iteration}")
        else:
            backend.reset(nwalkers, ndim)
            rng = np.random.default_rng(int(settings.get("seed", 1234)))
            initial_state = self._initial_state(nwalkers, rng)
            print(f"Starting new chain {chain_path}")
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.analysis.log_prob, backend=backend)
        self.sampler.run_mcmc(initial_state, nstep, progress=progress)
        burnin = int(burnin or settings.get("burnin", 500))
        thin = int(settings.get("thin", 5))
        if backend.iteration <= burnin:
            print(f"Chain has {backend.iteration} steps; burn-in is {burnin}. Returning unburned samples for inspection.")
            burnin = 0
        raw = backend.get_chain(discard=burnin, thin=thin, flat=True)
        self.samples = self._convert_samples(raw)
        np.save(self.paths["chains"] / f"planck_hfi_mask_{mask}_samples.npy", self.samples)
        return self.samples

    def _convert_samples(self, raw):
        """Convert angular chain columns from radians to degrees."""
        samples = np.asarray(raw, dtype=float).copy()
        number_of_A = int(self.config["likelihood"].get("number_of_A", 4))
        stop = -number_of_A if number_of_A else None
        samples[:, :stop] *= 180 / np.pi
        return samples

    def getdist_samples(self):
        if self.samples is None:
            mask = str(self.config["likelihood"].get("mask", "30"))
            path = self.paths["chains"] / f"planck_hfi_mask_{mask}_samples.npy"
            hdf_path = self.paths["chains"] / f"planck_hfi_mask_{mask}.h5"
            # Prefer HDF whenever present: it is the live resumable chain and
            # may be newer than a flattened NPY written by an earlier plot.
            if hdf_path.exists():
                backend = emcee.backends.HDFBackend(hdf_path, read_only=True)
                if backend.iteration == 0:
                    raise ValueError(f"Sampler checkpoint contains no steps: {hdf_path}")
                burnin = int(self.config.get("sampler", {}).get("burnin", 5000))
                thin = int(self.config.get("sampler", {}).get("thin", 5))
                if backend.iteration <= burnin:
                    print(f"Chain has {backend.iteration} steps, below burn-in {burnin}; plotting the full preliminary chain.")
                    burnin = 0
                raw = backend.get_chain(discard=burnin, thin=thin, flat=True)
                self.samples = self._convert_samples(raw)
                np.save(path, self.samples)
                print(f"Loaded {self.samples.shape[0]} samples directly from {hdf_path}")
            elif path.exists():
                self.samples = np.load(path)
            else:
                raise FileNotFoundError("No NPY samples or HDF sampler checkpoint exists")
        return MCSamples(samples=self.samples, names=self.parameter_names, labels=self.parameter_labels)

    def plot_corner(self, parameters=None, filled=True, filename=None, show=True):
        samples = self.getdist_samples()
        plotter = plots.get_subplot_plotter()
        plotter.triangle_plot([samples], params=parameters, filled=filled)
        if filename is None:
            mask = str(self.config["likelihood"].get("mask", "30"))
            filename = self.paths["plots"] / f"corner_planck_hfi_mask_{mask}.pdf"
        plotter.export(str(filename))
        if show:
            plt.show()
        return plotter
