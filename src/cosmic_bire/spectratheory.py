from __future__ import annotations

from pathlib import Path

import camb
import healpy as hp
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

from .configuration import ensure_output_paths, load_config


class SpectraTheory:
    """Prepare the reference QuickPol LCDM spectra and dust psi_ell."""

    def __init__(self, config: str | Path | dict):
        self.config, self.config_path = load_config(config)
        self.paths = ensure_output_paths(self.config)
        s = self.config["spectra"]
        self.frequencies = tuple(str(x) for x in s["frequencies"])
        self.splits = tuple(str(x) for x in s.get("splits", ["A", "B"]))
        self.nside = int(s["nside"])

    @property
    def lcdm_path(self):
        return self.paths["theory"] / f"beam_corrected_lcdm_spectra_{'_'.join(self.frequencies)}.npy"

    @property
    def psi_path(self):
        return self.paths["theory"] / "psi_l_sigma15.npy"

    def compute_beam_corrected_lcdm(self, overwrite=False):
        if self.lcdm_path.exists() and not overwrite:
            print(f"Using existing {self.lcdm_path}")
            return np.load(self.lcdm_path)
        t = self.config["theory"]
        lmax = int(t.get("lmax", 1600))
        params = camb.set_params(tau=t["tau"], ns=t["ns"], H0=t["H0"], ombh2=t["ombh2"],
                                 omch2=t["omch2"], As=t["As"], lmax=lmax)
        theory = camb.get_results(params).get_cmb_power_spectra(lmax=lmax, raw_cl=True)["total"]
        pixwin = hp.pixwin(self.nside, pol=True, lmax=lmax)
        out = np.zeros((lmax + 1, len(self.frequencies), len(self.frequencies), 2, 2, 2))
        quickpol = Path(self.config["inputs"]["quickpol_root"])
        for i, freq1 in enumerate(self.frequencies):
            for j, freq2 in enumerate(self.frequencies):
                for im, split1 in enumerate(self.splits):
                    for jm, split2 in enumerate(self.splits):
                        path = quickpol / f"Wl_npipe6v20_{freq1}{split1}x{freq2}{split2}.fits"
                        with fits.open(path, memmap=False) as window:
                            wee = np.asarray(window["EE"].data["EE_2_EE"][0, :lmax + 1])
                            wbb = np.asarray(window["BB"].data["BB_2_BB"][0, :lmax + 1])
                        out[:, i, j, im, jm, 0] = theory[:, 1] * pixwin[1] ** 2 * wee * 2.7255 ** 2
                        out[:, i, j, im, jm, 1] = theory[:, 2] * pixwin[1] ** 2 * wbb * 2.7255 ** 2
        np.save(self.lcdm_path, out)
        print(f"Wrote {self.lcdm_path}")
        return out

    def calculate_psi_ell(self, overwrite=False):
        if self.psi_path.exists() and not overwrite:
            print(f"Using existing {self.psi_path}")
            return np.load(self.psi_path)
        s, t = self.config["spectra"], self.config["theory"]
        lmin, lmax, width = int(s.get("bin_lmin", 51)), int(s.get("bin_lmax", 1491)), int(s.get("bin_width", 20))
        masks = tuple(str(x) for x in s.get("masks", ["0", "30"]))
        result = np.zeros(((lmax - lmin) // width, len(masks)))
        high = self.frequencies[-1]
        for column, mask in enumerate(masks):
            path = self.paths["raw"] / f"mask_percent_{mask}" / f"{high}A_{high}B.dat"
            if not path.exists():
                raise FileNotFoundError(f"Raw TE/TB spectrum required for psi_ell: {path}")
            spectra = np.loadtxt(path)
            symmetric_te = 0.5 * (spectra[:lmax, 4] + spectra[:lmax, 7])
            symmetric_tb = 0.5 * (spectra[:lmax, 5] + spectra[:lmax, 8])
            te = np.array([symmetric_te[start:start + width].mean() for start in range(lmin, lmax, width)])
            tb = np.array([symmetric_tb[start:start + width].mean() for start in range(lmin, lmax, width)])
            sigma = float(t.get("psi_sigma", 1.5))
            result[:, column] = 0.5 * np.arctan2(gaussian_filter1d(tb, sigma), gaussian_filter1d(te, sigma))
        np.save(self.psi_path, result)
        print(f"Wrote {self.psi_path}")
        return result

    def precompute(self, overwrite=False):
        return {
            "lcdm": self.compute_beam_corrected_lcdm(overwrite=overwrite),
            "psi_ell": self.calculate_psi_ell(overwrite=overwrite),
        }
