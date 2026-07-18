from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt

from .configuration import ensure_output_paths, load_config


class Spectra:
    """Compute, bin, save, load, and plot Planck HFI NaMaster spectra."""

    spectrum_names = ("EE", "BB", "EB")

    def __init__(self, config: str | Path | dict):
        self.config, self.config_path = load_config(config)
        self.paths = ensure_output_paths(self.config)
        section = self.config["spectra"]
        self.frequencies = tuple(str(x) for x in section["frequencies"])
        self.splits = tuple(str(x) for x in section.get("splits", ["A", "B"]))
        self.masks = tuple(str(x) for x in section.get("masks", ["0", "30"]))
        self.nside = int(section["nside"])
        self.lmax = int(section["lmax"])

    def map_path(self, frequency: str, split: str) -> Path:
        root = Path(self.config["inputs"]["npipe_root"])
        return root / f"npipe6v20{split}" / f"npipe6v20{split}_{frequency}_map.fits"

    def mask_path(self, mask: str) -> Path:
        root = Path(self.config["inputs"]["mask_root"])
        return root / f"combined_ps_co_{mask}_{self.nside}.fits"

    def raw_path(self, mask: str) -> Path:
        return self.paths["raw"] / f"cl_mask_percent_{mask}_freq_{'_'.join(self.frequencies)}.npy"

    def binned_path(self, mask: str, lmin: int, lmax: int, bin_width: int) -> Path:
        return self.paths["binned"] / f"cl_mask_percent_{mask}_lmin{lmin}_lmax{lmax}_bin{bin_width}.npz"

    def _workspace(self, mask_map, spin_a, spin_b, bins, mask, overwrite):
        path = self.paths["workspaces"] / f"mask{mask}_nside{self.nside}_lmax{self.lmax}_s{spin_a}{spin_b}.fits"
        workspace = nmt.NmtWorkspace()
        if path.exists() and not overwrite:
            workspace.read_from(str(path))
            return workspace
        zeros = np.zeros(mask_map.size)
        field_a = nmt.NmtField(mask_map, [zeros] if spin_a == 0 else [zeros, zeros], spin=spin_a, lmax=self.lmax)
        field_b = nmt.NmtField(mask_map, [zeros] if spin_b == 0 else [zeros, zeros], spin=spin_b, lmax=self.lmax)
        workspace.compute_coupling_matrix(field_a, field_b, bins)
        workspace.write_to(str(path))
        return workspace

    def compute_raw_spectra(self, masks=None, overwrite=False, save_dat=True):
        """Compute unit-bandwidth, mode-decoupled map-level C_ell.

        Beam and pixel windows are deliberately retained, exactly as in the
        existing pipeline, since they are applied to the likelihood theory.
        """
        selected_masks = self.masks if masks is None else tuple(str(x) for x in np.atleast_1d(masks))
        bins = nmt.NmtBin.from_lmax_linear(self.lmax, 1)
        labels = [(frequency, split) for frequency in self.frequencies for split in self.splits]
        products = {}
        for mask_name in selected_masks:
            final_path = self.raw_path(mask_name)
            if final_path.exists() and not overwrite:
                products[mask_name] = np.load(final_path)
                print(f"Using existing {final_path}")
                continue
            mask_map = hp.read_map(self.mask_path(mask_name), field=0, dtype=np.float64)
            mask_map = np.where(np.isfinite(mask_map), mask_map, 0.0)
            workspaces = {(a, b): self._workspace(mask_map, a, b, bins, mask_name, overwrite)
                          for a, b in ((0, 0), (0, 2), (2, 0), (2, 2))}
            fields = {}
            purify_e = bool(self.config["spectra"].get("purify_e", False))
            purify_b = bool(self.config["spectra"].get("purify_b", False))
            for label in labels:
                temperature, q_map, u_map = hp.read_map(self.map_path(*label), field=(0, 1, 2), dtype=np.float64)
                fields[(label, 0)] = nmt.NmtField(mask_map, [temperature], spin=0, lmax=self.lmax)
                fields[(label, 2)] = nmt.NmtField(mask_map, [q_map, u_map], spin=2, lmax=self.lmax,
                                                  purify_e=purify_e, purify_b=purify_b)
            cl = np.zeros((len(self.frequencies), len(self.frequencies), 2, 2, 3, self.lmax + 1))
            dat_dir = self.paths["raw"] / f"mask_percent_{mask_name}"
            dat_dir.mkdir(parents=True, exist_ok=True)
            for left, right in combinations_with_replacement(labels, 2):
                c00 = workspaces[(0, 0)].decouple_cell(nmt.compute_coupled_cell(fields[(left, 0)], fields[(right, 0)]))[0]
                c02 = workspaces[(0, 2)].decouple_cell(nmt.compute_coupled_cell(fields[(left, 0)], fields[(right, 2)]))
                c20 = workspaces[(2, 0)].decouple_cell(nmt.compute_coupled_cell(fields[(left, 2)], fields[(right, 0)]))
                c22 = workspaces[(2, 2)].decouple_cell(nmt.compute_coupled_cell(fields[(left, 2)], fields[(right, 2)]))
                full = np.zeros((9, self.lmax + 1))
                full[:, 2:] = np.vstack((c00, c22[0], c22[3], c02[0], c02[1], c22[1], c20[0], c20[1], c22[2]))
                i, im = self.frequencies.index(left[0]), self.splits.index(left[1])
                j, jm = self.frequencies.index(right[0]), self.splits.index(right[1])
                cl[i, j, im, jm] = full[[1, 2, 5]]
                cl[j, i, jm, im] = full[[1, 2, 8]]
                if save_dat:
                    table = np.column_stack((np.arange(self.lmax + 1), full.T))
                    np.savetxt(dat_dir / f"{left[0]}{left[1]}_{right[0]}{right[1]}.dat", table,
                               header="ell TT EE BB TE TB EB ET BT BE")
            np.save(final_path, cl)
            products[mask_name] = cl
            print(f"Wrote {final_path}")
        return products

    def compute_binned_spectra(self, bin_width=None, lmin=None, lmax=None, masks=None, overwrite=False):
        section = self.config["spectra"]
        bin_width = int(bin_width or section.get("bin_width", 20))
        lmin = int(lmin if lmin is not None else section.get("bin_lmin", 51))
        lmax = int(lmax if lmax is not None else section.get("bin_lmax", 1491))
        if (lmax - lmin) % bin_width:
            raise ValueError("lmax-lmin must be divisible by bin_width")
        selected_masks = self.masks if masks is None else tuple(str(x) for x in np.atleast_1d(masks))
        products = {}
        for mask in selected_masks:
            output = self.binned_path(mask, lmin, lmax, bin_width)
            if output.exists() and not overwrite:
                products[mask] = np.load(output)["cl"]
                continue
            raw = np.load(self.raw_path(mask))
            cl = np.stack([raw[..., start:start + bin_width].mean(axis=-1)
                           for start in range(lmin, lmax, bin_width)], axis=-1)
            ell = np.arange(lmin, lmax, bin_width) + (bin_width - 1) / 2
            np.savez(output, ell=ell, cl=cl, lmin=lmin, lmax=lmax, bin_width=bin_width)
            products[mask] = cl
            print(f"Wrote {output}")
        return products

    def _plot_data(self, binned, mask):
        if binned:
            s = self.config["spectra"]
            path = self.binned_path(mask, int(s.get("bin_lmin", 51)), int(s.get("bin_lmax", 1491)), int(s.get("bin_width", 20)))
            data = np.load(path)
            return data["ell"], data["cl"]
        return np.arange(self.lmax + 1), np.load(self.raw_path(mask))

    def plot_spectra(self, binned=True, freq=143, choose="EE", which="both", mask="30", ax=None):
        choose, which, freq = choose.upper(), which.upper(), str(freq)
        if choose not in self.spectrum_names or which not in ("A", "B", "BOTH"):
            raise ValueError("choose must be EE/BB/EB and which must be A/B/both")
        ell, cl = self._plot_data(binned, str(mask))
        ax = ax or plt.subplots(figsize=(8, 5))[1]
        i, k = self.frequencies.index(freq), self.spectrum_names.index(choose)
        selected = range(2) if which == "BOTH" else [self.splits.index(which)]
        for split in selected:
            # Show the same-frequency cross split by default (noise-bias free).
            other = 1 - split
            ax.loglog(ell, cl[i, i, split, other, k], label=f"{freq}{self.splits[split]}×{freq}{self.splits[other]}")
        ax.set(xlabel=r"$\ell$", ylabel=rf"$C_\ell^{{{choose}}}$ [K$^2$]")
        ax.legend()
        return ax

    def plot_spectra_matrix(self, binned=True, choose="EB", mask="30", ell_bin=None, ax=None):
        choose = choose.upper()
        ell, cl = self._plot_data(binned, str(mask))
        k = self.spectrum_names.index(choose)
        # Average A×B and B×A for each frequency pair, then show one ell bin.
        matrix = 0.5 * (cl[:, :, 0, 1, k] + cl[:, :, 1, 0, k])
        index = matrix.shape[-1] // 2 if ell_bin is None else int(ell_bin)
        ax = ax or plt.subplots(figsize=(6, 5))[1]
        image = ax.imshow(matrix[..., index], origin="lower")
        ax.set(xticks=range(len(self.frequencies)), yticks=range(len(self.frequencies)),
               xticklabels=self.frequencies, yticklabels=self.frequencies,
               title=f"{choose}, ell={ell[index]:.1f}")
        ax.figure.colorbar(image, ax=ax, label="K$^2$")
        return ax
