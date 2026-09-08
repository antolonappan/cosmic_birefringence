from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .configuration import ensure_output_paths, load_config

DEG = 180.0 / np.pi
_trapz = getattr(np, "trapezoid", None) or np.trapz


class RelCal:
    """Pairwise and network relative-angle estimation from a raw spectra file."""

    def __init__(self, config, mask=None, spectra_file=None):
        self.config, self.config_path = load_config(config)
        self.paths = ensure_output_paths(self.config)
        s, like = self.config["spectra"], self.config["likelihood"]
        self.frequencies = [str(x) for x in s["frequencies"]]
        self.splits = [str(x) for x in s.get("splits", ["A", "B"])]
        self.nob = len(self.frequencies)
        self.nmaps = 2 * self.nob
        self.mask = str(mask if mask is not None else like.get("mask", "0"))
        masks = [str(x) for x in s.get("masks", ["0", "30"])]
        self.f_sky = float(np.load(self.config["inputs"]["f_sky_file"])[masks.index(self.mask)])
        self.l_min = int(like.get("lmin", 51))
        self.l_max = int(like.get("lmax", 1491))
        self.l_bin = int(like.get("bin_width", 20))
        assert (self.l_max - self.l_min) % self.l_bin == 0
        self.n_bins = (self.l_max - self.l_min) // self.l_bin
        tag = "_".join(self.frequencies)
        default = self.paths["raw"] / f"cl_mask_percent_{self.mask}_freq_{tag}.npy"
        self.spectra_file = Path(spectra_file) if spectra_file else default
        self.ocs = np.load(self.spectra_file)
        self.pairs = list(combinations(range(self.nmaps), 2))  # unordered m<n
        self._prepare()

    # ---------------------------------------------------------------- labels
    def map_label(self, m):
        return f"{self.frequencies[m % self.nob]}{self.splits[m // self.nob]}"

    def _fi(self, m):
        """Map index -> (frequency index i, split index im)."""
        return m % self.nob, m // self.nob

    # ------------------------------------------------------------- internals
    def _cl(self, m, n, k):
        """C^{X_m Y_n} with k in {0:EE, 1:BB, 2:E_m B_n}, all ells."""
        i, im = self._fi(m)
        j, jm = self._fi(n)
        return self.ocs[i, j, im, jm, k]

    def _bin(self, arr_l):
        ells = np.arange(self.l_min, self.l_max)
        sel = arr_l[..., self.l_min:self.l_max]
        return sel.reshape(*arr_l.shape[:-1], self.n_bins, self.l_bin).mean(-1), ells

    def _prepare(self):
        """Per-ell X, Y for all pairs, binned, plus binned Knox covariances."""
        P, L = len(self.pairs), self.ocs.shape[-1]
        ells = np.arange(L)
        nu = (2.0 * ells + 1.0) * self.f_sky
        X = np.zeros((P, L))
        Y = np.zeros((P, L))
        for p, (m, n) in enumerate(self.pairs):
            X[p] = self._cl(m, n, 0) + self._cl(m, n, 1)
            Y[p] = self._cl(n, m, 2) - self._cl(m, n, 2)  # C^{E_n B_m} - C^{E_m B_n}
        # Knox covariance of Y between pairs, EB-dropping convention:
        # Cov(Y_mn, Y_pq) = [E_nq B_mp - E_np B_mq - E_mq B_np + E_mp B_nq]/nu
        covY = np.zeros((P, P, L))
        EE = lambda a, b: self._cl(a, b, 0)
        BB = lambda a, b: self._cl(a, b, 1)
        for p, (m, n) in enumerate(self.pairs):
            for q in range(p, P):
                r, s_ = self.pairs[q]
                c = (EE(n, s_) * BB(m, r) - EE(n, r) * BB(m, s_)
                     - EE(m, s_) * BB(n, r) + EE(m, r) * BB(n, s_)) / nu
                covY[p, q] = covY[q, p] = c
        # Cov(X_mn, X_pq) = [E_mp E_nq + E_mq E_np + B_mp B_nq + B_mq B_np]/nu
        covX = np.zeros((P, P, L))
        for p, (m, n) in enumerate(self.pairs):
            for q in range(p, P):
                r, s_ = self.pairs[q]
                c = (EE(m, r) * EE(n, s_) + EE(m, s_) * EE(n, r)
                     + BB(m, r) * BB(n, s_) + BB(m, s_) * BB(n, r)) / nu
                covX[p, q] = covX[q, p] = c
        # bin: means /dl, covariances /dl^2
        w = self.l_bin
        sl = slice(self.l_min, self.l_max)
        self.X_b = X[:, sl].reshape(P, self.n_bins, w).mean(-1)
        self.Y_b = Y[:, sl].reshape(P, self.n_bins, w).mean(-1)
        self.covY_b = covY[:, :, sl].reshape(P, P, self.n_bins, w).sum(-1) / w ** 2
        self.covX_b = covX[:, :, sl].reshape(P, P, self.n_bins, w).sum(-1) / w ** 2
        self.varX_b = np.einsum("ppb->pb", self.covX_b)
        self.varY_b = np.einsum("ppb->pb", self.covY_b)

    # ------------------------------------------------------------- per pair
    def fit_pair(self, m, n, grid_deg=1.0, npts=4001):
        """Exact 1-parameter likelihood for Dalpha_mn = alpha_m - alpha_n.

        -2 lnL(D) = sum_b [Y_b cos(2D) - X_b sin(2D)]^2 / [cos^2 VarY + sin^2 VarX]
        (Cov(X, Y) = 0 under the EB-dropping Knox convention).
        Returns posterior mean, std in degrees.
        """
        p = self.pairs.index((min(m, n), max(m, n)))
        sign = 1.0 if m < n else -1.0
        d = np.linspace(-grid_deg, grid_deg, npts) / DEG
        c, s = np.cos(2 * d)[:, None], np.sin(2 * d)[:, None]
        z = self.Y_b[p][None, :] * c - self.X_b[p][None, :] * s
        var = c ** 2 * self.varY_b[p][None, :] + s ** 2 * self.varX_b[p][None, :]
        chi2 = (z ** 2 / var).sum(1)
        post = np.exp(-0.5 * (chi2 - chi2.min()))
        post /= _trapz(post, d)
        mean = _trapz(post * d, d)
        std = np.sqrt(_trapz(post * (d - mean) ** 2, d))
        return sign * mean * DEG, std * DEG

    def fit_all_pairs(self):
        out = {}
        for (m, n) in self.pairs:
            out[(self.map_label(m), self.map_label(n))] = self.fit_pair(m, n)
        return out

    # -------------------------------------------------------------- network
    def fit_network(self, ref=None, debias=False, rcond=1e-10):
        """Small-angle GLS for the map-angle vector with full pair covariance.

        Model per bin: Y_b^{mn} = 2 (alpha_m - alpha_n) S_b^{mn}, template
        S_b -> X_b (its noise is negligible for HFI; ``debias`` subtracts the
        template-noise contribution from the normal matrix diagonal).
        Gauge: alpha_ref = 0 if ``ref`` given (map index or label), otherwise
        sum(alpha) = 0. Returns (alpha_deg, cov_deg2, labels).
        """
        P, K = len(self.pairs), self.nmaps
        D = np.zeros((P, K))
        for p, (m, n) in enumerate(self.pairs):
            D[p, m], D[p, n] = 1.0, -1.0
        F = np.zeros((K, K))
        g = np.zeros(K)
        for b in range(self.n_bins):
            N = self.covY_b[:, :, b]
            # eigenmode-projected inverse: with a common dust template, pair
            # fluctuations are strongly correlated and N is near-singular;
            # discard modes below rcond * max eigenvalue instead of inverting.
            evals, evecs = np.linalg.eigh(N)
            keep = evals > rcond * evals.max()
            Ninv = (evecs[:, keep] / evals[keep]) @ evecs[:, keep].T
            t = self.X_b[:, b]
            Td = 2.0 * (D * t[:, None])           # d<Y>/d alpha
            F += Td.T @ Ninv @ Td
            g += Td.T @ Ninv @ self.Y_b[:, b]
            if debias:
                # E[t_p Ninv_pq t_q] = tbar_p tbar_q + Cov(X_p, X_q):
                # subtract 4 D^T (Ninv o CovX) D  (elementwise product)
                F -= 4.0 * D.T @ (Ninv * self.covX_b[:, :, b]) @ D
        # gauge fixing
        if ref is None:
            constraint = np.ones(K)
        else:
            if isinstance(ref, str):
                ref = [self.map_label(k) for k in range(K)].index(ref)
            constraint = np.eye(K)[ref]
        Faug = F + np.outer(constraint, constraint) * F.max()
        cov = np.linalg.inv(Faug)
        alpha = cov @ g
        # project out the gauge direction from the covariance
        return alpha * DEG, cov * DEG ** 2, [self.map_label(k) for k in range(K)]

    # -------------------------------------------------------------- closure
    def closure_vs_mk(self, samples, ref=None, number_of_A=None):
        """Closure test against an MK chain (CBlike ``samples`` array, degrees).

        samples columns: [beta, alpha_(all A splits), alpha_(all B splits), A...]
        Returns dict with per-map d = alpha_rel - alpha_MK (both gauge-fixed to
        the same reference) and naive-quadrature significance. The naive
        significance ignores the estimator-MK correlation (same data!); the
        publication-grade covariance must come from running both pipelines on
        simulations, see ``fit_from_file``.
        """
        samples = np.asarray(samples)
        n_a = number_of_A if number_of_A is not None else int(self.config["likelihood"].get("number_of_A", 4))
        alpha_mk = samples[:, 1:1 + self.nmaps]
        mk_mean = alpha_mk.mean(0)
        mk_cov = np.cov(alpha_mk.T)
        a_rel, c_rel, labels = self.fit_network(ref=ref)
        if ref is None:
            proj = np.eye(self.nmaps) - 1.0 / self.nmaps
        else:
            k = labels.index(ref) if isinstance(ref, str) else ref
            proj = np.eye(self.nmaps)
            proj[:, k] -= 1.0
        d = proj @ a_rel - proj @ mk_mean
        cov_naive = proj @ c_rel @ proj.T + proj @ mk_cov @ proj.T
        sig = d / np.sqrt(np.abs(np.diag(cov_naive)) + 1e-30)
        return {"labels": labels, "alpha_rel": proj @ a_rel, "alpha_mk": proj @ mk_mean,
                "d": d, "naive_sigma": sig}

    # ----------------------------------------------------------------- sims
    @classmethod
    def fit_from_file(cls, config, spectra_file, mask=None, ref=None):
        """Run the network fit on one (simulation) spectra file."""
        rc = cls(config, mask=mask, spectra_file=spectra_file)
        return rc.fit_network(ref=ref)
