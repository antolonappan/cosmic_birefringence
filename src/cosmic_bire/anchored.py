"""Anchored birefringence estimate from low-foreground channels.

Stage 3 of the closure program:
  (1) MK likelihood gives (beta, alpha_i) per mask, degeneracy broken by the
      dust EB ansatz.
  (2) RelCal validates the MK alpha *differences* without any dust model.
  (3) This module fits the total rotation theta_m = alpha_m + beta of the
      low-foreground maps (default 100 and 143 GHz, A/B splits) with the
      MK likelihood machinery restricted to those maps and *no* dust term,
      then anchors alpha on the MK posterior to extract beta:

          beta_m = theta_m - alpha_m^MK,   combined by GLS over maps,

      or, more rigorously, samples the joint posterior

          -2 ln L(beta, alpha) = chi2_EB(theta = alpha + beta)
                                 + (alpha - alpha_MK)^T C_MK^{-1} (alpha - alpha_MK),

      mirroring the SAT-anchored LAT likelihood of the SO forecast.

Because the theta fit contains no foreground model, the entire dust-model
dependence of beta is compressed into the anchor. Fitting theta on the 30%
Galactic mask while anchoring alpha from the full-sky MK chain turns the
historical f_sky dependence of beta into a null test: alpha is an instrument
property, so the anchored beta must be mask stable.

Caveats carried explicitly: (i) on the same mask the anchor and the theta fit
share data, so the quadrature error double counts; the mask-30 theta with the
mask-0 anchor is the cleaner statement. (ii) The common offset of alpha (the
direction degenerate with beta) is inherited 1:1 from the MK anchor; RelCal
validates only the differences.

Data conventions follow the reference pipeline exactly:
ocs[i, j, im, jm, {EE,BB,EB}, ell], theory c_l_th[ell, i, j, im, jm, {EE,BB}],
map index m = nob*im + i, radians internally, Knox covariance via
tools_fast.get_covariance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import tools_fast as tf
from .configuration import ensure_output_paths, load_config

DEG = 180.0 / np.pi


class AnchoredBeta:
    def __init__(self, config, mask="0", freqs=("100", "143"),
                 spectra_file=None, lcdm_file=None, lmin=None, lmax=None,
                 bin_width=None):
        self.config, _ = load_config(config)
        self.paths = ensure_output_paths(self.config)
        s, like = self.config["spectra"], self.config["likelihood"]
        all_freqs = [str(x) for x in s["frequencies"]]
        self.freqs = [str(f) for f in freqs]
        self.idx = [all_freqs.index(f) for f in self.freqs]
        self.nob = len(self.freqs)
        self.nmaps = 2 * self.nob
        self.splits = [str(x) for x in s.get("splits", ["A", "B"])]
        self.mask = str(mask)
        masks = [str(x) for x in s.get("masks", ["0", "30"])]
        self.f_sky = float(np.load(self.config["inputs"]["f_sky_file"])[masks.index(self.mask)])
        self.l_min = int(lmin if lmin is not None else like.get("lmin", 51))
        self.l_max = int(lmax if lmax is not None else like.get("lmax", 1491))
        self.l_bin = int(bin_width if bin_width is not None else like.get("bin_width", 20))
        assert (self.l_max - self.l_min) % self.l_bin == 0
        self.n_bins = (self.l_max - self.l_min) // self.l_bin

        tag = "_".join(all_freqs)
        sp = Path(spectra_file) if spectra_file else self.paths["raw"] / f"cl_mask_percent_{self.mask}_freq_{tag}.npy"
        th = Path(lcdm_file) if lcdm_file else self.paths["theory"] / f"beam_corrected_lcdm_spectra_{tag}.npy"
        ocs_full = np.load(sp)
        cth_full = np.load(th)
        ix = np.asarray(self.idx)
        self.ocs = ocs_full[np.ix_(ix, ix)]                      # (nob, nob, 2, 2, 3, L)
        self.c_l_th = cth_full[:, ix][:, :, ix]                   # (L, nob, nob, 2, 2, 2)
        self._assemble()

    # ------------------------------------------------------------- assembly
    def map_label(self, m):
        return f"{self.freqs[m % self.nob]}{self.splits[m // self.nob]}"

    def _assemble(self):
        nob, size = self.nob, tf.size(self.nob)
        self.size = size
        f = np.ones((nob, 2)) * self.f_sky
        L = min(self.ocs.shape[-1] - 1, self.c_l_th.shape[0] - 1, self.l_max)
        obs_cov = tf.get_covariance(size, L, nob, self.ocs, f)
        self.c_l_o_bin = np.zeros((self.n_bins, size * 3))
        self.c_l_th_bin = np.zeros((self.n_bins, size * 2))
        self.obs_cov_bin = np.zeros((self.n_bins, size * 3, size * 3))
        for lb in range(self.n_bins):
            for l0 in range(self.l_bin):
                ell = self.l_min + self.l_bin * lb + l0
                vi = 0
                for i in range(nob):
                    for j in range(nob):
                        for im in range(2):
                            for jm in range(2):
                                if tf.do_not_count_this_eq(i, j, im, jm):
                                    continue
                                self.c_l_th_bin[lb, 2*vi:2*vi+2] += self.c_l_th[ell, i, j, im, jm, :2]
                                self.c_l_o_bin[lb, 3*vi:3*vi+3] += self.ocs[i, j, im, jm, :3, ell]
                                vi += 1
                self.obs_cov_bin[lb] += obs_cov[:, :, ell]
            self.c_l_o_bin[lb] /= self.l_bin
            self.c_l_th_bin[lb] /= self.l_bin
            self.obs_cov_bin[lb] /= self.l_bin ** 2

    # -------------------------------------------------------------- theta fit
    def loglike_theta(self, theta_rad):
        """MK likelihood restricted to the chosen maps, no dust term, beta=0:
        the model depends only on the total rotations theta_m."""
        return tf.likelihood_prob(self.c_l_o_bin, self.c_l_th_bin, self.obs_cov_bin,
                                  False, self.n_bins, self.nob,
                                  np.asarray(theta_rad, dtype=float),
                                  np.zeros(self.nob))

    def fit_theta(self, x0_deg=None, hstep=5e-4):
        """ML total-rotation angles theta_m (deg) with Hessian covariance."""
        x0 = np.zeros(self.nmaps) if x0_deg is None else np.asarray(x0_deg) / DEG
        res = minimize(lambda t: -self.loglike_theta(t), x0, method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 20000})
        res = minimize(lambda t: -self.loglike_theta(t), res.x, method="Powell",
                       options={"xtol": 1e-9, "ftol": 1e-11})
        t = res.x
        n = self.nmaps
        H = np.zeros((n, n))
        for a in range(n):
            for b in range(a, n):
                ea, eb = np.eye(n)[a] * hstep, np.eye(n)[b] * hstep
                H[a, b] = H[b, a] = (
                    - self.loglike_theta(t + ea + eb) + self.loglike_theta(t + ea - eb)
                    + self.loglike_theta(t - ea + eb) - self.loglike_theta(t - ea - eb)
                ) / (4 * hstep ** 2)
        cov = np.linalg.inv(H)  # H as built equals -Hessian = Fisher
        self.theta = t * DEG
        self.theta_cov = cov * DEG ** 2
        return self.theta, self.theta_cov

    # --------------------------------------------------------------- anchor
    def anchor_from_chain(self, samples, chain_freqs=None):
        """Extract the MK alpha posterior for the chosen maps from a full chain.

        samples columns: [beta, alpha_(all A), alpha_(all B), A_ell...] in deg;
        chain_freqs defaults to the config spectra frequencies. Returns
        (mean_deg[nmaps], cov_deg2[nmaps, nmaps]) in this object's map order.
        """
        samples = np.asarray(samples)
        chain_freqs = [str(x) for x in (chain_freqs or self.config["spectra"]["frequencies"])]
        nchain = len(chain_freqs)
        cols = []
        for m in range(self.nmaps):
            i, im = m % self.nob, m // self.nob
            cols.append(1 + nchain * im + chain_freqs.index(self.freqs[i]))
        a = samples[:, cols]
        return a.mean(0), np.cov(a.T)

    # ------------------------------------------------------------ estimators
    def beta_profile(self, anchor_mean, anchor_cov, theta=None, theta_cov=None,
                     cross_cov=None):
        """GLS-combined beta from beta_m = theta_m - alpha_m^MK.

        C = C_theta + C_anchor - 2*C_cross; cross_cov (deg^2) defaults to zero,
        which double counts shared data when the anchor mask equals the theta
        mask; pass the simulation-calibrated cross term when available.
        Returns (beta_deg, sigma_deg, per_map_beta, per_map_sigma).
        """
        if theta is None:
            theta, theta_cov = (self.theta, self.theta_cov) if hasattr(self, "theta") else self.fit_theta()
        b = np.asarray(theta) - np.asarray(anchor_mean)
        C = np.asarray(theta_cov) + np.asarray(anchor_cov)
        if cross_cov is not None:
            C = C - 2.0 * np.asarray(cross_cov)
        Cinv = np.linalg.inv(C)
        one = np.ones(self.nmaps)
        var = 1.0 / (one @ Cinv @ one)
        beta = var * (one @ Cinv @ b)
        return beta, np.sqrt(var), b, np.sqrt(np.diag(C))

    def joint_mcmc(self, anchor_mean, anchor_cov, nwalkers=32, nsteps=4000,
                   burn=1000, progress=False, seed=0):
        """Sample (beta, alpha_m) with the anchored joint likelihood

        -2 lnL = chi2_EB(theta = alpha + beta)
                 + (alpha - alpha_MK)^T C_MK^{-1} (alpha - alpha_MK).

        Returns samples in degrees, columns [beta, alpha_m...]. The beta
        marginal is the anchored estimate; the alpha marginals should
        reproduce the anchor (the EB block constrains only theta).
        """
        import emcee
        mean_r = np.asarray(anchor_mean) / DEG
        Cinv = np.linalg.inv(np.asarray(anchor_cov) / DEG ** 2)

        def logpost(p):
            beta, alpha = p[0], p[1:]
            da = alpha - mean_r
            return self.loglike_theta(alpha + beta) - 0.5 * da @ Cinv @ da

        rng = np.random.default_rng(seed)
        ndim = 1 + self.nmaps
        p0 = np.concatenate(([0.3 / DEG], mean_r))
        pos = p0 + 1e-3 / DEG * rng.standard_normal((nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost)
        sampler.run_mcmc(pos, nsteps, progress=progress)
        return sampler.get_chain(discard=burn, flat=True) * DEG

    # ------------------------------------------------------------ mask test
    @classmethod
    def mask_stability(cls, config, chain_samples, masks=("0", "30"),
                       freqs=("100", "143"), anchor_mask_chain_freqs=None,
                       spectra_files=None, lcdm_file=None,
                       run_mcmc=False, **mcmc_kw):
        """Anchored beta per mask with the alpha anchor held fixed
        (from the supplied chain, normally the mask-0 MK run).
        spectra_files: optional {mask: path} overriding the raw-path
        convention; lcdm_file: optional theory-file override."""
        out = {}
        anchor = None
        for mask in masks:
            sp = (spectra_files or {}).get(mask)
            ab = cls(config, mask=mask, freqs=freqs,
                     spectra_file=sp, lcdm_file=lcdm_file)
            if anchor is None:
                anchor = ab.anchor_from_chain(chain_samples, chain_freqs=anchor_mask_chain_freqs)
            theta, tcov = ab.fit_theta()
            beta, sig, bm, sm = ab.beta_profile(anchor[0], anchor[1], theta, tcov)
            rec = {"theta": theta, "theta_cov": tcov, "beta": beta, "sigma": sig,
                   "beta_per_map": bm, "sigma_per_map": sm,
                   "labels": [ab.map_label(m) for m in range(ab.nmaps)]}
            if run_mcmc:
                ch = ab.joint_mcmc(anchor[0], anchor[1], **mcmc_kw)
                rec["mcmc_beta"] = (ch[:, 0].mean(), ch[:, 0].std())
                rec["mcmc_chain"] = ch
            out[mask] = rec
        return out