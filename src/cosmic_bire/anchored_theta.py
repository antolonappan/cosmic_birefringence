"""Step 3 of the closure program: anchored birefringence from 100/143 GHz.

Pipeline (fully analytic, no simulations):

  1. Load the mask-0 MK chain (validated against RelCal in step 2).
  2. Fit the total rotations theta_m = alpha_m + beta of the 100/143 GHz
     A/B maps with the MK likelihood machinery restricted to those four
     maps and *no* dust term (low-foreground channels; degeneracy with
     beta is not broken internally -- that's the anchor's job).
  3. Anchor alpha from the mask-0 MK posterior (its four 100/143 entries)
     and extract beta_m = theta_m - alpha_m^MK, combined by GLS.
  4. Repeat the theta fit on the 30% Galactic mask with the SAME anchor.
     alpha is an instrument property, so if the historical f_sky
     dependence of MK's beta is a dust-degeneracy artifact rather than a
     real signal, the anchored beta must be stable between mask 0 and
     mask 30 -- this is the test.

Errors are analytic throughout:
  - sigma(theta) from the Fisher matrix of the restricted likelihood
    (AnchoredBeta.fit_theta).
  - sigma(alpha_MK) from the MK chain covariance (posterior, used as-is;
    optionally cross-checked against the MKResponse Fisher matrix).
  - the theta-anchor cross term via analytic_cov.anchored_beta_cov:
    cross="same" on mask 0 (theta and anchor share the identical data --
    the naive quadrature sum double counts and is NOT used there), and
    cross="nested" on mask 30 (nested-mask Knox: Cov(c^0, c^30) = Var(c^0)
    for the mask-30 region inside the mask-0 region).

Run as a script:  python scripts/anchored_theta.py [config]
or import and call ``run(config)`` from a notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosmic_bire import CBlike
from cosmic_bire.anchored import AnchoredBeta
from cosmic_bire.analytic_cov import MKResponse, anchored_beta_cov

DEG = 180.0 / np.pi
FREQS = ("100", "143")


def load_mk_chain(config, mask="0"):
    """MK chain and its restricted MKResponse (Fisher) at the given mask."""
    lh = CBlike(config)
    lh.config["likelihood"]["mask"] = str(mask)
    lh.precompute()
    _ = lh.getdist_samples()
    samples = lh.samples
    n_A = int(lh.config["likelihood"].get("number_of_A", 4))
    nmaps = 2 * len(lh.config["spectra"]["frequencies"])
    p_hat = np.concatenate((
        [samples[:, 0].mean()],
        samples[:, 1:1 + nmaps].mean(0),
        samples[:, 1 + nmaps:1 + nmaps + n_A].mean(0) if n_A else np.array([]),
    ))
    mkr = MKResponse(lh.analysis, p_hat, n_A=n_A)
    return lh, samples, mkr


def anchored_theta_fit(config, mask, anchor_mean, anchor_cov, mkr=None,
                       cross="same", freqs=FREQS):
    """Theta fit at 100/143 for one mask, anchored beta with analytic errors.

    ``mkr`` (an MKResponse) is required when ``cross`` is not None -- it
    supplies the analytic anchor-theta cross-covariance. Pass the mask-0
    MKResponse for both cross="same" (mask 0) and cross="nested" (mask 30):
    the anchor is always the mask-0 posterior.
    """
    ab = AnchoredBeta(config, mask=mask, freqs=freqs)
    theta, theta_cov = ab.fit_theta()
    if cross is None:
        beta, sigma, bm, sm = ab.beta_profile(anchor_mean, anchor_cov, theta, theta_cov)
        C = np.diag(sm ** 2)
    else:
        if mkr is None:
            raise ValueError("mkr (MKResponse) required for cross != None")
        beta, sigma, bm, sm, C = anchored_beta_cov(ab, mkr, anchor_mean, anchor_cov, cross=cross)
    return {
        "ab": ab, "theta": theta, "theta_cov": theta_cov,
        "beta": beta, "sigma": sigma, "beta_per_map": bm, "sigma_per_map": sm,
        "cov": C, "labels": [ab.map_label(m) for m in range(ab.nmaps)],
    }


def run(config="configs/planck_hfi.yml", freqs=FREQS, verbose=True):
    """Full step-3 pipeline: anchored beta on mask 0 and mask 30."""
    lh0, samples0, mkr0 = load_mk_chain(config, mask="0")
    ab_anchor = AnchoredBeta(config, mask="0", freqs=freqs)
    anchor_mean, anchor_cov = ab_anchor.anchor_from_chain(samples0)

    if verbose:
        print(f"anchor (mask 0 MK posterior, {freqs}):")
        for k in range(ab_anchor.nmaps):
            print(f"  {ab_anchor.map_label(k):5s} alpha = {anchor_mean[k]:+8.4f} "
                  f"+- {np.sqrt(anchor_cov[k, k]):6.4f} deg")

    r0 = anchored_theta_fit(config, "0", anchor_mean, anchor_cov, mkr=mkr0,
                            cross="same", freqs=freqs)
    r30 = anchored_theta_fit(config, "30", anchor_mean, anchor_cov, mkr=mkr0,
                             cross="nested", freqs=freqs)

    if verbose:
        for tag, r in (("mask 0 (same-data anchor)", r0),
                       ("mask 30 (nested anchor)", r30)):
            print(f"\n{tag}:")
            for k in range(r["ab"].nmaps):
                print(f"  {r['labels'][k]:5s} theta = {r['theta'][k]:+8.4f} "
                      f"+- {np.sqrt(r['theta_cov'][k, k]):6.4f}   "
                      f"beta_m = {r['beta_per_map'][k]:+8.4f} +- {r['sigma_per_map'][k]:6.4f}")
            print(f"  combined beta = {r['beta']:+.4f} +- {r['sigma']:.4f} deg")
        print(f"\nMK beta (mask 0, full network) = "
              f"{samples0[:, 0].mean():+.4f} +- {samples0[:, 0].std():.4f} deg")
        print(f"|anchored beta(mask0) - anchored beta(mask30)| = "
              f"{abs(r0['beta'] - r30['beta']):.4f} deg  "
              f"(combined sigma {np.sqrt(r0['sigma']**2 + r30['sigma']**2):.4f})")

    return {"anchor_mean": anchor_mean, "anchor_cov": anchor_cov,
            "mask0": r0, "mask30": r30, "mk_chain_beta": samples0[:, 0]}


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/planck_hfi.yml"
    run(cfg)
