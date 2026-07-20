"""Fully analytic covariances for the closure program. No simulations.

Everything derives from the single analytic ingredient already in the
pipeline: the binned Knox covariance M_b of the stacked data vector
c_b = {C_b^{E_iE_j}, C_b^{B_iB_j}, C_b^{E_iB_j}} (the saved cov_bin file).
In the small-angle regime every estimator in the program is a linear
functional of c_b, so all variances and cross-covariances follow from the
delta method:

  MK ML point:  the residual is v_b(p) = A_b(p) c_b - B_b(p) t_b with
  p = (beta, alpha_i[, A_ell]); stationarity of chi2 = sum_b v^T W v gives
      dp = -F^{-1} sum_b J_b^T W_b A_b dc_b,   F = sum_b J_b^T W_b J_b,
  with J_b = dv_b/dp and W_b = [A_b M_b A_b^T]^{-1} at the best fit.

  RelCal network:  alpha_rel = G sum_b 2 D^T diag(X_b) Ninv_b Y_b and
  Y_b = S c_b is a fixed selection of EB entries, so
      dalpha_rel = [G 2 D^T diag(X_b) Ninv_b S] dc_b.

  Anchored theta fit:  same structure as MK restricted to the chosen maps
  with no dust term; its data vector is a sub-selection iota of the full one.

Cross terms are then sums over bins of R M_b R'^T. For the mask-30 theta fit
against the mask-0 anchor, the nested-mask Knox result
Cov(c^A, c^B) = Var(c^A) for region B inside region A (here A = mask-0
region, B = mask-30 region) supplies the analytic cross-data covariance,
consistent with the same approximation scheme as everything else.
"""
from __future__ import annotations

import numpy as np

from . import tools_fast as tf

DEG = 180.0 / np.pi


def _quad_index(nob):
    """Map ordered quadruple (i, j, im, jm) -> row v_i of the stacked vector."""
    out = {}
    v = 0
    for i in range(nob):
        for j in range(nob):
            for im in range(2):
                for jm in range(2):
                    if tf.do_not_count_this_eq(i, j, im, jm):
                        continue
                    out[(i, j, im, jm)] = v
                    v += 1
    return out


class MKResponse:
    """Delta-method linearization of the MK maximum-likelihood point.

    Parameters
    ----------
    analysis : the reference-likelihood object (``lh.analysis`` after
        ``precompute``), exposing c_l_o_bin, c_l_th_bin, obs_cov_bin, psi_l
        and params (nob, l_bin, turn_off_for_lowest_indices).
    p_hat_deg : best-fit/posterior-mean parameter vector
        [beta, alpha_(2 nob), A_(n_A)] with angles in degrees. n_A = 0 runs
        the no-dust model.
    """

    def __init__(self, analysis, p_hat_deg, n_A=None):
        self.an = analysis
        self.nob = int(analysis.params["nob"])
        self.n_bins = int(analysis.number_of_bins)
        self.size = tf.size(self.nob)
        self.nmaps = 2 * self.nob
        self.turn_off = int(analysis.params.get("turn_off_for_lowest_indices", 0))
        self.psi_l = np.asarray(getattr(analysis, "psi_l", np.array([])))
        p = np.asarray(p_hat_deg, dtype=float)
        self.n_A = int(n_A if n_A is not None else max(0, p.size - 1 - self.nmaps))
        self.dust = self.n_A > 0 and self.psi_l.size > 0
        self.p_rad = p.copy()
        self.p_rad[:1 + self.nmaps] /= DEG          # angles deg -> rad, A unchanged
        self.ndim = 1 + self.nmaps + self.n_A
        if np.max(np.abs(self.p_rad[:1 + self.nmaps])) > 0.02:
            import warnings
            warnings.warn(
                "MKResponse: |beta| or |alpha| exceeds ~1.1 deg at the "
                "expansion point. The delta-method linearization and every "
                "small-angle model used throughout this pipeline (Y = 2 "
                "Dalpha X, sin(2x) = 2x, etc.) assume angles at the "
                "sub-degree scale reported for Planck HFI; results at this "
                "point should not be trusted.")
        self._build()

    # ---------------------------------------------------------------- model
    def _AB(self, p, k):
        beta = np.array([p[0]])
        alpha = p[1:1 + self.nmaps]
        if self.dust:
            A = p[1 + self.nmaps:]
            return tf.get_A_B_primed_k(self.nob, alpha, beta, self.psi_l, A,
                                       self.turn_off, k)
        return tf.get_A_B_unprimed(self.nob, alpha, beta)

    def _v(self, p, k):
        A, B = self._AB(p, k)
        return A @ self.an.c_l_o_bin[k] - B @ self.an.c_l_th_bin[k], A

    def _build(self, hstep=1e-5):
        nb, nd = self.n_bins, self.ndim
        F = np.zeros((nd, nd))
        self.R = np.zeros((nb, nd, self.size * 3))   # dp/dc per bin
        JtWA = []
        Ws, Js, As = [], [], []
        for k in range(nb):
            _, A0 = self._v(self.p_rad, k)
            M = self.an.obs_cov_bin[k]
            W = np.linalg.inv(A0 @ M @ A0.T)
            J = np.zeros((self.size, nd))
            for a in range(nd):
                e = np.zeros(nd); e[a] = hstep
                vp, _ = self._v(self.p_rad + e, k)
                vm, _ = self._v(self.p_rad - e, k)
                J[:, a] = (vp - vm) / (2 * hstep)
            F += J.T @ W @ J
            JtWA.append(J.T @ W @ A0)
            Ws.append(W); Js.append(J); As.append(A0)
        self.F = F
        ev = np.linalg.eigvalsh(F)
        self.condition = float(ev.max() / max(ev.min(), 1e-300))
        if ev.min() <= 0 or self.condition > 1e10:
            import warnings
            warnings.warn(f"MK Fisher nearly singular (cond={self.condition:.1e}); "
                          "alpha-beta degeneracy unbroken (no foreground leverage "
                          "in the data or model). Using pseudo-inverse.")
        self.Finv = np.linalg.pinv(F, rcond=1e-12, hermitian=True)
        for k in range(nb):
            self.R[k] = -self.Finv @ JtWA[k]
        self._Ws, self._Js, self._As = Ws, Js, As

    # -------------------------------------------------------------- outputs
    def fisher_cov_deg2(self):
        """Fisher covariance of (beta, alpha, A); angle block in deg^2.
        Cross-check against the chain covariance."""
        C = self.Finv.copy()
        na = 1 + self.nmaps
        C[:na, :na] *= DEG ** 2
        C[:na, na:] *= DEG
        C[na:, :na] *= DEG
        return C

    def alpha_rows(self):
        return slice(1, 1 + self.nmaps)

    def cov_with(self, other_R, rows=None, other_M=None, sub=None):
        """sum_b R_b M_b R'_b^T (radian units for angle rows).

        other_R : (n_bins, ndim', len') response of the other estimator
        rows    : slice of this estimator's parameters (default alpha block)
        other_M : per-bin data covariance of the other estimator's data
                  vector if it differs; sub : column index array embedding the
                  other data vector into this one (Cov = R M[:, sub] R'^T).
        """
        rows = rows if rows is not None else self.alpha_rows()
        out = 0.0
        for k in range(self.n_bins):
            M = self.an.obs_cov_bin[k]
            Mk = M if sub is None else M[:, sub]
            out = out + self.R[k][rows] @ Mk @ other_R[k].T
        return out


def relcal_response(rc, ref=None, rcond=1e-10):
    """Per-bin linear response of the RelCal network fit to the stacked data
    vector, matching ``RelCal.fit_network`` exactly (template X_b held fixed;
    its fluctuation enters at second order). Returns (R[n_bins, nmaps, 3*size],
    gauge constraint used)."""
    P, K, nb = len(rc.pairs), rc.nmaps, rc.n_bins
    nob, size = rc.nob, tf.size(rc.nob)
    quad = _quad_index(nob)
    S = np.zeros((P, size * 3))
    for p_, (m, n) in enumerate(rc.pairs):
        im, i = divmod(m, nob)[0], m % nob
        jm, j = divmod(n, nob)[0], n % nob
        S[p_, 3 * quad[(j, i, jm, im)] + 2] = +1.0   # C^{E_n B_m}
        S[p_, 3 * quad[(i, j, im, jm)] + 2] = -1.0   # C^{E_m B_n}
    D = np.zeros((P, K))
    for p_, (m, n) in enumerate(rc.pairs):
        D[p_, m], D[p_, n] = 1.0, -1.0
    F = np.zeros((K, K))
    blocks = []
    for b in range(nb):
        N = rc.covY_b[:, :, b]
        evals, evecs = np.linalg.eigh(N)
        keep = evals > rcond * evals.max()
        Ninv = (evecs[:, keep] / evals[keep]) @ evecs[:, keep].T
        Td = 2.0 * (D * rc.X_b[:, b][:, None])
        F += Td.T @ Ninv @ Td
        blocks.append(Td.T @ Ninv @ S)
    constraint = np.ones(K) if ref is None else np.eye(K)[ref]
    G = np.linalg.inv(F + np.outer(constraint, constraint) * F.max())
    R = np.zeros((nb, K, size * 3))
    for b in range(nb):
        R[b] = G @ blocks[b]
    return R, constraint


def closure_covariance(rc, mkr, mk_alpha_mean_deg, ref=None):
    """Fully analytic covariance of d = P(alpha_rel - alpha_MK).

    Builds the *combined* per-bin response of d to the shared data vector
    (Rrel - R_MK, both in radians) and contracts once with obs_cov_bin,
    rather than differencing three separately-computed covariance matrices.
    The latter is numerically unstable here: alpha_rel and alpha_MK are
    strongly positively correlated (same spectra), so Cov(d) is a small
    residual of near-equal, separately-noisy terms C_rel + C_mk - 2 C_cross;
    computing it as Var(A - B) directly avoids that cancellation.

    Returns per-map z and the closure chi2 over nmaps-1 degrees of freedom.
    """
    K = rc.nmaps
    if isinstance(ref, str):
        ref = [rc.map_label(k) for k in range(K)].index(ref)
    Rrel, _ = relcal_response(rc, ref=ref)         # (n_bins, K, len), radians
    a_rel, _, labels = rc.fit_network(ref=ref)
    if ref is None:
        Pm = np.eye(K) - 1.0 / K
    else:
        Pm = np.eye(K); Pm[:, ref] -= 1.0
    d = Pm @ a_rel - Pm @ np.asarray(mk_alpha_mean_deg)

    Cd_rad = np.zeros((K, K))
    for b in range(rc.n_bins):
        Rd_b = Pm @ (Rrel[b] - mkr.R[b][1:1 + K])
        Cd_rad += Rd_b @ mkr.an.obs_cov_bin[b] @ Rd_b.T
    Cd = Cd_rad * DEG ** 2

    ev, evec = np.linalg.eigh(Cd)
    keep = ev > 1e-12 * ev.max()
    Cd_pinv = (evec[:, keep] / ev[keep]) @ evec[:, keep].T
    chi2 = float(d @ Cd_pinv @ d)
    z = d / np.sqrt(np.abs(np.diag(Cd)) + 1e-30)
    return {"labels": labels, "d": d, "cov_d": Cd, "z": z,
            "chi2": chi2, "dof": int(keep.sum())}


def theta_response(ab, hstep=1e-5):
    """Delta-method response of the AnchoredBeta theta fit (no dust) and the
    embedding of its data vector into the full stacked vector.
    Returns (R[n_bins, nmaps, 3*size_sub], sub_index[3*size_sub])."""
    nob, size = ab.nob, tf.size(ab.nob)
    nb, K = ab.n_bins, ab.nmaps
    t_hat = (ab.theta if hasattr(ab, "theta") else ab.fit_theta()[0]) / DEG
    F = np.zeros((K, K))
    blocks = []
    for k in range(nb):
        A0, _ = tf.get_A_B_unprimed(nob, t_hat, np.zeros(1))
        M = ab.obs_cov_bin[k]
        W = np.linalg.inv(A0 @ M @ A0.T)
        J = np.zeros((size, K))
        for a in range(K):
            e = np.zeros(K); e[a] = hstep
            Ap, Bp = tf.get_A_B_unprimed(nob, t_hat + e, np.zeros(1))
            Am, Bm = tf.get_A_B_unprimed(nob, t_hat - e, np.zeros(1))
            J[:, a] = ((Ap @ ab.c_l_o_bin[k] - Bp @ ab.c_l_th_bin[k])
                       - (Am @ ab.c_l_o_bin[k] - Bm @ ab.c_l_th_bin[k])) / (2 * hstep)
        F += J.T @ W @ J
        blocks.append(J.T @ W @ A0)
    Finv = np.linalg.inv(F)
    R = np.zeros((nb, K, size * 3))
    for k in range(nb):
        R[k] = -Finv @ blocks[k]
    # embedding: quadruples of the restricted freq set inside the full vector
    quad_sub = _quad_index(nob)
    nob_full = len(ab.config["spectra"]["frequencies"])
    quad_full = _quad_index(nob_full)
    sub = np.zeros(size * 3, dtype=int)
    for (i, j, im, jm), v in quad_sub.items():
        vf = quad_full[(ab.idx[i], ab.idx[j], im, jm)]
        for c in range(3):
            sub[3 * v + c] = 3 * vf + c
    return R, sub


def anchored_beta_cov(ab, mkr, anchor_mean_deg, anchor_cov_deg2,
                      cross="same"):
    """Anchored beta with the analytic theta-anchor cross term.

    cross = "same"   : theta fit and anchor share the identical data
                       (mask-0 theta, mask-0 anchor). The per-map covariance
                       of beta_m = theta_m - alpha_m^MK is built from the
                       *combined* response (Rtheta - Ralpha) contracted once
                       with obs_cov_bin, not from separately differencing
                       Var(theta)+Var(alpha)-2Cov(theta,alpha): the two
                       estimators are strongly correlated on identical data,
                       so the latter is a small residual of noisy large
                       numbers and is numerically unstable.
    cross = "nested" : theta on the smaller (mask-30) region, anchor on the
                       larger (mask-0) region; nested-mask Knox gives
                       Cov(c^0, c^30) = Var(c^0), so the mask-0 M_b is used
                       as the cross matrix. The responses act on different
                       data vectors here (mask-30 vs mask-0 spectra), so the
                       combined-response trick does not apply directly; the
                       three-term formula is used, guarded against
                       pathologically small resulting variances.
    cross = None     : quadrature (conservative, always stable).
    Returns (beta_deg, sigma_deg, per_map_beta, per_map_sigma, C_total).
    """
    K = ab.nmaps
    theta, C_theta = (ab.theta, ab.theta_cov) if hasattr(ab, "theta") else ab.fit_theta()
    Rth, sub = theta_response(ab)
    b = np.asarray(theta) - np.asarray(anchor_mean_deg)

    if cross == "same":
        # theta fit and MK anchor share the identical stacked data vector,
        # i.e. ab and mkr.an must refer to the same spectra/mask. Build the
        # differenced response directly.
        C = np.zeros((K, K))
        for k in range(ab.n_bins):
            Rb_full = np.zeros((K, mkr.an.obs_cov_bin.shape[-1]))
            Rb_full[:, sub] = Rth[k]
            Rd = Rb_full - mkr.R[k][1:1 + K]
            C += Rd @ mkr.an.obs_cov_bin[k] @ Rd.T
        C = C * DEG ** 2
    elif cross == "nested":
        C_cross = np.zeros((K, K))
        chain_freqs = [str(x) for x in ab.config["spectra"]["frequencies"]]
        cols = [chain_freqs.index(ab.freqs[m % ab.nob])
                + len(chain_freqs) * (m // ab.nob) for m in range(K)]
        C_cross = mkr.cov_with(Rth, sub=sub)[cols, :] * DEG ** 2
        C = np.asarray(C_theta) + np.asarray(anchor_cov_deg2) - C_cross - C_cross.T
        floor = 0.05 * np.minimum(np.diag(C_theta), np.diag(anchor_cov_deg2))
        if np.any(np.diag(C) < floor):
            import warnings
            warnings.warn("anchored_beta_cov(nested): per-map variance fell "
                          "below 5% of the smaller input variance; the "
                          "three-term cancellation may be numerically "
                          "unstable here. Falling back to quadrature for "
                          "the affected maps.")
            bad = np.diag(C) < floor
            Cq = np.asarray(C_theta) + np.asarray(anchor_cov_deg2)
            for i in np.where(bad)[0]:
                C[i, :] = 0.0; C[:, i] = 0.0
                C[i, i] = Cq[i, i]
    else:
        C = np.asarray(C_theta) + np.asarray(anchor_cov_deg2)

    Cinv = np.linalg.inv(C)
    one = np.ones(K)
    var = 1.0 / (one @ Cinv @ one)
    beta = var * (one @ Cinv @ b)
    return beta, np.sqrt(var), b, np.sqrt(np.abs(np.diag(C))), C
