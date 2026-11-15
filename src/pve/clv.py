"""
BG/NBD and Gamma-Gamma customer lifetime value models, written from the
original papers with SciPy (the popular `lifetimes` library is no longer
maintained and fails on current pandas).

BG/NBD  (Fader, Hardie & Lee, 2005, "Counting your customers the easy way")
    Predicts how many betting days a player will have in the future.
    Each player bets at their own rate (lambda) while "alive", and after
    every betting day has their own chance (p) of dropping out for good.
    Across players, lambda ~ Gamma(r, alpha) and p ~ Beta(a, b).

Gamma-Gamma  (Fader, Hardie & Lee, 2005, "RFM and CLV")
    Predicts a player's typical spend per betting day, pulling players
    with little history towards the average.

Inputs per player (all measured in days, up to the snapshot):
    frequency  x    number of REPEAT betting days (total active days - 1)
    recency    t_x  days from first to last betting day
    T               days from first betting day to the snapshot
    monetary   m_x  average spend per betting day (repeat players only)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1


# ===================================================================== BG/NBD
class BetaGeo:
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer   # small L2 penalty keeps the fit stable
        self.params_: dict | None = None

    @staticmethod
    def _loglik(log_params, x, t_x, T):
        r, alpha, a, b = np.exp(log_params)      # optimise on log scale: always > 0
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        # a4 only exists for players with at least one repeat day
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.where(x > 0,
                          np.log(a) - np.log(np.maximum(b + x - 1, 1e-12))
                          - (r + x) * np.log(alpha + t_x),
                          -np.inf)
        return a1 + a2 + np.logaddexp(a3, a4)

    def fit(self, frequency, recency, T):
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        if np.any(t_x > T) or np.any(x < 0):
            raise ValueError("recency must be <= T and frequency >= 0")

        def objective(lp):
            return -self._loglik(lp, x, t_x, T).mean() + self.penalizer * np.sum(np.exp(lp) ** 2)

        res = minimize(objective, x0=np.log([1.0, 10.0, 1.0, 5.0]), method="Nelder-Mead",
                       options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-10})
        self.params_ = dict(zip(["r", "alpha", "a", "b"], np.exp(res.x)))
        self.converged_ = res.success
        return self

    def p_alive(self, frequency, recency, T):
        """Probability each player is still 'alive' (hasn't dropped out)."""
        r, alpha, a, b = self.params_.values()
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return 1.0 / (1.0 + ratio)

    def expected_days(self, t, frequency, recency, T):
        """Expected number of betting days in the next `t` days."""
        r, alpha, a, b = self.params_.values()
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        z = t / (alpha + T + t)
        top = (a + b + x - 1) / (a - 1) * (
            1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, z))
        return top * self.p_alive(frequency, recency, T)


# ================================================================ Gamma-Gamma
class GammaGamma:
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_: dict | None = None

    @staticmethod
    def _loglik(log_params, x, m):
        p, q, v = np.exp(log_params)
        return (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
                + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(x * m + v))

    def fit(self, frequency, monetary):
        x, m = np.asarray(frequency, float), np.asarray(monetary, float)
        if np.any(x <= 0) or np.any(m <= 0):
            raise ValueError("Gamma-Gamma needs repeat players with positive spend")

        def objective(lp):
            return -self._loglik(lp, x, m).mean() + self.penalizer * np.sum(np.exp(lp) ** 2)

        res = minimize(objective, x0=np.log([1.0, 2.0, np.median(m)]), method="Nelder-Mead",
                       options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-10})
        self.params_ = dict(zip(["p", "q", "v"], np.exp(res.x)))
        self.converged_ = res.success
        return self

    def expected_spend(self, frequency, monetary):
        """Expected average spend per betting day, shrunk towards the average."""
        p, q, v = self.params_.values()
        if q <= 1:
            raise ValueError("Gamma-Gamma prediction needs q > 1")
        x, m = np.asarray(frequency, float), np.asarray(monetary, float)
        population_mean = p * v / (q - 1)
        # players with no repeat days get the population average
        return np.where(x > 0, p * (v + x * np.where(x > 0, m, 0)) / (p * x + q - 1), population_mean)
