"""
Tests for our own BG/NBD and Gamma-Gamma implementation.

Because we wrote these models ourselves rather than using a library, we
prove they work the standard way: simulate players from the model with
KNOWN parameters, fit the model, and check it recovers them and predicts
the future correctly.
"""
import numpy as np
import pytest

from pve.clv import BetaGeo, GammaGamma

TRUE_BG = {"r": 0.8, "alpha": 6.0, "a": 1.4, "b": 4.0}
TRUE_GG = {"p": 4.0, "q": 3.0, "v": 15.0}


def simulate_bgnbd(n, T, future, params, rng):
    """Simulate the BG/NBD story directly: a personal betting rate, and a
    chance of dropping out after every betting day. Returns the summary
    (x, t_x) at time T plus the true number of betting days in (T, T+future]."""
    lam = rng.gamma(params["r"], 1 / params["alpha"], n)
    p = rng.beta(params["a"], params["b"], n)
    x = np.zeros(n); t_x = np.zeros(n); fut = np.zeros(n)
    for i in range(n):
        t, alive = 0.0, True
        while alive:
            t += rng.exponential(1 / lam[i])
            if t > T + future:
                break
            if t <= T:
                x[i] += 1; t_x[i] = t
            else:
                fut[i] += 1
            alive = rng.random() > p[i]
    return x, t_x, np.full(n, float(T)), fut


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(0)
    return simulate_bgnbd(6000, T=180, future=90, params=TRUE_BG, rng=rng)


def test_bgnbd_recovers_known_parameters(sim):
    x, t_x, T, _ = sim
    fit = BetaGeo().fit(x, t_x, T).params_
    for k, true in TRUE_BG.items():
        assert abs(fit[k] - true) / true < 0.25, f"{k}: fitted {fit[k]:.2f}, true {true}"


def test_bgnbd_predicts_future_betting_days(sim):
    # Total predicted future days should be close to what actually happened,
    # and players predicted to bet more should actually bet more.
    x, t_x, T, fut = sim
    pred = BetaGeo().fit(x, t_x, T).expected_days(90, x, t_x, T)
    assert abs(pred.sum() - fut.sum()) / fut.sum() < 0.10
    assert np.corrcoef(pred, fut)[0, 1] > 0.6


def test_p_alive_is_lower_for_players_gone_quiet(sim):
    # Same history, but one was last seen long ago: they're less likely alive.
    x, t_x, T, _ = sim
    m = BetaGeo().fit(x, t_x, T)
    recent, stale = m.p_alive([10, 10], [175, 40], [180, 180])
    assert recent > stale
    assert m.p_alive([0], [0], [180])[0] == 1.0   # never repeated: can't have dropped out yet


def test_gamma_gamma_recovers_known_parameters():
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.integers(1, 30, n).astype(float)
    nu = rng.gamma(TRUE_GG["q"], 1 / TRUE_GG["v"], n)              # each player's spend scale
    m = np.array([rng.gamma(TRUE_GG["p"], 1 / nu[i], int(k)).mean() for i, k in enumerate(x)])
    fit = GammaGamma().fit(x, m).params_
    for k, true in TRUE_GG.items():
        assert abs(fit[k] - true) / true < 0.25, f"{k}: fitted {fit[k]:.2f}, true {true}"


def test_bgnbd_works_when_a_is_below_one():
    # Our real players fit a < 1 (dropout chances bunched near zero: most
    # players are very loyal, a few drop out quickly). Predictions must
    # still match reality in that case.
    rng = np.random.default_rng(3)
    params = {"r": 2.4, "alpha": 10.0, "a": 0.25, "b": 25.0}
    x, t_x, T, fut = simulate_bgnbd(4000, T=180, future=180, params=params, rng=rng)
    m = BetaGeo().fit(x, t_x, T)
    assert m.params_["a"] < 1
    pred = m.expected_days(180, x, t_x, T)
    assert abs(pred.sum() - fut.sum()) / fut.sum() < 0.05
