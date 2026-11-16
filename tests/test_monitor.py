"""
Tests for drift monitoring: the PSI measure and the "event world".
"""
import numpy as np
import pandas as pd
import pytest

from pve import monitor
from pve.generate_data import generate, save


def test_psi_is_zero_for_the_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert monitor.psi(x, x) < 0.01


def test_psi_alerts_on_a_big_shift():
    rng = np.random.default_rng(1)
    ref, cur = rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)
    assert monitor.psi(ref, cur) > monitor.PSI_ALERT
    assert monitor.status(monitor.psi(ref, cur)) == "alert"


def test_psi_notices_more_missing_values():
    # e.g. far more players with no recent bets (their ratios become NULL)
    ref = pd.Series(np.r_[np.random.default_rng(2).normal(size=900), [np.nan] * 100])
    cur = pd.Series(np.r_[np.random.default_rng(3).normal(size=500), [np.nan] * 500])
    assert monitor.psi(ref, cur) > monitor.PSI_ALERT


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp("worlds")
    save(root / "raw", *generate(1500, seed=3))
    monitor.make_event_world(root / "raw", root / "events")
    return pd.read_parquet(root / "raw/bets.parquet"), pd.read_parquet(root / "events/bets.parquet")


def test_virtuals_launch_only_changes_the_launch_window(worlds):
    base, events = worlds
    start, end, _ = monitor.VIRTUALS_LAUNCH

    def virtuals_share(b, inside):
        w = b["placed_at"].between(start, end, inclusive="left")
        return (b.loc[w if inside else ~w, "market"] == "virtuals").mean()

    assert virtuals_share(events, True) > virtuals_share(base, True) + 0.2      # big jump in October

    # nothing before the launch changes: same bets, same markets
    m = base.merge(events, on="bet_id", suffixes=("_base", "_events"))
    early = m["placed_at_base"] < pd.Timestamp(start)
    assert (m.loc[early, "market_base"] == m.loc[early, "market_events"]).all()


def test_tournament_adds_football_bets(worlds):
    base, events = worlds
    start, end, _ = monitor.TOURNAMENT
    w = lambda b: b[b["placed_at"].between(start, end, inclusive="left")]   # noqa: E731
    assert len(w(events)) > 1.2 * len(w(base))
    assert (w(events)["market"] == "football").mean() > (w(base)["market"] == "football").mean()
    assert events["bet_id"].is_unique
