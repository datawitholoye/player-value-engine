import numpy as np

from pve.generate_data import OBS_END, make_bets, make_deposits, make_players


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    players = make_players(n, rng)
    bets = make_bets(players, rng)
    deposits = make_deposits(players, bets, rng)
    return players, bets, deposits


def test_same_seed_gives_same_data():
    _, b1, _ = _data(seed=7)
    _, b2, _ = _data(seed=7)
    assert b1.equals(b2)


def test_bets_are_valid():
    players, bets, _ = _data()
    assert bets["bet_id"].is_unique
    assert (bets["stake"] > 0).all()
    assert (bets["returns"] >= 0).all()
    assert bets["placed_at"].max() <= OBS_END
    assert set(bets["player_id"]).issubset(set(players["player_id"]))


def test_no_bet_before_signup():
    players, bets, _ = _data()
    merged = bets.merge(players[["player_id", "signup_date"]], on="player_id")
    assert (merged["placed_at"] >= merged["signup_date"]).all()


def test_operator_keeps_a_margin():
    _, bets, _ = _data(n=2000)
    margin = 1 - bets["returns"].sum() / bets["stake"].sum()
    assert 0.0 < margin < 0.15


def test_at_risk_players_show_harm_markers():
    players, bets, deposits = _data(n=3000)
    seg = players.set_index("player_id")["_true_segment"]
    late = (bets["placed_at"].dt.hour < 6).groupby(bets["player_id"].map(seg)).mean()
    failed = (deposits["status"] == "failed").groupby(deposits["player_id"].map(seg)).mean()
    assert late["at_risk"] > 2 * late["regular"]
    assert failed["at_risk"] > 2 * failed["regular"]
