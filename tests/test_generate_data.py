import pytest

from pve.generate_data import ALL_MARKETS, OBS_END, generate


@pytest.fixture(scope="module")
def data():
    return generate(3000, seed=1)


def test_same_seed_gives_same_data():
    _, b1, _ = generate(300, seed=7)
    _, b2, _ = generate(300, seed=7)
    assert b1.equals(b2)


def test_bets_are_valid(data):
    players, bets, _ = data
    assert bets["bet_id"].is_unique
    assert (bets["stake"] > 0).all()
    assert (bets["returns"] >= 0).all()
    assert bets["placed_at"].max() <= OBS_END
    assert set(bets["market"]).issubset(ALL_MARKETS)
    assert set(bets["player_id"]).issubset(set(players["player_id"]))


def test_no_bet_before_signup(data):
    players, bets, _ = data
    m = bets.merge(players[["player_id", "signup_date"]], on="player_id")
    assert (m["placed_at"] >= m["signup_date"]).all()


def test_operator_keeps_a_margin(data):
    _, bets, _ = data
    margin = 1 - bets["returns"].sum() / bets["stake"].sum()
    assert 0.03 < margin < 0.15


def test_vips_start_big_and_usually_have_source_of_funds(data):
    players, bets, _ = data
    first = bets.sort_values("placed_at").groupby("player_id").head(10)
    first = first.merge(players[["player_id", "_true_segment"]], on="player_id")
    med = first.groupby("_true_segment")["stake"].median()
    assert med["vip"] > 5 * med["at_risk"]
    sof = players.groupby("_true_segment")["source_of_funds_verified"].mean()
    assert sof["vip"] > 0.7 and sof["at_risk"] < 0.1


def test_at_risk_behaviour_changes_after_onset(data):
    players, bets, deposits = data
    ar = players[players["_harm_onset_date"].notna()]
    b = bets.merge(ar[["player_id", "home_market", "_harm_onset_date"]], on="player_id")
    after = b["placed_at"] >= b["_harm_onset_date"]
    before, post = b[~after], b[after]

    assert post["stake"].median() > 5 * before["stake"].median()               # stake jump
    assert (post["market"] == post["home_market"]).mean() < 0.6                # new markets
    assert (post["placed_at"].dt.hour < 6).mean() > 3 * (before["placed_at"].dt.hour < 6).mean()

    d = deposits.merge(ar[["player_id", "_harm_onset_date"]], on="player_id")
    d_after = d["requested_at"] >= d["_harm_onset_date"]
    fail = (d["status"] == "failed")
    assert fail[d_after].mean() > 3 * fail[~d_after].mean()                    # declined cards
