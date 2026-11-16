"""Tests for the campaign maths: an offer only creates value when it
reaches someone who would otherwise leave."""
import pandas as pd

from pve import impact


def players():
    return pd.DataFrame({
        "decision": ["eligible", "eligible", "eligible", "amber"],
        "churned": [True, False, True, True],
        "in_harm": [False, False, False, True],
        "value_if_retained_6m": [1000.0, 5000.0, 200.0, 3000.0],
        "predicted_value_6m": [100.0, 900.0, 50.0, 800.0],
        "value_at_risk": [800.0, 50.0, 150.0, 2500.0],
    })


def test_campaign_money():
    c = impact.campaign(players().iloc[[0, 1]], save_rate=0.2, offer_cost=10)
    assert c["value_at_stake"] == 1000           # only the churner counts
    assert c["value_saved"] == 200
    assert c["offer_cost"] == 20
    assert c["net_impact"] == 180
    assert c["offers_wasted_on_players_who_stay"] == 1


def test_guardrail_never_targets_flagged_players():
    t = impact.pick(players(), "value_at_risk", n=2, guardrail=True)
    assert (t["decision"] == "eligible").all()
    assert list(t.index) == [0, 2]                # ranked by value at risk


def test_without_guardrail_harmed_player_is_picked_first():
    t = impact.pick(players(), "value_at_risk", n=1, guardrail=False)
    assert bool(t["in_harm"].iloc[0])
