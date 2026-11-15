"""
Tests for the safer gambling guardrail policy.

These protect the business rules themselves: if someone changes the code
later, a player in harm must never quietly become eligible for offers.
"""
import numpy as np
import pandas as pd

from pve import guardrail as g


def test_failed_or_missing_source_of_funds_always_suspends():
    # The SoF rule overrides everything, even a "safe" harm score.
    sof = pd.Series(["failed", "not_provided", "verified", "none"])
    no_flags = pd.Series([False] * 4)
    d = g.decide(sof, amber=no_flags, red=no_flags)
    assert list(d) == ["suspended", "suspended", "eligible", "eligible"]


def test_red_outranks_amber_and_amber_outranks_eligible():
    sof = pd.Series(["none"] * 3)
    amber = pd.Series([True, True, False])
    red = pd.Series([True, False, False])
    assert list(g.decide(sof, amber, red)) == ["red", "amber", "eligible"]


def test_rules_score_adds_points_and_ignores_unknowns():
    row = {col: np.nan for col, _, _ in g.RULES}
    unknown = pd.DataFrame([row])
    assert g.rules_score(unknown).iloc[0] == 0          # NULL = unknown, not a warning sign

    row.update(stake_vs_first30=5.0, unfamiliar_share_30d=0.5)  # 2 + 2 points
    assert g.rules_score(pd.DataFrame([row])).iloc[0] == 4


def test_threshold_catches_target_share_of_harmed_players():
    rng = np.random.default_rng(0)
    y = rng.random(1000) < 0.1
    scores = np.where(y, rng.random(1000) + 0.5, rng.random(1000))
    t = g.threshold_for_recall(scores, y, 0.9)
    assert (scores[y] >= t).mean() >= 0.9
