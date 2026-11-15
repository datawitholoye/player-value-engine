"""
Safer gambling guardrail: decides what each player may be offered.

Policy (from retail safer gambling practice):
  suspended  source-of-funds check failed or not provided:
             no betting until documents are provided (hard rule)
  red        strong signs of harm: no offers + customer contact,
             with self-exclusion offered if they have a problem
  amber      signs of harm: no promotional offers
  eligible   may receive retention offers

Two guardrails are compared:
  rules only   a transparent points score built from the retail signals:
               amber at RULES_AMBER points, red at RULES_RED points
  model+rules  a LightGBM model flags players (amber). The amber threshold
               is chosen to catch at least TARGET_RECALL of harmed players,
               because missing a harmed player is worse than wrongly
               withholding an offer from a healthy VIP. A flagged player
               who also shows RED_RULE_POINTS+ retail warning signs goes
               to red, so every customer contact has concrete reasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_RECALL = 0.90       # model amber must catch at least 90% of harmed players
RULES_AMBER = 2            # rules-only guardrail: points needed for amber
RULES_RED = 4              # ... and for red
RED_RULE_POINTS = 3        # model-flagged + this many points = red (customer contact)

# each retail signal: (feature, threshold, points)
RULES = [
    ("stake_vs_first30",        3.0,  2),   # staking 3x+ their first month
    ("unfamiliar_share_30d",    0.20, 2),   # 20%+ of bets on markets they didn't use at first
    ("late_share_vs_first30",   0.10, 1),   # 10+ points more late-night betting than at first
    ("failed_deposits_30d",     3,    1),   # 3+ declined deposits in 30 days
    ("chase_rate_30d",          0.30, 1),   # doubles stake after 30%+ of losses
    ("latest_stake_jump_ratio", 2.0,  1),   # last week's stakes 2x+ the month before
    ("session_vs_first30",      1.5,  1),   # sessions 50%+ longer than at first
]


def rules_score(df: pd.DataFrame) -> pd.Series:
    """Points for each retail warning sign. Unknown (NULL) values score 0."""
    score = pd.Series(0, index=df.index, dtype=float)
    for col, threshold, points in RULES:
        score += np.where(df[col].fillna(-np.inf) >= threshold, points, 0)
    return score


def rule_reasons(row: pd.Series) -> list[str]:
    """Plain-English list of which warning signs a player shows."""
    return [f"{col} = {row[col]:.2f} (threshold {t})"
            for col, t, _ in RULES if pd.notna(row[col]) and row[col] >= t]


def threshold_for_recall(scores, y, target: float) -> float:
    """Highest threshold that still catches at least `target` of the positives."""
    pos = np.sort(np.asarray(scores)[np.asarray(y, bool)])[::-1]
    if len(pos) == 0:
        return np.inf
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[k])


def decide(sof_status: pd.Series, amber: pd.Series, red: pd.Series) -> pd.Series:
    """Apply the policy. The source-of-funds rule overrides everything."""
    return pd.Series(np.select(
        [sof_status.isin(["failed", "not_provided"]).to_numpy(),
         np.asarray(red, bool),
         np.asarray(amber, bool)],
        ["suspended", "red", "amber"], default="eligible"), index=sof_status.index)
