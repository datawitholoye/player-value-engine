"""
Safer gambling guardrail: decides what each player may be offered.

Policy (from retail safer gambling practice):
  suspended  source-of-funds check failed or not provided:
             no betting until documents are provided (hard rule)
  red        strong signs of harm: no offers + customer contact,
             with self-exclusion offered if they have a problem
  amber      signs of harm, OR a history of escalation (lifetime stake
             ESCALATION_HISTORY+ times their first month): no promotional
             offers. A player who escalated and then went quiet may be trying
             to stop, so they must never get a "come back" offer.
  eligible   good betting history and no concerns: may receive retention offers

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
ESCALATION_HISTORY = 2.0   # lifetime avg stake 2x+ first month = never offered

# each retail signal: (feature, threshold, points, plain-English reason)
RULES = [
    ("stake_vs_first30",        3.0,  2, "Recent stakes are {:.1f}x their first month"),
    ("unfamiliar_share_30d",    0.20, 2, "{:.0%} of recent bets are on markets they didn't use at first"),
    ("late_share_vs_first30",   0.10, 1, "Late-night betting is up {:.0f} percentage points on their first month"),
    ("failed_deposits_30d",     3,    1, "{:.0f} declined deposits in the last 30 days"),
    ("chase_rate_30d",          0.30, 1, "Doubled their stake straight after {:.0%} of losses (chasing)"),
    ("latest_stake_jump_ratio", 2.0,  1, "Last week's stakes are {:.1f}x the month before"),
    ("session_vs_first30",      1.5,  1, "Sessions are {:.1f}x longer than in their first month"),
]


def rules_score(df: pd.DataFrame) -> pd.Series:
    """Points for each retail warning sign. Unknown (NULL) values score 0."""
    score = pd.Series(0, index=df.index, dtype=float)
    for col, threshold, points, _ in RULES:
        score += np.where(df[col].fillna(-np.inf) >= threshold, points, 0)
    return score


def rule_reasons(row: pd.Series) -> list[str]:
    """Plain-English list of the warning signs a player shows."""
    reasons = [msg.format(row[col] * (100 if "percentage points" in msg else 1))
               for col, t, _, msg in RULES if pd.notna(row[col]) and row[col] >= t]
    if pd.notna(row.get("stake_lifetime_vs_first30")) and row["stake_lifetime_vs_first30"] >= ESCALATION_HISTORY:
        reasons.append(f"Lifetime average stake is {row['stake_lifetime_vs_first30']:.1f}x their first month "
                       "(past escalation)")
    return reasons


def threshold_for_recall(scores, y, target: float) -> float:
    """Highest threshold that still catches at least `target` of the positives."""
    pos = np.sort(np.asarray(scores)[np.asarray(y, bool)])[::-1]
    if len(pos) == 0:
        return np.inf
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[k])


def escalation_history(df: pd.DataFrame) -> pd.Series:
    """Players whose lifetime average stake is far above their first month."""
    return df["stake_lifetime_vs_first30"].fillna(0) >= ESCALATION_HISTORY


def decide(sof_status: pd.Series, amber: pd.Series, red: pd.Series) -> pd.Series:
    """Apply the policy. The source-of-funds rule overrides everything."""
    return pd.Series(np.select(
        [sof_status.isin(["failed", "not_provided"]).to_numpy(),
         np.asarray(red, bool),
         np.asarray(amber, bool)],
        ["suspended", "red", "amber"], default="eligible"), index=sof_status.index)
