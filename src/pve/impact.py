"""
Commercial impact of a retention campaign: what churn costs, what offers
save, what they cost, and what the safer gambling guardrail gives up.

The assumptions are deliberately simple and stated up front, because a
real operator would replace them with results from a controlled test:

  SAVE_RATE   share of would-be churners an offer keeps (a guess: 20%)
  OFFER_COST  cost per offer, e.g. a £10 free bet

An offer only creates value if it reaches someone who would otherwise
leave. An offer to a player who'd stay anyway is pure cost.

Value is measured as each player's recent run rate: stake in the last 90
days x house margin, scaled to the 6-month horizon ("value if retained").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SAVE_RATE = 0.20
OFFER_COST = 10.0
OFFER_SHARE = 0.10        # budget: offers for 10% of current players


def pick(df: pd.DataFrame, rank_by: str, n: int, guardrail: bool) -> pd.DataFrame:
    """The n players a strategy would send offers to."""
    pool = df[df["decision"] == "eligible"] if guardrail else df
    return pool.nlargest(n, rank_by)


def campaign(targets: pd.DataFrame, save_rate: float = SAVE_RATE,
             offer_cost: float = OFFER_COST) -> dict:
    """Money in and out for one campaign. Needs columns:
    churned (bool), in_harm (bool), value_if_retained_6m (£ over the next 6 months)."""
    churners = targets[targets["churned"]]
    at_stake = churners["value_if_retained_6m"].sum()
    saved = save_rate * at_stake
    cost = offer_cost * len(targets)
    return {
        "offers": len(targets),
        "offers_to_players_who_would_leave": int(len(churners)),
        "offers_wasted_on_players_who_stay": int((~targets["churned"]).sum()),
        "offers_to_harmed_players": int(targets["in_harm"].sum()),
        "value_at_stake": float(at_stake),
        "value_saved": float(saved),
        "offer_cost": float(cost),
        "net_impact": float(saved - cost),
        "value_saved_from_harmed_players": float(save_rate * churners.loc[churners["in_harm"], "value_if_retained_6m"].sum()),
    }


def compare_strategies(df: pd.DataFrame, n: int, **assumptions) -> pd.DataFrame:
    """The three strategies side by side."""
    strategies = {
        "Value, no guardrail": pick(df, "predicted_value_6m", n, guardrail=False),
        "Value + guardrail": pick(df, "predicted_value_6m", n, guardrail=True),
        "Value at risk + guardrail": pick(df, "value_at_risk", n, guardrail=True),
    }
    return pd.DataFrame({k: campaign(t, **assumptions) for k, t in strategies.items()}).T


def sensitivity(df: pd.DataFrame, n: int, rank_by: str = "value_at_risk",
                save_rates=(0.10, 0.20, 0.30), offer_costs=(5.0, 10.0, 25.0)) -> pd.DataFrame:
    """Net impact (£) of the guarded value-at-risk campaign under different assumptions."""
    targets = pick(df, rank_by, n, guardrail=True)
    return pd.DataFrame(
        [[campaign(targets, s, c)["net_impact"] for c in offer_costs] for s in save_rates],
        index=[f"saves {s:.0%} of leavers" for s in save_rates],
        columns=[f"£{c:.0f} per offer" for c in offer_costs])


def churn_cost(df: pd.DataFrame) -> dict:
    """How much value walks out of the door with churners, with no campaign."""
    churners = df[df["churned"]]
    return {
        "players": len(df),
        "churners": int(len(churners)),
        "churn_rate": float(df["churned"].mean()),
        "value_lost_to_churn": float(churners["value_if_retained_6m"].sum()),
        "value_of_all_players": float(df["value_if_retained_6m"].sum()),
    }
