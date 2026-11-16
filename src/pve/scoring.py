"""
Score every player with the saved models and apply the guardrail.

Priority order (from retail safer gambling practice):
  1. harm         source-of-funds and harm checks come first, always
  2. eligibility  only players with no concerns may get retention offers
  3. churn/value  eligible players are ranked by VALUE AT RISK
                  (value if retained x chance of churning), so offers go to
                  valuable players who are likely to leave

Used by the API (src/pve/api.py) and testable on its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pve import guardrail as g
from pve.clv import BetaGeo

ACTIONS = {
    "suspended": "No betting until source-of-funds documents are provided",
    "red": "No offers. Contact the customer about their gambling; offer self-exclusion if needed",
    "amber": "No promotional or retention offers",
    "eligible": "May receive retention offers",
}


def load_models(models_dir: Path):
    settings = json.loads((models_dir / "settings.json").read_text())
    churn = lgb.Booster(model_file=str(models_dir / "churn.txt"))
    harm = lgb.Booster(model_file=str(models_dir / "harm.txt"))
    return settings, churn, harm


def score_players(features: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    """One row per player with every score, the guardrail decision and reasons."""
    settings, churn, harm = load_models(models_dir)
    X = features[settings["feature_columns"]].copy()
    for c in settings["categorical"]:
        X[c] = X[c].astype("category")     # the saved models remember the category levels

    p_churn = churn.predict(X)
    p_harm = harm.predict(X)

    # ---- 1. harm: model flag or past escalation -> amber; + 3 retail signs -> red
    rules = g.rules_score(features)
    model_flag = p_harm >= settings["harm_amber_threshold"]
    amber = model_flag | g.escalation_history(features).to_numpy()
    red = model_flag & (rules >= g.RED_RULE_POINTS).to_numpy()
    decision = g.decide(features["sof_status"], pd.Series(amber, index=features.index),
                        pd.Series(red, index=features.index))

    # ---- 3. value: BG/NBD betting days x their stake per day x house margin
    bg = BetaGeo()
    bg.params_ = settings["bgnbd_params"]
    margin, horizon = settings["house_margin"], settings["horizon_days"]
    days = bg.expected_days(horizon, features["active_days"] - 1,
                            features["tenure_days"] - features["recency_days"], features["tenure_days"])
    value_6m = days * features["total_stake"] / features["active_days"] * margin
    value_if_retained = features["stake_90d"] * margin * horizon / 90

    out = pd.DataFrame({
        "player_id": features["player_id"].astype(int),
        "decision": decision,
        "harm_score": p_harm,
        "retail_warning_points": rules,
        "churn_probability": p_churn,
        "predicted_value_6m": value_6m,
        "value_decile": features["value_decile"].astype(int),
        "value_if_retained_6m": value_if_retained,
        "value_at_risk": value_if_retained * p_churn,
    }, index=features.index)

    # ---- 2. eligibility: only eligible players get a place in the offer queue
    eligible = out["decision"] == "eligible"
    out["offer_rank"] = (out["value_at_risk"].where(eligible)
                         .rank(ascending=False, method="first"))

    # reasons, in plain English
    reasons = []
    for i, row in features.iterrows():
        r = []
        if row["sof_status"] in ("failed", "not_provided"):
            r.append(f"Source-of-funds check {row['sof_status'].replace('_', ' ')}")
        if model_flag[features.index.get_loc(i)]:
            r.append(f"Harm model score {p_harm[features.index.get_loc(i)]:.4f} "
                     f"(flagged from {settings['harm_amber_threshold']:.4f})")
        r += g.rule_reasons(row)
        reasons.append(r)
    out["reasons"] = reasons
    return out


def churn_band(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"
