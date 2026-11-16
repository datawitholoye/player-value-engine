"""
Scoring API: answers "can we send this player a retention offer?"

The answer is ordered by priority: safer gambling first, then whether an
offer is allowed, then churn risk and value.

Run locally:
    uvicorn pve.api:app --reload
then open http://127.0.0.1:8000/docs to try it in the browser.

Endpoints:
    GET /health                 is the service up, and which snapshot it uses
    GET /players/{player_id}    decision, reasons, churn risk and value for one player
    GET /offers?limit=100       the retention campaign list: eligible players
                                ranked by value at risk
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from pve.scoring import ACTIONS, churn_band, load_models, score_players


class SaferGambling(BaseModel):
    decision: str            # suspended / red / amber / eligible
    action: str
    reasons: list[str]
    harm_score: float


class Offer(BaseModel):
    eligible: bool
    offer_rank: int | None   # place in the retention queue (1 = first); None if not eligible
    value_at_risk: float     # value if retained x churn probability: what an offer could save


class Retention(BaseModel):
    churn_probability: float
    churn_risk: str              # low / medium / high
    predicted_value_6m: float    # expected value, allowing for the chance they leave
    value_if_retained_6m: float  # what they'd be worth if they stay at their recent rate
    value_decile: int            # 10 = most valuable 10% of players so far


class PlayerDecision(BaseModel):
    player_id: int
    snapshot_date: str
    safer_gambling: SaferGambling      # 1. always checked first
    offer: Offer                       # 2. allowed?
    retention: Retention               # 3. churn risk and value


def create_app(features_path: Path, models_dir: Path) -> FastAPI:
    settings, _, _ = load_models(models_dir)
    features = pd.read_parquet(features_path)
    scores = score_players(features, models_dir).set_index("player_id")

    app = FastAPI(title="Player Value & Retention Engine",
                  description="Retention offers with a safer gambling guardrail. Synthetic data.")

    def to_response(pid: int, row: pd.Series) -> PlayerDecision:
        return PlayerDecision(
            player_id=pid,
            snapshot_date=settings["snapshot_date"],
            safer_gambling=SaferGambling(
                decision=row["decision"], action=ACTIONS[row["decision"]],
                reasons=row["reasons"], harm_score=round(float(row["harm_score"]), 4)),
            offer=Offer(
                eligible=row["decision"] == "eligible",
                offer_rank=None if pd.isna(row["offer_rank"]) else int(row["offer_rank"]),
                value_at_risk=round(float(row["value_at_risk"]), 2)),
            retention=Retention(
                churn_probability=round(float(row["churn_probability"]), 4),
                churn_risk=churn_band(row["churn_probability"]),
                predicted_value_6m=round(float(row["predicted_value_6m"]), 2),
                value_if_retained_6m=round(float(row["value_if_retained_6m"]), 2),
                value_decile=int(row["value_decile"])),
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "snapshot_date": settings["snapshot_date"], "players": len(scores)}

    @app.get("/players/{player_id}", response_model=PlayerDecision)
    def player(player_id: int):
        if player_id not in scores.index:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        return to_response(player_id, scores.loc[player_id])

    @app.get("/offers", response_model=list[PlayerDecision])
    def offers(limit: int = Query(100, ge=1, le=5000)):
        queue = scores[scores["decision"] == "eligible"].sort_values("offer_rank").head(limit)
        return [to_response(pid, row) for pid, row in queue.iterrows()]

    return app


# the app uvicorn runs; paths can be changed with environment variables
app = create_app(Path(os.environ.get("PVE_FEATURES", "data/features/player_features.parquet")),
                 Path(os.environ.get("PVE_MODELS", "models")))
