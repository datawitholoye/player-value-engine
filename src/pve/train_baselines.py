"""
Baseline models: the simple, explainable benchmarks every later model must beat.

  1. CLV (6-month value):  BG/NBD x spend per day x house margin
                           (Gamma-Gamma for spend if its assumptions hold)
                           vs a naive "the last 90 days will repeat" rule
  2. Churn (no bet in the next 90 days):
                           logistic regression on the SQL features
                           vs BG/NBD's own activity forecast
                           vs a one-line rule: "days since last bet"

Only players active in the 90 days before the snapshot are scored. They are
split 70/30: models learn on the 70% and are judged on the 30% they never saw.

Writes reports/baseline_metrics.json (committed) and
data/predictions/baselines_test.parquet (not committed).

Usage:
    python -m pve.train_baselines
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pve.clv import BetaGeo, GammaGamma

FEATURES = Path("data/features")
RAW = Path("data/raw")
REPORTS = Path("reports")
PREDICTIONS = Path("data/predictions")
HORIZON_DAYS = 184          # value is predicted for 1 Jul - 31 Dec 2025
CHURN_DAYS = 90

CATEGORICAL = ["age_band", "region", "acquisition_channel", "home_market", "sof_status"]
NOT_FEATURES = ["player_id", "snapshot_date"]


def load() -> pd.DataFrame:
    df = duckdb.sql(f"""
        SELECT f.*, l.future_active_days, l.future_stake, l.future_net_loss, l.churned_90d
        FROM '{FEATURES}/player_features.parquet' AS f
        JOIN '{FEATURES}/labels.parquet'          AS l USING (player_id)
        WHERE l.is_current_player
    """).df()
    # hidden truth, used ONLY to measure harm exposure, never as a feature
    truth = duckdb.sql(f"SELECT player_id, _true_segment, _harm_onset_date FROM '{RAW}/players.parquet'").df()
    df = df.merge(truth, on="player_id")
    df["_in_harm"] = df["_harm_onset_date"].notna() & (df["_harm_onset_date"] <= df["snapshot_date"])
    return df


def top_decile_capture(pred, actual) -> float:
    """Share of all actual future value that sits in the top 10% by prediction."""
    order = np.argsort(-np.asarray(pred))
    top = order[: max(1, len(order) // 10)]
    actual = np.asarray(actual)
    return float(actual[top].sum() / actual.sum())


# ---------------------------------------------------------------- CLV
def clv_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    def rfm(d):
        return dict(frequency=d["active_days"] - 1,
                    recency=d["tenure_days"] - d["recency_days"],   # first -> last bet
                    T=d["tenure_days"])                              # first bet -> snapshot

    bg = BetaGeo().fit(**rfm(train))

    # spend per betting day = stake, which is always positive (net loss can be negative
    # when a player gets lucky). The house margin then turns stake into expected value.
    rep = train[train["active_days"] > 1]
    spend = rep["total_stake"] / rep["active_days"]
    gg = GammaGamma().fit(rep["active_days"] - 1, spend)
    margin = train["total_net_loss"].sum() / train["total_stake"].sum()

    # Gamma-Gamma has two assumptions. Check them instead of trusting the model:
    #  1. q > 1, otherwise the "average player" spend is infinite (tail too heavy)
    #  2. how OFTEN someone bets is unrelated to how MUCH they bet
    freq_spend_corr = float(np.corrcoef(rep["active_days"], np.log(spend))[0, 1])
    gg_ok = gg.params_["q"] > 1 and abs(freq_spend_corr) < 0.3

    t = rfm(test)
    out = pd.DataFrame({"player_id": test["player_id"].to_numpy()})
    out["pred_days_6m"] = bg.expected_days(HORIZON_DAYS, t["frequency"], t["recency"], t["T"])
    out["pred_days_90d"] = bg.expected_days(CHURN_DAYS, t["frequency"], t["recency"], t["T"])
    out["p_alive"] = bg.p_alive(t["frequency"], t["recency"], t["T"])
    observed = (test["total_stake"] / test["active_days"]).to_numpy()
    # if Gamma-Gamma's assumptions fail, fall back to each player's own history
    out["pred_stake_per_day"] = gg.expected_spend(t["frequency"], observed) if gg_ok else observed
    out["pred_clv_6m"] = out["pred_days_6m"] * out["pred_stake_per_day"] * margin
    out["naive_clv_6m"] = test["net_loss_90d"].to_numpy() * HORIZON_DAYS / 90

    actual = test["future_net_loss"].to_numpy()
    metrics = {
        "bgnbd_params": {k: round(float(v), 4) for k, v in bg.params_.items()},
        "gamma_gamma_params": {k: round(float(v), 4) for k, v in gg.params_.items()},
        "gamma_gamma_checks": {
            "q_above_1": bool(gg.params_["q"] > 1),
            "frequency_spend_correlation": round(freq_spend_corr, 3),
            "used": bool(gg_ok),
            "fallback": None if gg_ok else "each player's observed average stake per betting day",
        },
        "house_margin": round(float(margin), 4),
        "betting_days": {
            "predicted_total": round(float(out["pred_days_6m"].sum())),
            "actual_total": int(test["future_active_days"].sum()),
            "spearman": round(float(spearmanr(out["pred_days_6m"], test["future_active_days"])[0]), 3),
        },
        "value": {
            "spearman_clv_model": round(float(spearmanr(out["pred_clv_6m"], actual)[0]), 3),
            "spearman_naive": round(float(spearmanr(out["naive_clv_6m"], actual)[0]), 3),
            "top_decile_capture_clv_model": round(top_decile_capture(out["pred_clv_6m"], actual), 3),
            "top_decile_capture_naive": round(top_decile_capture(out["naive_clv_6m"], actual), 3),
        },
    }
    return out, metrics


# ---------------------------------------------------------------- churn
def churn_baseline(train: pd.DataFrame, test: pd.DataFrame, clv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    label_cols = [c for c in train.columns if c.startswith("future_") or c.startswith("_")] + ["churned_90d"]
    numeric = [c for c in train.columns if c not in CATEGORICAL + NOT_FEATURES + label_cols]

    model = Pipeline([
        ("prep", ColumnTransformer([
            # ratios can be NULL (unknown): fill with the median and add a "was missing" flag
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    y_train, y_test = train["churned_90d"].astype(int), test["churned_90d"].astype(int)
    model.fit(train[numeric + CATEGORICAL], y_train)

    out = pd.DataFrame({"player_id": test["player_id"].to_numpy()})
    out["p_churn_logreg"] = model.predict_proba(test[numeric + CATEGORICAL])[:, 1]

    scores = {
        "logistic_regression": out["p_churn_logreg"],
        "bgnbd_activity": -clv["pred_days_90d"].to_numpy(),   # fewer expected days = more likely to churn
        "days_since_last_bet": test["recency_days"].to_numpy(),
    }
    metrics = {
        "churn_rate_test": round(float(y_test.mean()), 3),
        "roc_auc": {k: round(float(roc_auc_score(y_test, s)), 3) for k, s in scores.items()},
        "pr_auc": {k: round(float(average_precision_score(y_test, s)), 3) for k, s in scores.items()},
        "n_features": len(numeric) + len(CATEGORICAL),
    }
    return out, metrics


# ---------------------------------------------------------------- harm exposure
def harm_exposure(test: pd.DataFrame, clv: pd.DataFrame) -> dict:
    """If offers went to the top 10% by predicted value, how many go to players in harm?"""
    d = test[["player_id", "_in_harm"]].merge(clv[["player_id", "pred_clv_6m"]], on="player_id")
    top = d.nlargest(max(1, len(d) // 10), "pred_clv_6m")
    return {
        "share_of_all_current_players_in_harm": round(float(d["_in_harm"].mean()), 3),
        "share_of_top_decile_targets_in_harm": round(float(top["_in_harm"].mean()), 3),
    }


def main() -> None:
    df = load()
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["churned_90d"])
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    clv, clv_metrics = clv_baseline(train, test)
    churn, churn_metrics = churn_baseline(train, test, clv)
    report = {
        "players": {"current": len(df), "train": len(train), "test": len(test)},
        "clv": clv_metrics,
        "churn": churn_metrics,
        "harm_exposure_without_guardrail": harm_exposure(test, clv),
    }

    REPORTS.mkdir(exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    clv.merge(churn, on="player_id").to_parquet(PREDICTIONS / "baselines_test.parquet", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
