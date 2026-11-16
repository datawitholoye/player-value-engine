"""
Drift monitoring: is the world still like the one the models learned from?

The models were trained on players as of 30 June 2025. Every month after
that, this monitor re-scores current players with those same models and
asks three questions:

  1. Have the players changed?          PSI for each key feature
  2. Have the model's outputs changed?  PSI for the scores + guardrail flag rates
  3. Is the model still right?          churn ROC-AUC, once outcomes are known

PSI (Population Stability Index) compares a feature's distribution this month
with the training month: under 0.1 stable, 0.1-0.25 watch, over 0.25 alert.

It runs on two worlds:
  baseline  the data as generated
  events    a "what-if" copy with two real-world events after training:
            - a new virtuals game launches (October): players try it out
            - a winter football tournament (20 Nov - 18 Dec): extra football
              bets, including from players who normally bet on racing
            Healthy players try new things for a few weeks, then go back to
            their usual market. A good monitor should catch this, and show it
            is drift, not a rise in harm.

Usage:
    python -m pve.monitor
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pve.build_features import SQL, connect
from pve.generate_data import PAYBACK, _odds
from pve import guardrail as g
from pve.scoring import score_players

RAW = Path("data/raw")
EVENTS_RAW = Path("data/raw_events")
MODELS = Path("models")
REPORTS = Path("reports")

TRAINING_SNAPSHOT = "2025-06-30"
MONTH_ENDS = ["2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30", "2025-12-31"]
DATA_END = pd.Timestamp("2025-12-31")
CHURN_DAYS = 90

MONITORED_FEATURES = [
    "stake_30d", "bets_30d", "active_days_30d", "stake_vs_first30", "unfamiliar_share_30d",
    "late_share_vs_first30", "chase_rate_30d", "latest_stake_jump_ratio", "session_vs_first30",
]
MONITORED_SCORES = ["harm_score", "churn_probability"]
PSI_WATCH, PSI_ALERT = 0.10, 0.25

VIRTUALS_LAUNCH = ("2025-10-01", "2025-10-29", 0.35)    # start, end, share of bets moved to virtuals
TOURNAMENT = ("2025-11-20", "2025-12-19", 0.40)         # start, end, extra football bets per bet


# ---------------------------------------------------------------- the PSI measure
def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature.
    Bins are deciles of the reference; missing values get their own bin."""
    ref, cur = pd.Series(reference, dtype=float), pd.Series(current, dtype=float)
    edges = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(x: pd.Series) -> np.ndarray:
        counts = np.histogram(x.dropna(), edges)[0]
        counts = np.append(counts, x.isna().sum())               # the "missing" bin
        return np.clip(counts / max(len(x), 1), 1e-4, None)

    p, q = shares(ref), shares(cur)
    return float(np.sum((q - p) * np.log(q / p)))


def status(value: float) -> str:
    return "alert" if value > PSI_ALERT else "watch" if value > PSI_WATCH else "stable"


# ---------------------------------------------------------------- the event world
def make_event_world(raw: Path = RAW, out: Path = EVENTS_RAW, seed: int = 7) -> None:
    """Copy the data and add two temporary events after the training date."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["players", "deposits", "sof_checks"]:
        shutil.copy(raw / f"{name}.parquet", out / f"{name}.parquet")
    bets = pd.read_parquet(raw / "bets.parquet")

    def settle(df: pd.DataFrame) -> pd.DataFrame:
        """New odds and a new result for bets whose market changed."""
        df["odds"] = _odds(df["market"].to_numpy(), rng)
        won = rng.random(len(df)) < df["market"].map(PAYBACK).to_numpy() / df["odds"].to_numpy()
        df["returns"] = np.round(np.where(won, df["stake"] * df["odds"], 0.0), 2)
        df["outcome"] = np.where(won, "won", "lost")
        return df

    # 1. new virtuals game: some bets in the launch window go to virtuals instead
    start, end, share = VIRTUALS_LAUNCH
    in_window = bets["placed_at"].between(start, end, inclusive="left")
    switch = in_window & (rng.random(len(bets)) < share)
    bets.loc[switch, "market"] = "virtuals"
    bets.loc[switch] = settle(bets.loc[switch].copy())

    # 2. tournament: extra football bets on top of normal betting
    start, end, rate = TOURNAMENT
    pool = bets[bets["placed_at"].between(start, end, inclusive="left")]
    extra = pool.sample(frac=rate, random_state=seed).copy()
    extra["market"] = "football"
    extra["placed_at"] = extra["placed_at"] + pd.to_timedelta(rng.uniform(5, 90, len(extra)), unit="min")
    extra = settle(extra[extra["placed_at"] <= DATA_END + pd.Timedelta(hours=23, minutes=59)])
    extra["bet_id"] = np.arange(bets["bet_id"].max() + 1, bets["bet_id"].max() + 1 + len(extra))

    bets = pd.concat([bets, extra]).sort_values(["placed_at", "bet_id"]).reset_index(drop=True)
    bets.to_parquet(out / "bets.parquet", index=False, coerce_timestamps="us", allow_truncated_timestamps=True)


# ---------------------------------------------------------------- monthly snapshots
def monthly_snapshots(raw: Path) -> dict[str, pd.DataFrame]:
    """Player features at the training date and at each month end."""
    con = connect(raw)
    for path in sorted(SQL.glob("0[1-3]_*.sql")):       # daily, rolling and bet patterns: once
        con.execute(path.read_text())
    player_features_sql = (SQL / "04_player_features.sql").read_text()

    snaps = {}
    for date in [TRAINING_SNAPSHOT] + MONTH_ENDS:
        con.execute(f"SET VARIABLE snapshot_date = DATE '{date}'")
        con.execute(player_features_sql)
        f = con.execute("SELECT * FROM player_features ORDER BY player_id").df()
        f = f[f["active_days_90d"] > 0].reset_index(drop=True)      # current players only

        # did they churn? only knowable once 90 days have passed
        if pd.Timestamp(date) + pd.Timedelta(days=CHURN_DAYS) <= DATA_END:
            came_back = con.execute(f"""
                SELECT DISTINCT player_id FROM player_daily
                WHERE bet_date > DATE '{date}' AND bet_date <= DATE '{date}' + {CHURN_DAYS}
            """).df()["player_id"]
            f["churned_90d"] = ~f["player_id"].isin(came_back)
        snaps[date] = f
    return snaps


def monitor(raw: Path, models: Path = MODELS) -> list[dict]:
    snaps = monthly_snapshots(raw)
    truth = pd.read_parquet(raw / "players.parquet")[["player_id", "_harm_onset_date"]]

    scored = {}
    for date, f in snaps.items():
        s = score_players(f, models)
        s = s.merge(f, on="player_id", suffixes=("", "_f"))
        s = s.merge(truth, on="player_id")
        s["in_harm"] = s["_harm_onset_date"].notna() & (s["_harm_onset_date"] <= pd.Timestamp(date))
        scored[date] = s
    ref = scored[TRAINING_SNAPSHOT]

    rows = []
    for date, s in scored.items():
        flagged = s["decision"] != "eligible"
        row = {
            "snapshot": date,
            "current_players": len(s),
            "psi": {c: round(psi(ref[c], s[c]), 3) for c in MONITORED_FEATURES + MONITORED_SCORES},
            "flagged_share": round(float(flagged.mean()), 4),
            "decisions": {k: round(float(v), 4) for k, v in s["decision"].value_counts(normalize=True).items()},
            "churn_roc_auc": (round(float(roc_auc_score(s["churned_90d"], s["churn_probability"])), 3)
                              if "churned_90d" in s and s["churned_90d"].notna().all() else None),
            # hindsight only: the monitor can't see who is truly in harm
            "hindsight": {
                "share_in_harm": round(float(s["in_harm"].mean()), 4),
                "healthy_players_flagged": round(float(flagged[~s["in_harm"]].mean()), 4),
                "harmed_players_flagged": round(float(flagged[s["in_harm"]].mean()), 4),
                # what the transparent retail rules alone would have done
                "rules_only_healthy_flagged": round(float(
                    (g.rules_score(s) >= g.RULES_AMBER)[~s["in_harm"]].mean()), 4),
            },
        }
        worst = max(row["psi"].values())
        flag_change = row["flagged_share"] / max(rows[0]["flagged_share"], 1e-9) - 1 if rows else 0.0
        row["status"] = "alert" if worst > PSI_ALERT or abs(flag_change) > 0.5 else status(worst)
        row["drifted_features"] = [c for c, v in row["psi"].items() if v > PSI_WATCH]
        rows.append(row)
    return rows


def main() -> None:
    make_event_world()
    report = {"baseline": monitor(RAW), "events": monitor(EVENTS_RAW),
              "thresholds": {"psi_watch": PSI_WATCH, "psi_alert": PSI_ALERT}}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "monitoring.json").write_text(json.dumps(report, indent=2))
    for world in ["baseline", "events"]:
        print(f"\n{world}")
        for r in report[world]:
            print(f"  {r['snapshot']}  {r['status']:<6}  flagged {r['flagged_share']:.1%}  "
                  f"healthy flagged {r['hindsight']['healthy_players_flagged']:.1%}  "
                  f"churn AUC {r['churn_roc_auc']}  drifted: {', '.join(r['drifted_features']) or '-'}")


if __name__ == "__main__":
    main()
