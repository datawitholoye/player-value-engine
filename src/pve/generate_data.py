"""
Synthetic betting data generator for the Player Value & Retention Engine.

Creates three tables that look like a sportsbook/casino operator's raw data:
  players.parquet   one row per player (signup info + hidden true segment)
  bets.parquet      one row per bet
  deposits.parquet  one row per deposit attempt (successful or failed)

Players belong to one of four hidden segments. Each segment behaves
differently, so the models we build later have real patterns to find:
  casual      bets rarely, small stakes, churns quickly
  regular     steady weekly betting, medium stakes
  vip         frequent, high stakes, long lifetime
  at_risk     stakes ESCALATE over time, more late-night play, more failed
              deposits (these are the harm markers the guardrail must catch)

The data is SYNTHETIC. Nothing here comes from a real operator.

Usage:
    python -m pve.generate_data --players 20000 --out data/raw
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OBS_START = pd.Timestamp("2024-01-01")
OBS_END = pd.Timestamp("2025-12-31 23:59:59")
OBS_DAYS = (OBS_END - OBS_START).days + 1


@dataclass(frozen=True)
class Segment:
    share: float              # share of players in this segment
    active_days_per_week: float
    bets_per_active_day: float
    stake_median: float       # £
    stake_sigma: float        # lognormal spread
    lifetime_mean_days: float # mean days before churning
    casino_share: float       # share of bets on casino vs sports
    late_night_share: float   # share of bets placed 00:00-05:59
    stake_growth: float       # stake multiplier per 30 days active (1.0 = flat)
    failed_deposit_rate: float


SEGMENTS: dict[str, Segment] = {
    "casual":  Segment(0.55, 0.6, 2.0,  5.0, 0.8,  60, 0.20, 0.03, 1.00, 0.03),
    "regular": Segment(0.30, 2.0, 4.0, 10.0, 0.7, 300, 0.30, 0.05, 1.00, 0.04),
    "vip":     Segment(0.07, 4.5, 9.0, 60.0, 0.9, 500, 0.35, 0.06, 1.00, 0.03),
    "at_risk": Segment(0.08, 4.0, 8.0, 12.0, 0.9, 260, 0.55, 0.22, 1.18, 0.18),
}

CHANNELS = ["organic", "paid_search", "affiliate", "social", "retail_crosssell"]
REGIONS = ["North East", "North West", "Yorkshire", "Midlands", "London",
           "South East", "South West", "Scotland", "Wales"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]


def make_players(n: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s].share for s in names])
    seg = rng.choice(names, size=n, p=shares / shares.sum())

    # signups spread across the first 20 months so we have room to observe churn
    signup_offset = rng.integers(0, OBS_DAYS - 120, size=n)
    lifetime = np.array([rng.exponential(SEGMENTS[s].lifetime_mean_days) for s in seg])
    lifetime = np.maximum(lifetime, 1).astype(int)

    return pd.DataFrame({
        "player_id": np.arange(1, n + 1),
        "signup_date": OBS_START + pd.to_timedelta(signup_offset, unit="D"),
        "age_band": rng.choice(AGE_BANDS, size=n, p=[0.18, 0.30, 0.22, 0.17, 0.13]),
        "region": rng.choice(REGIONS, size=n),
        "acquisition_channel": rng.choice(CHANNELS, size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
        "_true_segment": seg,              # hidden: for validation only, never a model feature
        "_true_lifetime_days": lifetime,   # hidden
        "_signup_offset": signup_offset,
    })


def make_bets(players: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for name, cfg in SEGMENTS.items():
        p = players[players["_true_segment"] == name]
        if p.empty:
            continue
        # how many days each player is "alive" inside the observation window
        alive = np.minimum(p["_true_lifetime_days"].to_numpy(),
                           OBS_DAYS - p["_signup_offset"].to_numpy())
        # per-player activity varies around the segment average
        rate = cfg.active_days_per_week / 7 * rng.lognormal(0, 0.35, len(p))
        n_active = rng.binomial(alive, np.clip(rate, 0, 1))
        n_active = np.maximum(n_active, 1)

        pid = np.repeat(p["player_id"].to_numpy(), n_active)
        start = np.repeat(p["_signup_offset"].to_numpy(), n_active)
        span = np.repeat(alive, n_active)
        day_in_life = (rng.random(len(pid)) * span).astype(int)

        bets_today = rng.poisson(cfg.bets_per_active_day, len(pid)) + 1
        pid = np.repeat(pid, bets_today)
        day_abs = np.repeat(start + day_in_life, bets_today)
        life_day = np.repeat(day_in_life, bets_today)

        # time of day: daytime/evening mostly, late night for some
        late = rng.random(len(pid)) < cfg.late_night_share
        hour = np.where(late, rng.integers(0, 6, len(pid)),
                        rng.choice(np.arange(6, 24), len(pid)))
        secs = rng.integers(0, 3600, len(pid))

        growth = cfg.stake_growth ** (life_day / 30)
        stake = rng.lognormal(np.log(cfg.stake_median), cfg.stake_sigma, len(pid)) * growth
        stake = np.round(np.clip(stake, 0.10, 25_000), 2)

        casino = rng.random(len(pid)) < cfg.casino_share
        # sports odds (decimal) vs casino "odds" (~ house edge baked in)
        odds = np.where(casino, rng.choice([2.0, 3.0, 36.0], len(pid), p=[.6, .3, .1]),
                        np.round(1 + rng.gamma(1.6, 0.9, len(pid)), 2))
        margin = np.where(casino, 0.95, 0.93)   # operator edge
        win = rng.random(len(pid)) < (margin / odds)
        returns = np.round(np.where(win, stake * odds, 0.0), 2)

        frames.append(pd.DataFrame({
            "player_id": pid,
            "placed_at": OBS_START + pd.to_timedelta(day_abs, unit="D")
                         + pd.to_timedelta(hour * 3600 + secs, unit="s"),
            "product": np.where(casino, "casino", "sports"),
            "stake": stake,
            "odds": odds,
            "returns": returns,
        }))
    bets = pd.concat(frames, ignore_index=True)
    bets = bets[bets["placed_at"] <= OBS_END].sort_values("placed_at", kind="stable")
    bets.insert(0, "bet_id", np.arange(1, len(bets) + 1))
    bets["outcome"] = np.where(bets["returns"] > 0, "won", "lost")
    return bets.reset_index(drop=True)


def make_deposits(players: pd.DataFrame, bets: pd.DataFrame,
                  rng: np.random.Generator) -> pd.DataFrame:
    # one deposit attempt per player per betting day, sized to cover that day's stakes
    daily = (bets.assign(day=bets["placed_at"].dt.floor("D"))
                 .groupby(["player_id", "day"], as_index=False)
                 .agg(staked=("stake", "sum"), first_bet=("placed_at", "min")))
    seg = players.set_index("player_id")["_true_segment"]
    fail_rate = daily["player_id"].map(seg).map(lambda s: SEGMENTS[s].failed_deposit_rate)
    failed = rng.random(len(daily)) < fail_rate.to_numpy()

    amount = np.round(np.ceil(daily["staked"] * rng.uniform(0.6, 1.2, len(daily)) / 5) * 5, 2)
    deps = pd.DataFrame({
        "player_id": daily["player_id"],
        "requested_at": daily["first_bet"] - pd.to_timedelta(rng.integers(60, 1800, len(daily)), unit="s"),
        "amount": np.maximum(amount, 5),
        "status": np.where(failed, "failed", "success"),
    })
    deps = deps.sort_values("requested_at", kind="stable").reset_index(drop=True)
    deps.insert(0, "deposit_id", np.arange(1, len(deps) + 1))
    return deps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    players = make_players(args.players, rng)
    bets = make_bets(players, rng)
    deposits = make_deposits(players, bets, rng)

    players.drop(columns=["_signup_offset"]).to_parquet(args.out / "players.parquet", index=False)
    bets.to_parquet(args.out / "bets.parquet", index=False)
    deposits.to_parquet(args.out / "deposits.parquet", index=False)

    ggr = bets["stake"].sum() - bets["returns"].sum()
    print(f"players:  {len(players):>12,}")
    print(f"bets:     {len(bets):>12,}")
    print(f"deposits: {len(deposits):>12,}")
    print(f"turnover: £{bets['stake'].sum():>13,.0f}")
    print(f"GGR:      £{ggr:>13,.0f}  (margin {ggr / bets['stake'].sum():.1%})")
    print(f"saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
