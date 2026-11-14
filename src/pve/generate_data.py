"""
Synthetic online betting data for the Player Value & Retention Engine.

Writes three tables to data/raw/:
  players.parquet    one row per player: signup details, home market,
                     plus hidden "truth" columns
  bets.parquet       one row per bet: time, market, stake, odds, returns
  deposits.parquet   one row per deposit attempt: amount, success/failed
  sof_checks.parquet one row per source-of-funds check: when it was
                     triggered, why, and the outcome

Player types (hidden, for validation only):
  casual    bets now and then, small stakes, usually churns quickly
  regular   steady betting on their favourite market. Some are
            "upgraders" whose stakes rise slowly over time for normal
            reasons (e.g. a pay rise) with NO other warning signs
  vip       stakes big from their very first bets, long lifetime
  at_risk   starts like a small regular (about £1-5 on one market).
            At a hidden "harm onset" day their behaviour changes:
              - stakes jump and keep escalating vs their own baseline
              - they spread into markets they never used (virtuals,
                casino, sports they don't normally follow)
              - more betting very late at night / very early morning
              - longer sessions and more betting days
              - chasing losses (bigger stake straight after a loss)
              - more declined deposits

Source-of-funds checks are TRIGGERED by behaviour, not assigned:
  - net losses of £10,000+ within any 90 days, or
  - successful deposits of £30,000+ within any 365 days.
VIPs usually pass (they genuinely have the money). At-risk players who
hit the threshold after harm onset often fail or never provide documents.

These behaviours come from what retail safer gambling staff look for.
The key idea: harm is a CHANGE from a player's own normal, not a big
number. A VIP staking £500 is not a red flag; a £2 punter staking £50 is.

All data is SYNTHETIC. Nothing comes from a real operator.

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

# ---------------------------------------------------------------- markets
HOME_MARKETS = ["football", "horse_racing", "greyhounds"]
HOME_WEIGHTS = [0.55, 0.28, 0.17]
OTHER_SPORTS = ["tennis", "rugby", "ice_hockey", "basketball", "darts"]
ALL_MARKETS = HOME_MARKETS + OTHER_SPORTS + ["virtuals", "casino"]

# where a bet goes when it is NOT on the player's home market
EXPLORE_NORMAL = {"football": .30, "horse_racing": .25, "greyhounds": .15,
                  "tennis": .10, "darts": .08, "rugby": .05, "ice_hockey": .03,
                  "basketball": .02, "virtuals": .01, "casino": .01}
EXPLORE_HARM = {"virtuals": .32, "casino": .28, "greyhounds": .08,
                "horse_racing": .06, "football": .04, "ice_hockey": .06,
                "rugby": .05, "basketball": .05, "tennis": .04, "darts": .02}

# operator margin by market: share of stakes paid back on average
PAYBACK = {"football": .93, "tennis": .93, "rugby": .93, "ice_hockey": .93,
           "basketball": .93, "darts": .93, "horse_racing": .86,
           "greyhounds": .86, "virtuals": .88, "casino": .95}


@dataclass(frozen=True)
class Behaviour:
    """How a player behaves in one phase of their life."""
    active_days_per_week: float
    bets_per_session: float
    minutes_between_bets: float
    stake_median: float          # £, at the start of the phase
    home_share: float            # share of bets on their home market
    late_night_share: float      # share of sessions starting 00:00-05:59
    failed_deposit_rate: float
    chase_prob: float            # chance of raising stake after a loss
    explore: dict


@dataclass(frozen=True)
class Segment:
    share: float
    lifetime_mean_days: float
    normal: Behaviour
    harm: Behaviour | None = None


SEGMENTS: dict[str, Segment] = {
    "casual": Segment(0.55, 60,
        Behaviour(0.6, 2, 6, 4.0, 0.85, 0.03, 0.03, 0.03, EXPLORE_NORMAL)),
    "regular": Segment(0.30, 320,
        Behaviour(2.0, 4, 5, 8.0, 0.80, 0.04, 0.04, 0.05, EXPLORE_NORMAL)),
    "vip": Segment(0.07, 500,
        Behaviour(4.0, 8, 4, 80.0, 0.70, 0.06, 0.03, 0.05, EXPLORE_NORMAL)),
    "at_risk": Segment(0.08, 300,
        Behaviour(2.0, 4, 5, 3.0, 0.85, 0.04, 0.04, 0.05, EXPLORE_NORMAL),
        Behaviour(4.0, 9, 3, 6.0, 0.35, 0.28, 0.25, 0.60, EXPLORE_HARM)),
}

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
REGIONS = ["North East", "North West", "Yorkshire", "Midlands", "London",
           "South East", "South West", "Scotland", "Wales"]
CHANNELS = ["organic", "paid_search", "affiliate", "social", "retail_crosssell"]


# ================================================================ players
def make_players(n: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s].share for s in names])
    seg = rng.choice(names, size=n, p=shares / shares.sum())

    signup = rng.integers(0, OBS_DAYS - 120, size=n)
    lifetime = np.array([rng.exponential(SEGMENTS[s].lifetime_mean_days) for s in seg])
    lifetime = np.maximum(lifetime, 1).astype(int)
    alive = np.minimum(lifetime, OBS_DAYS - signup)   # days visible to us

    # at-risk players change behaviour 3 weeks to 5 months after joining
    onset = np.where(seg == "at_risk", rng.integers(21, 150, size=n), -1)
    onset = np.where(onset >= alive, -1, onset)       # churned before onset

    upgrader = (seg == "regular") & (rng.random(n) < 0.12)

    return pd.DataFrame({
        "player_id": np.arange(1, n + 1),
        "signup_date": OBS_START + pd.to_timedelta(signup, unit="D"),
        "age_band": rng.choice(AGE_BANDS, size=n, p=[.18, .30, .22, .17, .13]),
        "region": rng.choice(REGIONS, size=n),
        "acquisition_channel": rng.choice(CHANNELS, size=n, p=[.30, .25, .20, .15, .10]),
        "home_market": rng.choice(HOME_MARKETS, size=n, p=HOME_WEIGHTS),
        # hidden truth: never used as model features
        "_true_segment": seg,
        "_upgrader": upgrader,
        "_harm_onset_date": (OBS_START + pd.to_timedelta(signup + onset, unit="D"))
                            .where(onset >= 0),
        "_signup_day": signup,
        "_alive_days": alive,
        "_onset_day": onset,
    })


# ================================================================== bets
def _sessions(day0, length, rate, rng):
    """Pick active days inside [day0, day0+length) for each player.
    Returns (player_index, day_in_life) for every active day."""
    n = rng.binomial(np.maximum(length, 0), np.clip(rate, 0, 1))
    idx = np.repeat(np.arange(len(length)), n)
    day = np.repeat(day0, n) + (rng.random(len(idx)) * np.repeat(length, n)).astype(int)
    return idx, day


def _phase_bets(p: pd.DataFrame, b: Behaviour, day0, length, harm: bool,
                rng: np.random.Generator) -> pd.DataFrame:
    """All bets for players `p` during one phase of their life."""
    personal = rng.lognormal(0, 0.35, len(p))           # people differ
    idx, life_day = _sessions(day0, length, b.active_days_per_week / 7 * personal, rng)
    if len(idx) == 0:
        return pd.DataFrame()

    # each active day = one session of several bets a few minutes apart
    n_bets = rng.poisson(b.bets_per_session * np.sqrt(personal[idx])) + 1
    sess_start_hour = np.where(rng.random(len(idx)) < b.late_night_share,
                               rng.integers(0, 6, len(idx)),
                               rng.choice(np.arange(8, 24), len(idx)))
    session_id = np.repeat(np.arange(len(idx)), n_bets)
    bi = np.repeat(idx, n_bets)                          # player index per bet
    gaps = rng.exponential(b.minutes_between_bets, len(bi))
    first = np.r_[0, np.cumsum(n_bets)[:-1]]
    gaps[first] = 0
    csum = np.cumsum(gaps)
    minutes = csum - np.repeat(csum[first], n_bets)      # minutes into session

    day_abs = p["_signup_day"].to_numpy()[bi] + np.repeat(life_day, n_bets)
    placed = (OBS_START + pd.to_timedelta(day_abs, unit="D")
              + pd.to_timedelta(np.repeat(sess_start_hour, n_bets) * 60 + minutes, unit="min")
              + pd.to_timedelta(rng.integers(0, 60, len(bi)), unit="s"))

    # market: mostly home market, sometimes exploring elsewhere
    home = p["home_market"].to_numpy()[bi]
    ex_m, ex_p = list(b.explore), np.array(list(b.explore.values()))
    market = np.where(rng.random(len(bi)) < b.home_share, home,
                      rng.choice(ex_m, len(bi), p=ex_p / ex_p.sum()))

    # stake: lognormal around the phase median, scaled per player
    stake = rng.lognormal(np.log(b.stake_median), 0.6, len(bi)) * personal[bi]
    day_in_life = np.repeat(life_day, n_bets)
    if harm:   # keeps escalating after onset, capped at 10x
        since = day_in_life - day0[bi]
        stake *= np.minimum(1.015 ** since, 10)
    upg = p["_upgrader"].to_numpy()[bi]                   # slow, healthy growth
    stake *= np.where(upg, 1 + 0.6 * day_in_life / 365, 1)

    return pd.DataFrame({
        "player_id": p["player_id"].to_numpy()[bi],
        "placed_at": placed,
        "market": market,
        "stake": stake,
        "_chase_prob": b.chase_prob,
        "_session_key": session_id,
        "_phase": "harm" if harm else "normal",
    })


def _odds(market, rng):
    n = len(market)
    sports = np.round(1.2 + rng.gamma(1.6, 0.9, n), 2)
    racing = np.round(1.5 + rng.gamma(1.4, 3.5, n), 2)
    casino = rng.choice([2.0, 3.0, 36.0], n, p=[.6, .3, .1])
    return np.select([np.isin(market, ["horse_racing", "greyhounds", "virtuals"]),
                      market == "casino"], [racing, casino], sports)


def make_bets(players: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for name, cfg in SEGMENTS.items():
        p = players[players["_true_segment"] == name].reset_index(drop=True)
        if p.empty:
            continue
        alive, onset = p["_alive_days"].to_numpy(), p["_onset_day"].to_numpy()
        normal_len = np.where(onset >= 0, onset, alive)
        frames.append(_phase_bets(p, cfg.normal, np.zeros(len(p), int), normal_len, False, rng))
        if cfg.harm is not None:
            has = onset >= 0
            q = p[has].reset_index(drop=True)
            frames.append(_phase_bets(q, cfg.harm, onset[has], alive[has] - onset[has], True, rng))

    bets = pd.concat(frames, ignore_index=True)
    bets = bets[bets["placed_at"] <= OBS_END]
    bets = bets.sort_values(["player_id", "placed_at"], kind="stable").reset_index(drop=True)

    # outcome first (depends only on odds), then chasing: a bigger stake
    # straight after a lost bet, for players prone to chasing
    bets["odds"] = _odds(bets["market"].to_numpy(), rng)
    payback = bets["market"].map(PAYBACK).to_numpy()
    won = rng.random(len(bets)) < payback / bets["odds"].to_numpy()
    prev_lost = np.r_[False, ~won[:-1]] & np.r_[False, bets["player_id"].to_numpy()[1:]
                                                 == bets["player_id"].to_numpy()[:-1]]
    chase = prev_lost & (rng.random(len(bets)) < bets["_chase_prob"].to_numpy())
    stake = bets["stake"].to_numpy() * np.where(chase, rng.uniform(1.5, 3.0, len(bets)), 1)
    bets["stake"] = np.round(np.clip(stake, 0.10, 25_000), 2)
    bets["returns"] = np.round(np.where(won, bets["stake"] * bets["odds"], 0.0), 2)
    bets["outcome"] = np.where(won, "won", "lost")

    bets = bets.sort_values("placed_at", kind="stable").reset_index(drop=True)
    bets.insert(0, "bet_id", np.arange(1, len(bets) + 1))
    return bets


# ============================================================== deposits
def make_deposits(bets: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One deposit attempt before each session, sized to roughly cover it.
    A failed attempt is followed by a retry a few minutes later."""
    sess = (bets.groupby(["player_id", "_phase", "_session_key"], as_index=False)
                .agg(staked=("stake", "sum"), start=("placed_at", "min")))
    fail_rate = np.where(sess["_phase"] == "harm", 0.25, 0.035)
    failed = rng.random(len(sess)) < fail_rate
    amount = np.maximum(np.round(np.ceil(sess["staked"] * rng.uniform(.5, 1.1, len(sess)) / 5) * 5), 5)
    t0 = sess["start"] - pd.to_timedelta(rng.integers(2, 20, len(sess)), unit="min")

    attempt = pd.DataFrame({"player_id": sess["player_id"], "requested_at": t0,
                            "amount": amount, "status": np.where(failed, "failed", "success")})
    retry = attempt[failed].assign(requested_at=lambda d: d["requested_at"] + pd.Timedelta(minutes=1),
                                   status="success")
    deps = pd.concat([attempt, retry]).sort_values("requested_at", kind="stable").reset_index(drop=True)
    deps.insert(0, "deposit_id", np.arange(1, len(deps) + 1))
    return deps


# ==================================================== source-of-funds checks
SOF_LOSS_90D = 10_000        # £ net loss within any 90 days
SOF_DEPOSITS_365D = 30_000   # £ successful deposits within any 365 days

# chance of each outcome once a check is requested
SOF_OUTCOMES = ["verified", "not_provided", "failed"]
SOF_PROBS_NORMAL = [0.90, 0.06, 0.04]   # people who can afford it
SOF_PROBS_HARM = [0.25, 0.45, 0.30]     # at-risk players after harm onset


def _first_crossing(daily: pd.DataFrame, col: str, window: str, limit: float) -> pd.Series:
    """First date each player's rolling `window` total of `col` reaches `limit`."""
    rolled = (daily.set_index("day").groupby("player_id")[col]
                   .rolling(window).sum().reset_index())
    hit = rolled[rolled[col] >= limit]
    return hit.groupby("player_id")["day"].min()


def make_sof_checks(players: pd.DataFrame, bets: pd.DataFrame,
                    deposits: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One source-of-funds check per player who crosses a threshold."""
    loss = (bets.assign(day=bets["placed_at"].dt.floor("D"), loss=bets["stake"] - bets["returns"])
                .groupby(["player_id", "day"], as_index=False)["loss"].sum())
    deps = (deposits[deposits["status"] == "success"]
            .assign(day=lambda d: d["requested_at"].dt.floor("D"))
            .groupby(["player_id", "day"], as_index=False)["amount"].sum())

    by_loss = _first_crossing(loss, "loss", "90D", SOF_LOSS_90D).rename("loss")
    by_deps = _first_crossing(deps, "amount", "365D", SOF_DEPOSITS_365D).rename("deposits")
    trig = pd.concat([by_loss, by_deps], axis=1)
    trig["requested_at"] = trig.min(axis=1) + pd.Timedelta(days=1, hours=9)   # reviewed next morning
    trig["trigger"] = np.where(trig["loss"].notna() & (trig["loss"] <= trig["deposits"].fillna(trig["loss"])),
                               "loss_90d", "deposits_365d")
    trig = trig.reset_index().rename(columns={"index": "player_id"})

    # was the player already in harm when the check was requested?
    onset = players.set_index("player_id")["_harm_onset_date"]
    harmed = (trig["requested_at"] >= trig["player_id"].map(onset)).to_numpy()
    u = rng.random(len(trig))
    normal = np.searchsorted(np.cumsum(SOF_PROBS_NORMAL), u)
    harm = np.searchsorted(np.cumsum(SOF_PROBS_HARM), u)
    outcome = np.array(SOF_OUTCOMES)[np.where(harmed, harm, normal)]

    # verified/failed after documents are reviewed; not_provided after a 30-day deadline
    days = np.where(outcome == "not_provided", 30, rng.integers(2, 15, len(trig)))
    trig["outcome"] = outcome
    trig["resolved_at"] = trig["requested_at"] + pd.to_timedelta(days, unit="D")

    out = trig[["player_id", "requested_at", "trigger", "outcome", "resolved_at"]]
    out = out.sort_values("requested_at").reset_index(drop=True)
    out.insert(0, "check_id", np.arange(1, len(out) + 1))
    return out


# ================================================================== main
def generate(n_players: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    players = make_players(n_players, rng)
    bets = make_bets(players, rng)
    deposits = make_deposits(bets, rng)
    sof_checks = make_sof_checks(players, bets, deposits, rng)
    players = players.drop(columns=["_signup_day", "_alive_days", "_onset_day"])
    bets = bets.drop(columns=["_chase_prob", "_session_key", "_phase"])
    return players, bets, deposits, sof_checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    players, bets, deposits, sof_checks = generate(args.players, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    players.to_parquet(args.out / "players.parquet", index=False)
    bets.to_parquet(args.out / "bets.parquet", index=False)
    deposits.to_parquet(args.out / "deposits.parquet", index=False)
    sof_checks.to_parquet(args.out / "sof_checks.parquet", index=False)

    ggr = bets["stake"].sum() - bets["returns"].sum()
    print(f"players:  {len(players):>12,}")
    print(f"bets:     {len(bets):>12,}")
    print(f"deposits: {len(deposits):>12,}")
    print(f"SoF checks:{len(sof_checks):>11,}")
    print(f"turnover: £{bets['stake'].sum():>13,.0f}")
    print(f"GGR:      £{ggr:>13,.0f}  (margin {ggr / bets['stake'].sum():.1%})")
    print(f"saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
