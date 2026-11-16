"""
The SQL feature pipeline (sql/01-04) rebuilt in PySpark.

DuckDB runs on one machine. Spark splits the same work across many
machines, which is how a feature pipeline would run on a real operator's
data (billions of bets). The logic here mirrors the SQL files step by step,
and tests/test_spark_parity.py proves both engines give the same answers.

Usage (needs Java and Python <= 3.13, e.g. inside the Docker image):
    python -m pve.spark_features
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

RAW = Path("data/raw")
OUT = Path("data/features_spark")
SNAPSHOT = "2025-06-30"


def spark_session() -> SparkSession:
    spark = (SparkSession.builder
             .master("local[*]")                          # all cores on this machine
             .appName("player-value-engine")
             .config("spark.sql.session.timeZone", "UTC")  # no daylight-saving surprises
             .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")   # no progress bars
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def seconds(col: str):
    """Timestamp -> seconds since 1970, keeping fractions of a second."""
    return F.unix_micros(F.col(col).cast("timestamp")) / 1e6


def minute_diff(start_sec, end_sec):
    """Same as DuckDB DATE_DIFF('minute', ...): minute boundaries crossed."""
    return F.floor(end_sec / 60) - F.floor(start_sec / 60)


# ------------------------------------------------------------ 01 player_daily
def player_daily(bets: DataFrame, deposits: DataFrame) -> DataFrame:
    bet_days = (bets
        .withColumn("bet_date", F.to_date("placed_at"))
        .withColumn("sec", seconds("placed_at"))
        .groupBy("player_id", "bet_date")
        .agg(F.count("*").alias("n_bets"),
             F.sum("stake").alias("total_stake"),
             (F.sum("stake") - F.sum("returns")).alias("net_loss"),
             F.max("stake").alias("max_stake"),
             F.countDistinct("market").alias("n_markets"),
             F.sum(F.when(F.hour("placed_at") < 6, 1).otherwise(0)).alias("late_night_bets"),
             minute_diff(F.min("sec"), F.max("sec")).alias("minutes_active")))
    deposit_days = (deposits
        .withColumn("bet_date", F.to_date("requested_at"))
        .groupBy("player_id", "bet_date")
        .agg(F.sum(F.when(F.col("status") == "failed", 1).otherwise(0)).alias("failed_deposits"),
             F.sum(F.when(F.col("status") == "success", F.col("amount")).otherwise(0)).alias("deposited")))
    return (bet_days.join(deposit_days, ["player_id", "bet_date"], "left")
            .fillna(0, subset=["failed_deposits", "deposited"]))


# -------------------------------------------------------- 02 rolling_features
def rolling_features(daily: DataFrame) -> DataFrame:
    day_no = F.datediff("bet_date", F.lit("1970-01-01"))   # RANGE needs a number
    by_player = Window.partitionBy("player_id").orderBy("day_no")
    last_7 = by_player.rangeBetween(-6, 0)
    prior_30 = by_player.rangeBetween(-36, -7)
    d = daily.withColumn("day_no", day_no)
    d = d.select(
        "player_id", "bet_date",
        F.sum("n_bets").over(last_7).alias("bets_7d"),
        F.sum("total_stake").over(last_7).alias("stake_7d"),
        F.sum("minutes_active").over(last_7).alias("minutes_7d"),
        F.sum("failed_deposits").over(last_7).alias("failed_deposits_7d"),
        F.sum("late_night_bets").over(last_7).alias("late_bets_7d"),
        F.count("*").over(last_7).alias("active_days_7d"),
        F.sum("n_bets").over(prior_30).alias("bets_prior30"),
        F.sum("total_stake").over(prior_30).alias("stake_prior30"),
        F.sum("minutes_active").over(prior_30).alias("minutes_prior30"),
        F.sum("late_night_bets").over(prior_30).alias("late_bets_prior30"),
        F.count("*").over(prior_30).alias("active_days_prior30"),
        F.datediff("bet_date", F.lag("bet_date").over(by_player)).alias("days_since_prev_bet"))
    # try_divide = divide, or NULL instead of an error when dividing by zero (like NULLIF)
    return d.select(
        "*",
        F.try_divide(F.col("stake_7d") / F.col("bets_7d"),
                     F.try_divide("stake_prior30", "bets_prior30")).alias("stake_jump_ratio"),
        F.try_divide(F.col("minutes_7d") / F.col("active_days_7d"),
                     F.try_divide("minutes_prior30", "active_days_prior30")).alias("time_jump_ratio"),
        (F.col("late_bets_7d") / F.col("bets_7d")).alias("late_share_7d"),
        F.try_divide("late_bets_prior30", "bets_prior30").alias("late_share_prior30"))


# ------------------------------------------------------------ 03 bet_patterns
def bet_patterns(bets: DataFrame) -> DataFrame:
    by_time = Window.partitionBy("player_id").orderBy("placed_at", "bet_id")
    b = (bets
        .withColumn("bet_date", F.to_date("placed_at"))
        .withColumn("sec", seconds("placed_at"))
        .withColumn("prev_outcome", F.lag("outcome").over(by_time))
        .withColumn("prev_stake", F.lag("stake").over(by_time))
        .withColumn("prev_sec", F.lag("sec").over(by_time))
        .withColumn("first_on_market", F.min("sec").over(Window.partitionBy("player_id", "market")))
        .withColumn("first_ever", F.min("sec").over(Window.partitionBy("player_id"))))
    gap = F.col("sec") - F.col("prev_sec")
    lost = F.col("prev_outcome") == "lost"
    b = (b
        .withColumn("is_chase", F.when(lost & (F.col("stake") >= 2 * F.col("prev_stake")) & (gap <= 1800), 1).otherwise(0))
        .withColumn("after_loss", F.when(lost, 1).otherwise(0))
        .withColumn("unfamiliar_market", F.when(F.col("first_on_market") > F.col("first_ever") + 30 * 86400, 1).otherwise(0))
        .withColumn("new_session", F.when(F.col("prev_sec").isNull() | (gap > 1800), 1).otherwise(0))
        # running total of session starts = a session number per player (gaps and islands)
        .withColumn("session_no", F.sum("new_session").over(by_time.rowsBetween(Window.unboundedPreceding, 0))))

    session_lengths = (b.groupBy("player_id", "session_no")
        .agg(F.min("bet_date").alias("bet_date"),
             minute_diff(F.min("sec"), F.max("sec")).alias("session_minutes")))
    daily_sessions = (session_lengths.groupBy("player_id", "bet_date")
        .agg(F.count("*").alias("n_sessions"),
             F.avg("session_minutes").alias("avg_session_minutes"),
             F.max("session_minutes").alias("longest_session_minutes")))
    per_day = (b.groupBy("player_id", "bet_date")
        .agg(F.sum("is_chase").alias("chase_bets"),
             F.sum("after_loss").alias("bets_after_loss"),
             F.sum("unfamiliar_market").alias("unfamiliar_market_bets")))
    return per_day.join(daily_sessions, ["player_id", "bet_date"], "left")


# --------------------------------------------------------- 04 player_features
def player_features(players, daily, rolling, patterns, sof_checks) -> DataFrame:
    snap = F.to_date(F.lit(SNAPSHOT))
    d = (daily.join(patterns.select("player_id", "bet_date", "chase_bets", "bets_after_loss",
                                    "unfamiliar_market_bets", "avg_session_minutes"),
                    ["player_id", "bet_date"])
         .where(F.col("bet_date") <= snap)
         .withColumn("first_bet_date", F.min("bet_date").over(Window.partitionBy("player_id"))))

    lifetime = d.groupBy("player_id").agg(
        F.min("bet_date").alias("first_bet_date"), F.max("bet_date").alias("last_bet_date"),
        F.count("*").alias("active_days"), F.sum("n_bets").alias("total_bets"),
        F.sum("total_stake").alias("total_stake"), F.sum("net_loss").alias("total_net_loss"))
    first_30 = (d.where(F.col("bet_date") < F.date_add("first_bet_date", 30))
        .groupBy("player_id").agg(
            (F.sum("total_stake") / F.sum("n_bets")).alias("first30_avg_stake"),
            (F.sum("late_night_bets") / F.sum("n_bets")).alias("first30_late_share"),
            F.avg("avg_session_minutes").alias("first30_session_minutes")))
    last_30 = (d.where(F.col("bet_date") > F.date_sub(snap, 30))
        .groupBy("player_id").agg(
            F.count("*").alias("active_days_30d"), F.sum("n_bets").alias("bets_30d"),
            F.sum("total_stake").alias("stake_30d"),
            (F.sum("total_stake") / F.sum("n_bets")).alias("avg_stake_30d"),
            (F.sum("late_night_bets") / F.sum("n_bets")).alias("late_share_30d"),
            F.sum("failed_deposits").alias("failed_deposits_30d"),
            (F.sum("unfamiliar_market_bets") / F.sum("n_bets")).alias("unfamiliar_share_30d"),
            F.try_divide(F.sum("chase_bets"), F.sum("bets_after_loss")).alias("chase_rate_30d"),
            F.avg("avg_session_minutes").alias("session_minutes_30d")))
    last_90 = (d.where(F.col("bet_date") > F.date_sub(snap, 90))
        .groupBy("player_id").agg(
            F.count("*").alias("active_days_90d"), F.sum("total_stake").alias("stake_90d"),
            F.sum("net_loss").alias("net_loss_90d")))
    latest = (rolling.where(F.col("bet_date") <= snap)
        .withColumn("rn", F.row_number().over(Window.partitionBy("player_id").orderBy(F.desc("bet_date"))))
        .where("rn = 1")
        .select("player_id", F.col("stake_jump_ratio").alias("latest_stake_jump_ratio"),
                F.col("failed_deposits_7d").alias("latest_failed_deposits_7d")))
    snap_ts = F.lit(SNAPSHOT).cast("timestamp_ntz")
    sof = sof_checks.select("player_id", F.when(F.col("requested_at") > snap_ts, "none")
                                           .when(F.col("resolved_at") > snap_ts, "pending")
                                           .otherwise(F.col("outcome")).alias("sof_status"))

    f = (players.join(lifetime, "player_id")
         .join(first_30, "player_id", "left").join(last_30, "player_id", "left")
         .join(last_90, "player_id", "left").join(latest, "player_id", "left")
         .join(sof, "player_id", "left"))
    z = lambda c: F.coalesce(F.col(c), F.lit(0)).alias(c)   # noqa: E731  counts: missing = 0
    return f.select(
        "player_id", snap.alias("snapshot_date"),
        "age_band", "region", "acquisition_channel", "home_market",
        F.coalesce("sof_status", F.lit("none")).alias("sof_status"),
        F.datediff(snap, "first_bet_date").alias("tenure_days"),
        F.datediff(snap, "last_bet_date").alias("recency_days"),
        "active_days", "total_bets", "total_stake", "total_net_loss",
        (F.col("total_stake") / F.col("total_bets")).alias("avg_stake_per_bet"),
        z("active_days_90d"), z("stake_90d"), z("net_loss_90d"),
        z("active_days_30d"), z("bets_30d"), z("stake_30d"), z("failed_deposits_30d"),
        "late_share_30d", "unfamiliar_share_30d", "chase_rate_30d", "session_minutes_30d",
        F.try_divide("avg_stake_30d", "first30_avg_stake").alias("stake_vs_first30"),
        (F.col("late_share_30d") - F.col("first30_late_share")).alias("late_share_vs_first30"),
        F.try_divide("session_minutes_30d", "first30_session_minutes").alias("session_vs_first30"),
        F.try_divide(F.col("total_stake") / F.col("total_bets"), F.col("first30_avg_stake"))
         .alias("stake_lifetime_vs_first30"),
        "latest_stake_jump_ratio", "latest_failed_deposits_7d",
        F.ntile(10).over(Window.orderBy(F.round("total_net_loss", 2), "player_id")).alias("value_decile"))


def build(spark: SparkSession, raw: Path = RAW, out: Path = OUT) -> dict[str, DataFrame]:
    read = lambda name: spark.read.parquet(str(raw / f"{name}.parquet"))   # noqa: E731
    players, bets, deposits, sof = (read(n) for n in ["players", "bets", "deposits", "sof_checks"])
    daily = player_daily(bets, deposits).cache()            # reused by later steps
    rolling = rolling_features(daily)
    patterns = bet_patterns(bets).cache()
    tables = {
        "player_daily": daily,
        "rolling_features": rolling,
        "bet_patterns": patterns,
        "player_features": player_features(players, daily, rolling, patterns, sof),
    }
    for name, df in tables.items():
        df.write.mode("overwrite").parquet(str(out / name))
        print(f"{name:<20} {df.count():>10,} rows")
    return tables


if __name__ == "__main__":
    spark = spark_session()
    build(spark)
    spark.stop()
