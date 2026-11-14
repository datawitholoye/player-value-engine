-- For every player-day: their last 7 days compared with their previous 30 days.

CREATE OR REPLACE TABLE rolling_features AS
WITH windows AS (
    SELECT
        player_id,
        bet_date,

        -- last 7 days (today + 6 days before)
        SUM(n_bets)          OVER last_7  AS bets_7d,
        SUM(total_stake)     OVER last_7  AS stake_7d,
        SUM(minutes_active)  OVER last_7  AS minutes_7d,
        SUM(failed_deposits) OVER last_7  AS failed_deposits_7d,
        SUM(late_night_bets) OVER last_7  AS late_bets_7d,
        COUNT(*)             OVER last_7  AS active_days_7d,

        -- the 30 days before that: the player's own "normal"
        SUM(n_bets)          OVER prior_30 AS bets_prior30,
        SUM(total_stake)     OVER prior_30 AS stake_prior30,
        SUM(minutes_active)  OVER prior_30 AS minutes_prior30,
        SUM(late_night_bets) OVER prior_30 AS late_bets_prior30,
        COUNT(*)             OVER prior_30 AS active_days_prior30,

        -- gap since the previous betting day
        bet_date - LAG(bet_date) OVER by_date AS days_since_prev_bet

    FROM player_daily
    WINDOW
        by_date  AS (PARTITION BY player_id ORDER BY bet_date),
        last_7   AS (PARTITION BY player_id ORDER BY bet_date
                     RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW),
        prior_30 AS (PARTITION BY player_id ORDER BY bet_date
                     RANGE BETWEEN INTERVAL 36 DAYS PRECEDING AND INTERVAL 7 DAYS PRECEDING)
)
SELECT
    *,
    -- average stake per bet: recent vs normal (1.0 = no change)
    (stake_7d / bets_7d) / NULLIF(stake_prior30 / bets_prior30, 0)        AS stake_jump_ratio,
    -- time spent per active day: recent vs normal
    (minutes_7d / active_days_7d)
        / NULLIF(minutes_prior30 / active_days_prior30, 0)                AS time_jump_ratio,
    -- share of bets placed midnight-6am: recent vs normal
    late_bets_7d / bets_7d                                                AS late_share_7d,
    late_bets_prior30 / NULLIF(bets_prior30, 0)                           AS late_share_prior30
FROM windows;
