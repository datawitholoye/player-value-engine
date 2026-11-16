-- What actually happened AFTER the snapshot date: the answers the models
-- try to predict. Kept in its own table so labels can never be used as
-- features by mistake.

CREATE OR REPLACE TABLE labels AS
WITH params AS (
    -- the snapshot date is set by the runner (build_features.py, default 30 June 2025)
    SELECT
        getvariable('snapshot_date') AS snapshot_date,
        90                AS churn_days,     -- no bet for 90 days = churned
        184               AS value_days      -- 1 Jul - 31 Dec 2025
),
future AS (
    SELECT
        d.player_id,
        COUNT(*)           AS future_active_days,
        SUM(d.total_stake) AS future_stake,
        SUM(d.net_loss)    AS future_net_loss,
        MIN(d.bet_date)    AS next_bet_date
    FROM player_daily AS d
    CROSS JOIN params AS s
    WHERE d.bet_date >  s.snapshot_date
      AND d.bet_date <= s.snapshot_date + s.value_days
    GROUP BY d.player_id
)
SELECT
    f.player_id,
    -- only players seen in the 90 days before the snapshot are "current"
    f.active_days_90d > 0                                   AS is_current_player,
    COALESCE(u.future_active_days, 0)                       AS future_active_days,
    COALESCE(u.future_stake, 0)                             AS future_stake,
    COALESCE(u.future_net_loss, 0)                          AS future_net_loss,
    -- churned: no bet within 90 days of the snapshot
    u.next_bet_date IS NULL
        OR u.next_bet_date > s.snapshot_date + s.churn_days AS churned_90d
FROM player_features AS f
CROSS JOIN params AS s
LEFT JOIN future AS u ON f.player_id = u.player_id;
