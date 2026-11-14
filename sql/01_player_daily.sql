-- One row per player per day they bet.

CREATE OR REPLACE TABLE player_daily AS
WITH bet_days AS (
    SELECT
        player_id,
        CAST(placed_at AS DATE)                              AS bet_date,
        COUNT(*)                                             AS n_bets,
        SUM(stake)                                           AS total_stake,
        SUM(stake) - SUM(returns)                            AS net_loss,
        MAX(stake)                                           AS max_stake,
        COUNT(DISTINCT market)                               AS n_markets,
        SUM(CASE WHEN HOUR(placed_at) < 6 THEN 1 ELSE 0 END) AS late_night_bets,
        DATE_DIFF('minute', MIN(placed_at), MAX(placed_at))  AS minutes_active
    FROM bets
    GROUP BY player_id, CAST(placed_at AS DATE)
),
deposit_days AS (
    SELECT
        player_id,
        CAST(requested_at AS DATE)                               AS bet_date,
        SUM(CASE WHEN status = 'failed'  THEN 1      ELSE 0 END) AS failed_deposits,
        SUM(CASE WHEN status = 'success' THEN amount ELSE 0 END) AS deposited
    FROM deposits
    GROUP BY player_id, CAST(requested_at AS DATE)
)
SELECT
    b.*,
    COALESCE(d.failed_deposits, 0) AS failed_deposits,
    COALESCE(d.deposited, 0)       AS deposited
FROM bet_days AS b
LEFT JOIN deposit_days AS d
    ON b.player_id = d.player_id
   AND b.bet_date  = d.bet_date;
