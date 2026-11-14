-- One row per player, as of a snapshot date.
-- Features use ONLY data up to the snapshot. What happens afterwards
-- (churn, future value) becomes the label later, so nothing from the
-- future can leak into the features.

CREATE OR REPLACE TABLE player_features AS
WITH params AS (
    SELECT DATE '2025-06-30' AS snapshot_date
),
daily AS (
    -- daily summary + bet patterns, cut off at the snapshot
    SELECT
        d.*,
        b.chase_bets,
        b.bets_after_loss,
        b.unfamiliar_market_bets,
        b.avg_session_minutes,
        MIN(d.bet_date) OVER (PARTITION BY d.player_id) AS first_bet_date
    FROM player_daily AS d
    JOIN bet_patterns AS b
        ON d.player_id = b.player_id
       AND d.bet_date  = b.bet_date
    WHERE d.bet_date <= (SELECT snapshot_date FROM params)
),
lifetime AS (
    -- recency, frequency, monetary: the classic inputs for value models
    SELECT
        player_id,
        MIN(bet_date)   AS first_bet_date,
        MAX(bet_date)   AS last_bet_date,
        COUNT(*)        AS active_days,
        SUM(n_bets)     AS total_bets,
        SUM(total_stake) AS total_stake,
        SUM(net_loss)   AS total_net_loss
    FROM daily
    GROUP BY player_id
),
first_30 AS (
    -- the player's first month: their true starting "normal"
    SELECT
        player_id,
        SUM(total_stake) / SUM(n_bets)     AS first30_avg_stake,
        SUM(late_night_bets) / SUM(n_bets) AS first30_late_share,
        AVG(avg_session_minutes)           AS first30_session_minutes
    FROM daily
    WHERE bet_date < first_bet_date + INTERVAL 30 DAY
    GROUP BY player_id
),
last_30 AS (
    -- the most recent month before the snapshot
    SELECT
        player_id,
        COUNT(*)                                              AS active_days_30d,
        SUM(n_bets)                                           AS bets_30d,
        SUM(total_stake)                                      AS stake_30d,
        SUM(total_stake) / SUM(n_bets)                        AS avg_stake_30d,
        SUM(late_night_bets) / SUM(n_bets)                    AS late_share_30d,
        SUM(failed_deposits)                                  AS failed_deposits_30d,
        SUM(unfamiliar_market_bets) / SUM(n_bets)            AS unfamiliar_share_30d,
        SUM(chase_bets) / NULLIF(SUM(bets_after_loss), 0)     AS chase_rate_30d,
        AVG(avg_session_minutes)                              AS session_minutes_30d
    FROM daily
    WHERE bet_date > (SELECT snapshot_date FROM params) - INTERVAL 30 DAY
    GROUP BY player_id
),
last_90 AS (
    SELECT
        player_id,
        COUNT(*)         AS active_days_90d,
        SUM(total_stake) AS stake_90d,
        SUM(net_loss)    AS net_loss_90d
    FROM daily
    WHERE bet_date > (SELECT snapshot_date FROM params) - INTERVAL 90 DAY
    GROUP BY player_id
),
latest_rolling AS (
    -- each player's most recent 7-day vs prior-30-day comparison
    SELECT
        player_id,
        stake_jump_ratio   AS latest_stake_jump_ratio,
        failed_deposits_7d AS latest_failed_deposits_7d
    FROM rolling_features
    WHERE bet_date <= (SELECT snapshot_date FROM params)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY bet_date DESC) = 1
),
sof AS (
    -- source-of-funds status AS KNOWN ON the snapshot date:
    -- a check requested later doesn't exist yet, and one still being
    -- reviewed is "pending" even if it is later verified
    SELECT
        c.player_id,
        CASE
            WHEN c.requested_at > s.snapshot_date THEN 'none'
            WHEN c.resolved_at  > s.snapshot_date THEN 'pending'
            ELSE c.outcome
        END AS sof_status
    FROM sof_checks AS c
    CROSS JOIN params AS s
)
SELECT
    p.player_id,
    s.snapshot_date,

    -- who they are (hidden "_" columns are deliberately left out)
    p.age_band,
    p.region,
    p.acquisition_channel,
    p.home_market,
    COALESCE(c.sof_status, 'none')      AS sof_status,

    -- value: recency, frequency, monetary
    s.snapshot_date - l.first_bet_date              AS tenure_days,
    s.snapshot_date - l.last_bet_date               AS recency_days,
    l.active_days,
    l.total_bets,
    l.total_stake,
    l.total_net_loss,
    l.total_stake / l.total_bets                    AS avg_stake_per_bet,
    COALESCE(n.active_days_90d, 0)                  AS active_days_90d,
    COALESCE(n.stake_90d, 0)                        AS stake_90d,
    COALESCE(n.net_loss_90d, 0)                     AS net_loss_90d,

    -- last 30 days (counts become 0 if inactive; ratios stay NULL = unknown)
    COALESCE(m.active_days_30d, 0)                  AS active_days_30d,
    COALESCE(m.bets_30d, 0)                         AS bets_30d,
    COALESCE(m.stake_30d, 0)                        AS stake_30d,
    COALESCE(m.failed_deposits_30d, 0)              AS failed_deposits_30d,
    m.late_share_30d,
    m.unfamiliar_share_30d,
    m.chase_rate_30d,
    m.session_minutes_30d,

    -- change vs their FIRST month: catches slow drift ("new normal")
    m.avg_stake_30d / NULLIF(f.first30_avg_stake, 0)              AS stake_vs_first30,
    m.late_share_30d - f.first30_late_share                       AS late_share_vs_first30,
    m.session_minutes_30d / NULLIF(f.first30_session_minutes, 0)  AS session_vs_first30,

    -- change vs the previous 30 days: catches sudden jumps
    r.latest_stake_jump_ratio,
    r.latest_failed_deposits_7d,

    -- value band: 10 = the 10% of players worth most to the operator so far
    NTILE(10) OVER (ORDER BY l.total_net_loss)      AS value_decile

FROM players AS p
JOIN lifetime AS l            ON p.player_id = l.player_id
CROSS JOIN params AS s
LEFT JOIN first_30 AS f       ON p.player_id = f.player_id
LEFT JOIN last_30 AS m        ON p.player_id = m.player_id
LEFT JOIN last_90 AS n        ON p.player_id = n.player_id
LEFT JOIN latest_rolling AS r ON p.player_id = r.player_id
LEFT JOIN sof AS c            ON p.player_id = c.player_id;
