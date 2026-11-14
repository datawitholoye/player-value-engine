-- Patterns you can only see bet by bet: chasing losses, betting on
-- unfamiliar markets, and how long sessions really last. Rolled up to player-day.

CREATE OR REPLACE TABLE bet_patterns AS
WITH ordered AS (
    SELECT
        player_id,
        placed_at,
        CAST(placed_at AS DATE)  AS bet_date,
        market,
        stake,
        LAG(outcome)   OVER by_time AS prev_outcome,
        LAG(stake)     OVER by_time AS prev_stake,
        LAG(placed_at) OVER by_time AS prev_placed_at,
        MIN(placed_at) OVER (PARTITION BY player_id, market) AS first_time_on_market,
        MIN(placed_at) OVER (PARTITION BY player_id)         AS first_bet_ever
    FROM bets
    WINDOW by_time AS (PARTITION BY player_id ORDER BY placed_at)
),
flagged AS (
    SELECT
        *,
        -- chasing: stake at least doubled within 30 minutes of a lost bet
        CASE WHEN prev_outcome = 'lost'
              AND stake >= 2 * prev_stake
              AND placed_at - prev_placed_at <= INTERVAL 30 MINUTE
             THEN 1 ELSE 0 END                                     AS is_chase,
        CASE WHEN prev_outcome = 'lost' THEN 1 ELSE 0 END          AS after_loss,
        -- a bet on a market they never used in their first month
        CASE WHEN first_time_on_market > first_bet_ever + INTERVAL 30 DAY
             THEN 1 ELSE 0 END                                     AS unfamiliar_market,
        -- a gap of over 30 minutes starts a new session
        CASE WHEN prev_placed_at IS NULL
               OR placed_at - prev_placed_at > INTERVAL 30 MINUTE
             THEN 1 ELSE 0 END                                     AS new_session
    FROM ordered
),
sessions AS (
    SELECT
        *,
        -- running total of session starts = a session number per player
        SUM(new_session) OVER (PARTITION BY player_id ORDER BY placed_at
                               ROWS UNBOUNDED PRECEDING)          AS session_no
    FROM flagged
),
session_lengths AS (
    SELECT
        player_id,
        session_no,
        MIN(bet_date)                                         AS bet_date,
        DATE_DIFF('minute', MIN(placed_at), MAX(placed_at))   AS session_minutes
    FROM sessions
    GROUP BY player_id, session_no
),
daily_sessions AS (
    SELECT
        player_id,
        bet_date,
        COUNT(*)             AS n_sessions,
        AVG(session_minutes) AS avg_session_minutes,
        MAX(session_minutes) AS longest_session_minutes
    FROM session_lengths
    GROUP BY player_id, bet_date
)
SELECT
    s.player_id,
    s.bet_date,
    SUM(s.is_chase)          AS chase_bets,
    SUM(s.after_loss)        AS bets_after_loss,
    SUM(s.unfamiliar_market) AS unfamiliar_market_bets,
    d.n_sessions,
    d.avg_session_minutes,
    d.longest_session_minutes
FROM sessions AS s
LEFT JOIN daily_sessions AS d
    ON s.player_id = d.player_id
   AND s.bet_date  = d.bet_date
GROUP BY s.player_id, s.bet_date, d.n_sessions, d.avg_session_minutes, d.longest_session_minutes;
