WITH base AS (
    SELECT
        trade_pair,
        timestamp,
        close,
        LAG(close) OVER (
            PARTITION BY trade_pair
            ORDER BY timestamp
        ) AS prev_close
    FROM dev.raw.ohlc 
),
deltas AS (
    SELECT
        trade_pair,
        timestamp,
        close,
        (close - prev_close) AS delta,
        GREATEST((close - prev_close), 0) AS gain,
        GREATEST(-(close - prev_close), 0) AS loss
    FROM base
),
averages AS (
    SELECT
        trade_pair,
        timestamp,
        close,
        delta,
        gain,
        loss,
        AVG(gain) OVER (
            PARTITION BY trade_pair
            ORDER BY timestamp
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS avg_gain_14,
        AVG(loss) OVER (
            PARTITION BY trade_pair
            ORDER BY timestamp
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS avg_loss_14
    FROM deltas
)
SELECT
    trade_pair,
    timestamp,
    close,
    delta,
    gain,
    loss,
    avg_gain_14,
    avg_loss_14,
    CASE WHEN avg_gain_14 IS NULL OR avg_loss_14 IS NULL THEN NULL
         ELSE 100 - (100 / (1 + (avg_gain_14 / NULLIF(avg_loss_14, 0))))
    END AS rsi_14
FROM averages
ORDER BY trade_pair, timestamp