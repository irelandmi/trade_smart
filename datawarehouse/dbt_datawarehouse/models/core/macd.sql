WITH base AS (
    SELECT
        trade_pair,
        timestamp AS trade_date,
        close AS close,

        -- 12-day SMA of close
        AVG(close) OVER (
            PARTITION BY trade_pair
            ORDER BY timestamp
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) AS sma_12,

        -- 26-day SMA of close
        AVG(close) OVER (
            PARTITION BY trade_pair
            ORDER BY timestamp
            ROWS BETWEEN 25 PRECEDING AND CURRENT ROW
        ) AS sma_26

    FROM dev.raw.ohlc

),
macd_calc AS (
    SELECT
        trade_pair,
        trade_date,
        close,
        sma_12,
        sma_26,
        (sma_12 - sma_26) AS macd_line
    FROM base
),
macd_signal AS (
    SELECT
        trade_pair,
        trade_date,
        close,
        sma_12,
        sma_26,
        macd_line,
        
        -- 9-day SMA of the MACD line as "signal"
        AVG(macd_line) OVER (
            PARTITION BY trade_pair
            ORDER BY trade_date
            ROWS BETWEEN 8 PRECEDING AND CURRENT ROW
        ) AS macd_signal

    FROM macd_calc
)
SELECT
    trade_pair,
    trade_date,
    close,
    macd_line AS macd,
    macd_signal AS signal,
    (macd_line - macd_signal) AS histogram
FROM macd_signal
ORDER BY trade_pair, trade_date