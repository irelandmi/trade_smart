CREATE OR ALTER TABLE {{environment}}.CORE.ohlc (
    timestamp timestamp,
    trade_pair string,    
    open number,
    high float,
    low float,
    close float,
    volume float,
    trades number
);