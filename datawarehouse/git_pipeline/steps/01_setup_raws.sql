CREATE OR ALTER TABLE {{environment}}.RAW.TEST_TABLE (
  ID STRING,
  TEXT_FIELD STRING,
  TEXT_FIELD_2 STRING
);

create or replace table {{environment}}.RAW.land_ohlc (
    asset_pair string,
    JSON variant,
    meta_collection_timestamp datetime
);

create or replace table {{environment}}.RAW.ohlc (
    timestamp timestamp,
    trade_pair string,    
    open number,
    high float,
    low float,
    close float,
    volume float,
    trades number
);