with parsed as (
select 
    timestamp,
    message:channel::string as channel,
    message:type::string as type,
    parse_json(message:data[0]::variant) as json
from TF_TEST.DATA.VW_KRAKEN_BATCHES_FLATTENED
)

select 
    timestamp,
    channel,
    type,
    -- json,
    json:ask::float as ask,
    json:ask_qty::float as ask_qty,
    json:bid::float as bid,
    json:bid_qty::float as bid_qty,
    json:change::float as change,
    json:change_pct::float as change_pct,
    json:high::float as high,
    json:last::float as last,
    json:low::float as low,
    json:symbol::string as symbol,
    json:volume::float as volume,
    json:vwap::float as vwap,
from parsed
order by timestamp