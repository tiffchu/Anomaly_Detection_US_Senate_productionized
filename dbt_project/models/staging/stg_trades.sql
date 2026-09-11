with source as (
    select *
    from {{ source('congress', 'raw_trades') }}
),

cleaned as (
    select
        coalesce(trade_id, md5(
            coalesce(member, '') || '|' ||
            coalesce(member_slug, '') || '|' ||
            coalesce(ticker, '') || '|' ||
            coalesce(type, '') || '|' ||
            coalesce(cast(amount_low as varchar), '') || '|' ||
            coalesce(cast(amount_high as varchar), '') || '|' ||
            coalesce(cast(transaction_date as varchar), '') || '|' ||
            coalesce(cast(disclosure_date as varchar), '')
        )) as trade_id,
        trim(member) as member_name,
        trim(member_slug) as member_slug,
        lower(trim(chamber)) as chamber,
        trim(state) as state,
        upper(trim(ticker)) as ticker,
        trim(asset) as asset_name,
        lower(trim(type)) as transaction_type,
        amount_low,
        amount_high,
        trim(amount_range) as amount_range,
        transaction_date,
        disclosure_date,
        date_diff('day', transaction_date, disclosure_date) as disclosure_lag_days,
        est_price as estimated_trade_price,
        recent_price,
        recent_price_date,
        perf_pct as performance_pct,
        realized_return_pct,
        lower(trim(outcome)) as outcome,
        trim(filing_portal) as filing_portal
    from source
)

select *
from cleaned
