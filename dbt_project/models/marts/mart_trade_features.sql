with trades as (
    select *
    from {{ ref('stg_trades') }}
),

features as (
    select
        trade_id,
        member_name,
        member_slug,
        chamber,
        state,
        ticker,
        asset_name,
        transaction_type,
        amount_low,
        amount_high,
        amount_range,
        transaction_date,
        disclosure_date,
        disclosure_lag_days,
        estimated_trade_price,
        recent_price,
        recent_price_date,
        performance_pct,
        outcome,
        filing_portal,
        case
            when transaction_type = 'purchase' then amount_high
            when transaction_type = 'sale' then -amount_high
            else 0
        end as signed_amount_high,
        avg(amount_high) over (partition by member_slug) as member_avg_amount_high,
        sum(
            case
                when transaction_type = 'purchase' then amount_high
                when transaction_type = 'sale' then -amount_high
                else 0
            end
        ) over (
            partition by member_slug
            order by transaction_date, disclosure_date, trade_id
            rows between unbounded preceding and current row
        ) as member_running_net_amount,
        count(*) over (partition by member_slug) as member_trade_count,
        count(*) over (partition by ticker) as ticker_trade_count
    from trades
)

select *
from features
