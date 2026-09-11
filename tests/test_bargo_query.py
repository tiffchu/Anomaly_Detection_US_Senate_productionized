from ingestion.bargo_client import BargoQuery


def test_bargo_query_omits_empty_filters() -> None:
    params = BargoQuery(from_date="2026-01-01", limit=50).to_params(page=2)

    assert params == {"from": "2026-01-01", "limit": 50, "page": 2}


def test_bargo_query_maps_filters_to_api_names() -> None:
    params = BargoQuery(
        from_date="2026-01-01",
        to_date="2026-02-01",
        ticker="NVDA",
        member="Pelosi",
        chamber="house",
        transaction_type="purchase",
    ).to_params(page=0)

    assert params["from"] == "2026-01-01"
    assert params["to"] == "2026-02-01"
    assert params["ticker"] == "NVDA"
    assert params["member"] == "Pelosi"
    assert params["chamber"] == "house"
    assert params["type"] == "purchase"
