from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class BargoQuery:
    from_date: str | None = None
    to_date: str | None = None
    ticker: str | None = None
    member: str | None = None
    chamber: str | None = None
    transaction_type: str | None = None
    limit: int = 100

    def to_params(self, page: int) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": self.limit, "page": page}
        if self.from_date:
            params["from"] = self.from_date
        if self.to_date:
            params["to"] = self.to_date
        if self.ticker:
            params["ticker"] = self.ticker
        if self.member:
            params["member"] = self.member
        if self.chamber:
            params["chamber"] = self.chamber
        if self.transaction_type:
            params["type"] = self.transaction_type
        return params


class BargoCongressTradesClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout_seconds: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-Api-Key": api_key})

    def get_trades_page(self, query: BargoQuery, page: int = 0) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/trades",
            params=query.to_params(page),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def iter_trades(self, query: BargoQuery, max_pages: int | None = None) -> Iterator[dict[str, Any]]:
        page = 0
        while True:
            payload = self.get_trades_page(query, page=page)
            trades = payload.get("trades", [])
            if not trades:
                break

            for trade in trades:
                yield trade

            count = int(payload.get("count", len(trades)))
            limit = int(payload.get("limit", query.limit))
            if count < limit:
                break

            page += 1
            if max_pages is not None and page >= max_pages:
                break
