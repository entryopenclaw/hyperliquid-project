from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .models import AssetContext, CandleBar, OrderBookLevel, OrderBookSnapshot, TradeTick
from .utils import utc_now


def _parse_dt_ms(value: int | float | None) -> datetime:
    if not value:
        return utc_now()
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)


@dataclass(slots=True)
class NormalizedEnvelope:
    stream_type: str
    raw: dict[str, Any]
    book: OrderBookSnapshot | None = None
    trades: list[TradeTick] = field(default_factory=list)
    candle: CandleBar | None = None
    context: AssetContext | None = None
    order_updates: list[dict[str, Any]] = field(default_factory=list)
    user_fills: list[dict[str, Any]] = field(default_factory=list)
    user_cancels: list[dict[str, Any]] = field(default_factory=list)


class MarketDataService:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def normalize(self, stream_type: str, message: Any) -> NormalizedEnvelope:
        raw = message if isinstance(message, dict) else {"payload": message}
        envelope = NormalizedEnvelope(stream_type=stream_type, raw=raw)

        if stream_type == "l2Book":
            levels = raw.get("data", raw)
            book_levels = levels.get("levels", [[], []])
            bids = [
                OrderBookLevel(price=float(level["px"]), size=float(level["sz"]), count=int(level.get("n", 0)))
                for level in book_levels[0]
            ]
            asks = [
                OrderBookLevel(price=float(level["px"]), size=float(level["sz"]), count=int(level.get("n", 0)))
                for level in book_levels[1]
            ]
            best_bid = bids[0].price if bids else 0.0
            best_ask = asks[0].price if asks else best_bid
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else max(best_bid, best_ask)
            spread_bps = ((best_ask - best_bid) / mid * 10_000.0) if mid else 0.0
            envelope.book = OrderBookSnapshot(
                timestamp=_parse_dt_ms(levels.get("time")),
                symbol=levels.get("coin", self.symbol),
                bids=bids,
                asks=asks,
                mid_price=mid,
                spread_bps=spread_bps,
            )

        elif stream_type == "trades":
            payload = raw.get("data", raw)
            trades = payload if isinstance(payload, list) else payload.get("trades", [])
            envelope.trades = [
                TradeTick(
                    timestamp=_parse_dt_ms(trade.get("time")),
                    symbol=trade.get("coin", self.symbol),
                    price=float(trade.get("px", 0.0) or 0.0),
                    size=float(trade.get("sz", 0.0) or 0.0),
                    side=str(trade.get("side", trade.get("dir", "buy"))).lower(),
                )
                for trade in trades
            ]

        elif stream_type == "candle":
            payload = raw.get("data", raw)
            candle = payload.get("candle", payload)
            envelope.candle = CandleBar(
                timestamp=_parse_dt_ms(candle.get("t") or candle.get("T")),
                symbol=candle.get("s", self.symbol),
                interval=str(candle.get("i", "1m")),
                open_price=float(candle.get("o", 0.0) or 0.0),
                high_price=float(candle.get("h", 0.0) or 0.0),
                low_price=float(candle.get("l", 0.0) or 0.0),
                close_price=float(candle.get("c", 0.0) or 0.0),
                volume=float(candle.get("v", 0.0) or 0.0),
            )

        elif stream_type in {"activeAssetCtx", "allMids"}:
            payload = raw.get("data", raw)
            if stream_type == "allMids":
                mid = float(payload.get(self.symbol, 0.0) or 0.0)
                envelope.context = AssetContext(
                    timestamp=utc_now(),
                    symbol=self.symbol,
                    mark_price=mid,
                    mid_price=mid,
                    funding_rate=0.0,
                    open_interest=0.0,
                    premium=0.0,
                )
            else:
                ctx = payload.get("ctx", payload)
                envelope.context = AssetContext(
                    timestamp=_parse_dt_ms(ctx.get("time")),
                    symbol=payload.get("coin", self.symbol),
                    mark_price=float(ctx.get("markPx", 0.0) or 0.0),
                    mid_price=float(ctx.get("midPx", 0.0) or 0.0),
                    funding_rate=float(ctx.get("funding", 0.0) or 0.0),
                    open_interest=float(ctx.get("openInterest", 0.0) or 0.0),
                    premium=float(ctx.get("premium", 0.0) or 0.0),
                )

        elif stream_type == "orderUpdates":
            payload = raw.get("data", raw)
            if isinstance(payload, list):
                envelope.order_updates = [item for item in payload if isinstance(item, dict)]
            elif isinstance(payload, dict):
                updates = payload.get("orderUpdates") or payload.get("orders") or payload.get("data") or []
                if isinstance(updates, list):
                    envelope.order_updates = [item for item in updates if isinstance(item, dict)]

        elif stream_type == "userFills":
            payload = raw.get("data", raw)
            if isinstance(payload, dict):
                fills = payload.get("fills", [])
                if isinstance(fills, list):
                    envelope.user_fills = [item for item in fills if isinstance(item, dict)]

        elif stream_type == "userEvents":
            payload = raw.get("data", raw)
            if isinstance(payload, dict):
                fills = payload.get("fills", [])
                cancels = payload.get("nonUserCancel", [])
                if isinstance(fills, list):
                    envelope.user_fills = [item for item in fills if isinstance(item, dict)]
                if isinstance(cancels, list):
                    envelope.user_cancels = [item for item in cancels if isinstance(item, dict)]

        return envelope
