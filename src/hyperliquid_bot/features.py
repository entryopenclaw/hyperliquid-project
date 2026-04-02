from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from statistics import mean, pstdev

from .models import AssetContext, CandleBar, FeatureVector, OrderBookSnapshot, TradeTick


class FeaturePipeline:
    def __init__(self, depth_levels: int = 5, feature_window: int = 200):
        self.depth_levels = depth_levels
        self.books: deque[OrderBookSnapshot] = deque(maxlen=feature_window)
        self.trades: deque[TradeTick] = deque(maxlen=feature_window * 5)
        self.candles: deque[CandleBar] = deque(maxlen=feature_window)
        self.contexts: deque[AssetContext] = deque(maxlen=feature_window)

    def ingest_book(self, book: OrderBookSnapshot) -> FeatureVector:
        self.books.append(book)
        return self._build(book.symbol, book.timestamp)

    def ingest_trade(self, trade: TradeTick) -> FeatureVector | None:
        self.trades.append(trade)
        if not self.books:
            return None
        return self._build(trade.symbol, trade.timestamp)

    def ingest_candle(self, candle: CandleBar) -> FeatureVector | None:
        self.candles.append(candle)
        if not self.books:
            return None
        return self._build(candle.symbol, candle.timestamp)

    def ingest_context(self, context: AssetContext) -> FeatureVector | None:
        self.contexts.append(context)
        if not self.books:
            return None
        return self._build(context.symbol, context.timestamp)

    def reference_price_count(self) -> int:
        return len(self._reference_prices())

    def is_ready(self, min_price_points: int) -> bool:
        return bool(self.books) and self.reference_price_count() >= min_price_points

    def last_market_timestamp(self) -> datetime | None:
        timestamps = [
            container[-1].timestamp
            for container in (self.books, self.trades, self.candles, self.contexts)
            if container
        ]
        return max(timestamps) if timestamps else None

    def last_mid_price(self) -> float:
        if self.books:
            return self.books[-1].mid_price
        if self.contexts:
            return self.contexts[-1].mid_price or self.contexts[-1].mark_price
        if self.candles:
            return self.candles[-1].close_price
        return 0.0

    def _build(self, symbol: str, timestamp) -> FeatureVector:
        book = self.books[-1]
        bids = book.bids[: self.depth_levels]
        asks = book.asks[: self.depth_levels]
        bid_depth = sum(level.size for level in bids)
        ask_depth = sum(level.size for level in asks)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) else 0.0

        trade_sizes = [trade.size if trade.side.startswith("b") else -trade.size for trade in list(self.trades)[-25:]]
        trade_flow = sum(trade_sizes)
        total_flow = sum(abs(size) for size in trade_sizes)
        trade_imbalance = trade_flow / total_flow if total_flow else 0.0

        prices = self._reference_prices()
        recent_returns = []
        for idx in range(1, len(prices)):
            prev = prices[idx - 1]
            curr = prices[idx]
            if prev:
                recent_returns.append((curr - prev) / prev)

        momentum_5 = ((prices[-1] / prices[-5]) - 1.0) if len(prices) >= 5 and prices[-5] else 0.0
        momentum_20 = ((prices[-1] / prices[-20]) - 1.0) if len(prices) >= 20 and prices[-20] else 0.0
        realized_vol = pstdev(recent_returns[-20:]) * math.sqrt(20) if len(recent_returns) >= 2 else 0.0

        last_candle = self.candles[-1] if self.candles else None
        candle_return = 0.0
        candle_range = 0.0
        if last_candle and last_candle.open_price:
            candle_return = (last_candle.close_price - last_candle.open_price) / last_candle.open_price
            candle_range = (last_candle.high_price - last_candle.low_price) / last_candle.open_price

        last_context = self.contexts[-1] if self.contexts else None
        funding = last_context.funding_rate if last_context else 0.0
        open_interest = last_context.open_interest if last_context else 0.0
        premium = last_context.premium if last_context else 0.0

        microprice = self._microprice(book)
        microprice_dev = ((microprice - book.mid_price) / book.mid_price) if book.mid_price else 0.0
        trend_alignment = mean([momentum_5, momentum_20, candle_return]) if any([momentum_5, momentum_20, candle_return]) else 0.0

        values = {
            "spread_bps": book.spread_bps,
            "depth_imbalance": imbalance,
            "trade_imbalance": trade_imbalance,
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
            "realized_vol_20": realized_vol,
            "candle_return": candle_return,
            "candle_range": candle_range,
            "funding_rate": funding,
            "open_interest": open_interest,
            "premium": premium,
            "microprice_deviation": microprice_dev,
            "trend_alignment": trend_alignment,
        }
        return FeatureVector(
            timestamp=timestamp,
            symbol=symbol,
            values=values,
            mid_price=book.mid_price,
            spread_bps=book.spread_bps,
        )

    def _reference_prices(self) -> list[float]:
        candle_prices = [item.close_price for item in self.candles if item.close_price > 0]
        book_prices = [item.mid_price for item in self.books if item.mid_price > 0]
        if len(book_prices) >= 20:
            return book_prices
        return [price for price in [*candle_prices, *book_prices] if price > 0]

    @staticmethod
    def _microprice(book: OrderBookSnapshot) -> float:
        if not book.bids or not book.asks:
            return book.mid_price
        best_bid = book.bids[0]
        best_ask = book.asks[0]
        denom = best_bid.size + best_ask.size
        if not denom:
            return book.mid_price
        return (best_ask.price * best_bid.size + best_bid.price * best_ask.size) / denom
