from __future__ import annotations

import math
from collections import deque
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

        mids = [item.mid_price for item in self.books if item.mid_price > 0]
        recent_returns = []
        for idx in range(1, len(mids)):
            prev = mids[idx - 1]
            curr = mids[idx]
            if prev:
                recent_returns.append((curr - prev) / prev)

        momentum_5 = ((mids[-1] / mids[-5]) - 1.0) if len(mids) >= 5 and mids[-5] else 0.0
        momentum_20 = ((mids[-1] / mids[-20]) - 1.0) if len(mids) >= 20 and mids[-20] else 0.0
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
