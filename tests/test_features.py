from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hyperliquid_bot.features import FeaturePipeline
from hyperliquid_bot.models import AssetContext, CandleBar, OrderBookLevel, OrderBookSnapshot, TradeTick


def test_feature_pipeline_builds_microstructure_features() -> None:
    now = datetime.now(tz=UTC)
    pipeline = FeaturePipeline(depth_levels=2, feature_window=50)

    for idx in range(25):
        mid = 100.0 + idx * 0.1
        book = OrderBookSnapshot(
            timestamp=now + timedelta(seconds=idx),
            symbol="BTC",
            bids=[OrderBookLevel(price=mid - 0.05, size=10 + idx), OrderBookLevel(price=mid - 0.10, size=8 + idx)],
            asks=[OrderBookLevel(price=mid + 0.05, size=9), OrderBookLevel(price=mid + 0.10, size=8)],
            mid_price=mid,
            spread_bps=10.0,
        )
        features = pipeline.ingest_book(book)
        pipeline.ingest_trade(
            TradeTick(
                timestamp=now + timedelta(seconds=idx),
                symbol="BTC",
                price=mid,
                size=1.5,
                side="buy",
            )
        )

    pipeline.ingest_candle(
        CandleBar(
            timestamp=now + timedelta(minutes=1),
            symbol="BTC",
            interval="1m",
            open_price=100.0,
            high_price=103.0,
            low_price=99.0,
            close_price=102.0,
            volume=50.0,
        )
    )
    features = pipeline.ingest_context(
        AssetContext(
            timestamp=now + timedelta(minutes=1),
            symbol="BTC",
            mark_price=102.0,
            mid_price=102.0,
            funding_rate=0.0001,
            open_interest=100000.0,
            premium=0.0002,
        )
    )

    assert features is not None
    assert features.values["depth_imbalance"] > 0
    assert features.values["trade_imbalance"] > 0
    assert "realized_vol_20" in features.values
    assert features.values["open_interest"] == 100000.0
