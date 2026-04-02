from __future__ import annotations

from datetime import UTC, datetime

from hyperliquid_bot.backtest import BacktestEngine
from hyperliquid_bot.models import ModelArtifact


def test_backtest_applies_roundtrip_costs() -> None:
    artifact = ModelArtifact(
        version="test-model",
        model_type="linear",
        created_at=datetime.now(tz=UTC),
        weights={"signal": 1.0},
        intercept=0.0,
        metrics={},
    )
    rows = [
        {"features": {"signal": 1.0}, "future_return_bps": 5.0, "notional_usd": 100.0},
        {"features": {"signal": -1.0}, "future_return_bps": -5.0, "notional_usd": 100.0},
    ]

    result = BacktestEngine().run(rows, artifact, fee_bps=1.0, slippage_bps=1.0, min_signal_bps=0.1)

    assert result.trades == 2
    assert result.gross_expectancy_bps == 5.0
    assert result.expectancy_bps == 1.0
    assert result.total_cost_usd > 0
