from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hyperliquid_bot.config import ExecutionConfig, RiskConfig, StrategyConfig
from hyperliquid_bot.models import FeatureVector, PortfolioState, SignalPrediction
from hyperliquid_bot.policy import PolicyEngine
from hyperliquid_bot.risk import RiskManager


def _portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime.now(tz=UTC),
        symbol="BTC",
        account_value_usd=1000.0,
        position_size=0.0,
        entry_price=0.0,
        mark_price=100.0,
        leverage=1.0,
        unrealized_pnl_usd=0.0,
        realized_pnl_usd=0.0,
        daily_pnl_usd=0.0,
        open_orders=0,
    )


def test_policy_enters_long_when_edge_is_strong() -> None:
    policy = PolicyEngine(StrategyConfig(), RiskConfig(), ExecutionConfig())
    feature = FeatureVector(
        timestamp=datetime.now(tz=UTC),
        symbol="BTC",
        values={"realized_vol_20": 0.01},
        mid_price=100.0,
        spread_bps=2.0,
    )
    prediction = SignalPrediction(
        timestamp=feature.timestamp,
        symbol="BTC",
        model_version="test",
        expected_return_bps=4.0,
        adverse_move_bps=1.0,
        confidence=0.8,
        score=0.4,
        feature_values=feature.values,
    )

    decision = policy.decide(prediction, feature, _portfolio())

    assert decision.action in {"enter", "add"}
    assert decision.side == "buy"
    assert decision.target_notional_usd >= 50.0


def test_risk_blocks_stale_data() -> None:
    risk = RiskManager(RiskConfig(max_data_age_s=1.0))
    feature = FeatureVector(
        timestamp=datetime.now(tz=UTC) - timedelta(seconds=10),
        symbol="BTC",
        values={},
        mid_price=100.0,
        spread_bps=1.0,
    )
    prediction = SignalPrediction(
        timestamp=feature.timestamp,
        symbol="BTC",
        model_version="test",
        expected_return_bps=4.0,
        adverse_move_bps=1.0,
        confidence=0.8,
        score=0.4,
        feature_values={},
    )
    decision = PolicyEngine(StrategyConfig(), RiskConfig(), ExecutionConfig()).decide(prediction, feature, _portfolio())

    result = risk.evaluate(decision, _portfolio(), feature)

    assert not result.allowed
    assert "market data stale" in result.reasons
