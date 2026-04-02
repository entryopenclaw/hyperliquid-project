from __future__ import annotations

from hyperliquid_bot.execution import ExecutionEngine
from hyperliquid_bot.models import FeatureVector, RiskDecision, TradingDecision
from hyperliquid_bot.utils import utc_now


class _FakeAdapter:
    def build_portfolio_state(self, symbol: str):  # pragma: no cover - not used in paper tests
        raise NotImplementedError


def test_paper_execution_updates_portfolio_and_pnl() -> None:
    engine = ExecutionEngine(_FakeAdapter(), paper_starting_balance_usd=1000.0, paper_fee_bps=0.0, paper_slippage_bps=0.0)  # type: ignore[arg-type]
    buy = TradingDecision(
        timestamp=utc_now(),
        symbol="BTC",
        action="enter",
        side="buy",
        target_notional_usd=100.0,
        order_type="limit",
        limit_price=100.0,
        reduce_only=False,
    )
    risk = RiskDecision(allowed=True, reasons=[])
    features = FeatureVector(timestamp=utc_now(), symbol="BTC", values={}, mid_price=100.0, spread_bps=1.0)

    opened = engine.execute_paper(buy, risk, features)
    marked = engine.reconcile("BTC", mark_price=110.0, use_exchange=False)
    sell = TradingDecision(
        timestamp=utc_now(),
        symbol="BTC",
        action="exit",
        side="sell",
        target_notional_usd=100.0,
        order_type="ioc",
        limit_price=None,
        reduce_only=True,
    )
    closed = engine.execute_paper(sell, risk, FeatureVector(timestamp=utc_now(), symbol="BTC", values={}, mid_price=110.0, spread_bps=1.0))

    assert opened.report.success
    assert opened.portfolio.position_size > 0
    assert marked.unrealized_pnl_usd > 0
    assert closed.portfolio.position_size == 0
    assert closed.portfolio.account_value_usd > 1000.0


def test_passive_limit_order_partially_fills_in_paper_mode() -> None:
    engine = ExecutionEngine(_FakeAdapter(), paper_starting_balance_usd=1000.0, paper_fee_bps=0.0, paper_slippage_bps=0.0)  # type: ignore[arg-type]
    buy = TradingDecision(
        timestamp=utc_now(),
        symbol="BTC",
        action="enter",
        side="buy",
        target_notional_usd=100.0,
        order_type="limit",
        limit_price=99.99,
        reduce_only=False,
    )
    risk = RiskDecision(allowed=True, reasons=[])
    features = FeatureVector(
        timestamp=utc_now(),
        symbol="BTC",
        values={"momentum_5": 0.0, "trade_imbalance": 0.0, "depth_imbalance": 0.0},
        mid_price=100.0,
        spread_bps=1.0,
    )

    opened = engine.execute_paper(buy, risk, features)

    assert opened.report.success
    assert opened.report.message == "paper partial fill"
    assert 0.0 < opened.report.response["fill_ratio"] < 1.0
    assert 0.0 < opened.portfolio.position_size < 1.0


def test_aggressive_buy_order_pays_latency_penalty_in_paper_mode() -> None:
    engine = ExecutionEngine(
        _FakeAdapter(),
        paper_starting_balance_usd=1000.0,
        paper_fee_bps=0.0,
        paper_slippage_bps=1.0,
        paper_latency_bps=1.0,
        paper_max_latency_bps=4.0,
    )  # type: ignore[arg-type]
    buy = TradingDecision(
        timestamp=utc_now(),
        symbol="BTC",
        action="enter",
        side="buy",
        target_notional_usd=100.0,
        order_type="ioc",
        limit_price=None,
        reduce_only=False,
    )
    risk = RiskDecision(allowed=True, reasons=[])
    features = FeatureVector(
        timestamp=utc_now(),
        symbol="BTC",
        values={
            "momentum_5": 0.03,
            "trade_imbalance": 1.0,
            "depth_imbalance": -1.0,
            "realized_vol_20": 0.002,
        },
        mid_price=100.0,
        spread_bps=1.0,
    )

    opened = engine.execute_paper(buy, risk, features)

    assert opened.report.success
    assert opened.report.response["fill_price"] > 100.0
    assert opened.report.response["latency_bps"] > 1.0


def test_far_passive_limit_order_can_miss_in_paper_mode() -> None:
    engine = ExecutionEngine(
        _FakeAdapter(),
        paper_starting_balance_usd=1000.0,
        paper_fee_bps=0.0,
        paper_slippage_bps=0.0,
        paper_fill_tolerance_bps=1.0,
    )  # type: ignore[arg-type]
    buy = TradingDecision(
        timestamp=utc_now(),
        symbol="BTC",
        action="enter",
        side="buy",
        target_notional_usd=100.0,
        order_type="limit",
        limit_price=99.95,
        reduce_only=False,
    )
    risk = RiskDecision(allowed=True, reasons=[])
    features = FeatureVector(timestamp=utc_now(), symbol="BTC", values={}, mid_price=100.0, spread_bps=1.0)

    opened = engine.execute_paper(buy, risk, features)

    assert not opened.report.success
    assert opened.report.message == "paper no fill"
    assert opened.portfolio.position_size == 0.0
