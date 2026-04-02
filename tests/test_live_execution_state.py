from __future__ import annotations

from datetime import UTC, datetime

from hyperliquid_bot.execution import ExecutionEngine
from hyperliquid_bot.models import ExecutionReport, FeatureVector, PortfolioState, RiskDecision, TradingDecision
from hyperliquid_bot.utils import utc_now


class _LiveAdapter:
    def __init__(self) -> None:
        self.open_orders: list[dict[str, object]] = []
        self.limit_orders: list[tuple[str, str, float, float, bool]] = []

    def build_portfolio_state(self, symbol: str) -> PortfolioState:
        return PortfolioState(
            timestamp=datetime.now(tz=UTC),
            symbol=symbol,
            account_value_usd=1000.0,
            position_size=0.0,
            entry_price=0.0,
            mark_price=100.0,
            leverage=0.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_usd=0.0,
            daily_pnl_usd=0.0,
            open_orders=len(self.open_orders),
        )

    def get_open_orders_for_symbol(self, symbol: str) -> list[dict[str, object]]:
        return [order for order in self.open_orders if order.get("coin") == symbol]

    def place_limit_order(self, symbol: str, side: str, size: float, limit_price: float, reduce_only: bool) -> dict[str, object]:
        self.limit_orders.append((symbol, side, size, limit_price, reduce_only))
        return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}}}

    def place_ioc_order(self, symbol: str, side: str, size: float) -> dict[str, object]:  # pragma: no cover - not used
        return {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": str(size)}}]}}}

    def close_position(self, symbol: str) -> dict[str, object]:  # pragma: no cover - not used
        return {"status": "ok", "response": {"data": {"statuses": [{"filled": {"coin": symbol}}]}}}

    def safe_report(self, symbol: str, action: str, response: dict[str, object], success: bool = True) -> ExecutionReport:
        return ExecutionReport(timestamp=utc_now(), symbol=symbol, action=action, success=success, message="accepted", response=dict(response))


def _decision() -> TradingDecision:
    return TradingDecision(
        timestamp=utc_now(),
        symbol="BTC",
        action="enter",
        side="buy",
        target_notional_usd=100.0,
        order_type="limit",
        limit_price=100.0,
        reduce_only=False,
    )


def _features() -> FeatureVector:
    return FeatureVector(timestamp=utc_now(), symbol="BTC", values={}, mid_price=100.0, spread_bps=1.0)


def test_live_execution_requires_exchange_reconcile_before_order() -> None:
    engine = ExecutionEngine(_LiveAdapter())  # type: ignore[arg-type]

    report = engine.execute(_decision(), RiskDecision(allowed=True, reasons=[]), _features())

    assert not report.success
    assert report.message == "exchange state not reconciled"


def test_live_execution_blocks_when_exchange_has_open_orders() -> None:
    adapter = _LiveAdapter()
    adapter.open_orders = [{"coin": "BTC", "oid": 55, "side": "buy", "sz": "0.5", "limitPx": "100.0"}]
    engine = ExecutionEngine(adapter)  # type: ignore[arg-type]

    portfolio = engine.reconcile("BTC", use_exchange=True)
    report = engine.execute(_decision(), RiskDecision(allowed=True, reasons=[]), _features())

    assert portfolio.open_orders == 1
    assert not report.success
    assert report.message == "open exchange orders require reconciliation"
    assert engine.live_state("BTC").status == "blocked_open_orders"


def test_live_execution_marks_state_pending_after_submission() -> None:
    adapter = _LiveAdapter()
    engine = ExecutionEngine(adapter)  # type: ignore[arg-type]
    engine.reconcile("BTC", use_exchange=True)

    report = engine.execute(_decision(), RiskDecision(allowed=True, reasons=[]), _features())
    state = engine.live_state("BTC")

    assert report.success
    assert len(adapter.limit_orders) == 1
    assert state.status == "needs_reconcile"
    assert state.pending_reconcile is True
    assert state.last_action == "enter"


def test_order_update_can_mark_live_order_as_resting() -> None:
    adapter = _LiveAdapter()
    engine = ExecutionEngine(adapter)  # type: ignore[arg-type]
    engine.reconcile("BTC", use_exchange=True)
    engine.execute(_decision(), RiskDecision(allowed=True, reasons=[]), _features())

    state = engine.handle_order_update(
        {"coin": "BTC", "oid": 123, "side": "buy", "sz": "1.0", "limitPx": "100.0", "status": "resting"}
    )

    assert state is not None
    assert state.status == "blocked_open_orders"
    assert state.pending_reconcile is False
    assert len(state.open_orders) == 1


def test_user_fill_clears_pending_reconcile_when_no_open_orders_remain() -> None:
    adapter = _LiveAdapter()
    engine = ExecutionEngine(adapter)  # type: ignore[arg-type]
    engine.reconcile("BTC", use_exchange=True)
    engine.execute(_decision(), RiskDecision(allowed=True, reasons=[]), _features())

    state = engine.handle_user_fill({"coin": "BTC", "oid": 123, "side": "buy", "sz": "1.0", "startPosition": "0.0"})

    assert state is not None
    assert state.status == "ready"
    assert state.pending_reconcile is False
    assert state.position_size == 1.0


def test_user_cancel_removes_open_order_from_live_state() -> None:
    adapter = _LiveAdapter()
    engine = ExecutionEngine(adapter)  # type: ignore[arg-type]
    engine.reconcile("BTC", use_exchange=True)
    engine.handle_order_update(
        {"coin": "BTC", "oid": 123, "side": "buy", "sz": "1.0", "limitPx": "100.0", "status": "open"}
    )

    state = engine.handle_user_cancel({"coin": "BTC", "oid": 123})

    assert state is not None
    assert state.status == "ready"
    assert state.pending_reconcile is False
    assert state.open_orders == []
