from __future__ import annotations

from pathlib import Path

from hyperliquid_bot.execution import ExecutionEngine
from hyperliquid_bot.exchange_adapter import HyperliquidAdapter
from hyperliquid_bot.models import ExecutionReport, FeatureVector, RiskDecision, TradingDecision
from hyperliquid_bot.monitoring import MonitoringService
from hyperliquid_bot.config import MonitoringConfig
from hyperliquid_bot.utils import utc_now


class _FakeAdapter:
    def build_portfolio_state(self, symbol: str):  # pragma: no cover - not used in these tests
        raise NotImplementedError


def test_safe_report_marks_exchange_error_as_failure() -> None:
    adapter = object.__new__(HyperliquidAdapter)

    report = adapter.safe_report(
        "BTC",
        "enter",
        {"status": "ok", "response": {"data": {"statuses": [{"error": "order rejected"}]}}},
    )

    assert not report.success
    assert report.message == "order rejected"


def test_execution_engine_blocks_zero_quantity_orders() -> None:
    engine = ExecutionEngine(_FakeAdapter())  # type: ignore[arg-type]
    decision = TradingDecision(
        timestamp=utc_now(),
        symbol="BTC",
        action="enter",
        side="buy",
        target_notional_usd=10.0,
        order_type="limit",
        limit_price=100.0,
        reduce_only=False,
    )
    risk = RiskDecision(allowed=True, reasons=[])
    features = FeatureVector(
        timestamp=utc_now(),
        symbol="BTC",
        values={},
        mid_price=0.0,
        spread_bps=1.0,
    )

    report = engine.execute(decision, risk, features)

    assert isinstance(report, ExecutionReport)
    assert not report.success
    assert report.message == "zero order quantity"


def test_monitoring_creates_parent_dir_for_status_file(tmp_path: Path) -> None:
    status_path = tmp_path / "nested" / "status.json"
    monitoring = MonitoringService(MonitoringConfig(), str(status_path))

    monitoring.heartbeat()

    assert status_path.exists()
