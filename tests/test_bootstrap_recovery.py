from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from hyperliquid_bot.config import (
    BacktestConfig,
    BotConfig,
    ExecutionConfig,
    HyperliquidConfig,
    MarketConfig,
    MonitoringConfig,
    RiskConfig,
    StorageConfig,
    StrategyConfig,
    TrainingConfig,
)
from hyperliquid_bot.orchestrator import AutonomousBot
from hyperliquid_bot.utils import utc_now


def _bot_config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        hyperliquid=HyperliquidConfig(),
        market=MarketConfig(startup_candle_lookback=30, min_warm_price_points=20),
        execution=ExecutionConfig(mode="paper", reconcile_interval_s=1),
        risk=RiskConfig(max_data_age_s=1.0),
        strategy=StrategyConfig(),
        training=TrainingConfig(enabled=False),
        backtest=BacktestConfig(),
        storage=StorageConfig(
            root_dir=str(tmp_path / "data"),
            sqlite_path=str(tmp_path / "data" / "runtime.db"),
            raw_events_path=str(tmp_path / "data" / "raw_events.jsonl"),
            features_path=str(tmp_path / "data" / "features.jsonl"),
            models_dir=str(tmp_path / "data" / "models"),
            reports_dir=str(tmp_path / "data" / "reports"),
            status_path=str(tmp_path / "data" / "status.json"),
        ),
        monitoring=MonitoringConfig(),
        source_path=tmp_path / "bot.toml",
    )


class _FakeAdapter:
    def __init__(self) -> None:
        self.connect_calls: list[bool] = []
        self.l2_calls = 0

    def connect(self, *, require_exchange: bool = False) -> None:
        self.connect_calls.append(require_exchange)

    def close(self) -> None:
        return

    def subscribe_market_streams(self, symbol: str, callback) -> None:  # pragma: no cover - not used in these tests
        return

    def get_candles(self, symbol: str, interval: str, start_time: int, end_time: int) -> list[dict[str, str | int]]:
        base = datetime.now(tz=UTC) - timedelta(minutes=30)
        candles: list[dict[str, str | int]] = []
        for idx in range(30):
            open_price = 100.0 + idx
            close_price = open_price + 0.5
            candles.append(
                {
                    "t": int((base + timedelta(minutes=idx)).timestamp() * 1000),
                    "T": int((base + timedelta(minutes=idx + 1)).timestamp() * 1000),
                    "s": symbol,
                    "i": interval,
                    "o": f"{open_price:.2f}",
                    "h": f"{close_price + 0.25:.2f}",
                    "l": f"{open_price - 0.25:.2f}",
                    "c": f"{close_price:.2f}",
                    "v": "10.0",
                }
            )
        return candles

    def get_all_mids(self) -> dict[str, str]:
        return {"BTC": "130.0"}

    def get_asset_context(self, symbol: str) -> dict[str, object]:
        return {
            "coin": symbol,
            "ctx": {
                "time": int(utc_now().timestamp() * 1000),
                "markPx": "130.0",
                "midPx": "130.0",
                "funding": "0.0001",
                "openInterest": "12345.0",
                "premium": "0.0002",
            },
        }

    def get_l2_snapshot(self, symbol: str) -> dict[str, object]:
        self.l2_calls += 1
        return {
            "coin": symbol,
            "time": int(utc_now().timestamp() * 1000),
            "levels": [
                [{"px": "129.95", "sz": "5.0", "n": 1}],
                [{"px": "130.05", "sz": "5.0", "n": 1}],
            ],
        }


def test_bootstrap_warms_feature_pipeline_and_sets_metrics(tmp_path: Path) -> None:
    bot = AutonomousBot(_bot_config(tmp_path))
    bot.adapter = _FakeAdapter()  # type: ignore[assignment]

    bot.bootstrap()

    assert bot.adapter.connect_calls == [False]
    assert bot.features.is_ready(bot.config.market.min_warm_price_points)
    assert bot.monitoring.metrics["feature_ready"] is True
    assert bot.monitoring.metrics["market_bootstrap_status"] == "ready"
    assert bot.monitoring.metrics["warm_price_points"] >= bot.config.market.min_warm_price_points
    assert bot.portfolio.mark_price == 130.0


def test_runtime_refresh_rebootstraps_when_market_data_turns_stale(tmp_path: Path) -> None:
    bot = AutonomousBot(_bot_config(tmp_path))
    bot.adapter = _FakeAdapter()  # type: ignore[assignment]
    bot.bootstrap()
    initial_calls = bot.adapter.l2_calls
    stale_time = utc_now() - timedelta(minutes=5)

    for container in (bot.features.books, bot.features.trades, bot.features.candles, bot.features.contexts):
        if container:
            container[-1].timestamp = stale_time

    bot.last_reconcile_at = utc_now() - timedelta(seconds=5)
    bot._maybe_refresh_runtime_state()

    assert bot.adapter.l2_calls == initial_calls + 1
    assert bot.monitoring.metrics["market_bootstrap_status"] == "ready"
