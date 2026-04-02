from __future__ import annotations

from pathlib import Path

from hyperliquid_bot.config import TrainingConfig
from hyperliquid_bot.model_registry import ModelRegistry
from hyperliquid_bot.trainer import Trainer


def test_trainer_promotes_when_validation_passes(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    trainer = Trainer(
        TrainingConfig(
            min_training_rows=20,
            train_window_rows=100,
            validation_rows=10,
            lookahead_bars=1,
            promotion_min_expectancy_bps=0.1,
            promotion_min_win_rate=0.5,
            promotion_max_drawdown_usd=1000.0,
            promotion_max_turnover=1000.0,
        ),
        registry,
    )

    rows = []
    mid = 100.0
    for idx in range(40):
        rows.append(
            {
                "timestamp": f"2026-01-01T00:{idx:02d}:00+00:00",
                "symbol": "BTC",
                "mid_price": mid,
                "spread_bps": 1.0,
                "features": {
                    "depth_imbalance": 1.0,
                    "trade_imbalance": 0.8,
                    "momentum_5": 0.01,
                },
            }
        )
        mid += 0.2

    outcome = trainer.train(rows)

    assert outcome is not None
    assert outcome.accepted
    assert registry.load_active() is not None
