from __future__ import annotations

from dataclasses import dataclass

from .backtest import BacktestEngine
from .config import TrainingConfig
from .model_registry import ModelRegistry
from .models import ModelArtifact, TrainingOutcome
from .utils import utc_now


@dataclass(slots=True)
class TrainingRow:
    timestamp: str
    mid_price: float
    features: dict[str, float]
    future_return_bps: float


class Trainer:
    def __init__(self, config: TrainingConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.backtester = BacktestEngine()

    def build_training_rows(self, feature_rows: list[dict]) -> list[TrainingRow]:
        rows: list[TrainingRow] = []
        lookahead = self.config.lookahead_bars
        for idx in range(len(feature_rows) - lookahead):
            current = feature_rows[idx]
            future = feature_rows[idx + lookahead]
            current_mid = float(current["mid_price"])
            future_mid = float(future["mid_price"])
            if not current_mid:
                continue
            future_return_bps = ((future_mid - current_mid) / current_mid) * 10_000.0
            rows.append(
                TrainingRow(
                    timestamp=current["timestamp"],
                    mid_price=current_mid,
                    features={key: float(value) for key, value in current["features"].items()},
                    future_return_bps=future_return_bps,
                )
            )
        return rows

    def train(self, feature_rows: list[dict]) -> TrainingOutcome | None:
        rows = self.build_training_rows(feature_rows)
        if len(rows) < self.config.min_training_rows:
            return None

        rows = rows[-self.config.train_window_rows :]
        split_idx = max(1, len(rows) - self.config.validation_rows)
        train_rows = rows[:split_idx]
        validation_rows = rows[split_idx:]

        weights: dict[str, float] = {}
        feature_names = sorted(train_rows[0].features.keys())
        label_mean = sum(row.future_return_bps for row in train_rows) / len(train_rows)

        for feature_name in feature_names:
            xs = [row.features[feature_name] for row in train_rows]
            ys = [row.future_return_bps for row in train_rows]
            x_mean = sum(xs) / len(xs)
            variance = sum((x - x_mean) ** 2 for x in xs)
            covariance = sum((x - x_mean) * (y - label_mean) for x, y in zip(xs, ys))
            weights[feature_name] = 0.0 if variance == 0 else covariance / variance / 10.0

        intercept = label_mean / 10.0
        artifact = ModelArtifact(
            version=f"model-{utc_now().strftime('%Y%m%d%H%M%S')}",
            model_type="linear-covariance",
            created_at=utc_now(),
            weights=weights,
            intercept=intercept,
            metrics={},
        )

        evaluation_rows = [
            {"features": row.features, "future_return_bps": row.future_return_bps, "mid_price": row.mid_price}
            for row in validation_rows
        ]
        result = self.backtester.run(evaluation_rows, artifact)
        metrics = {
            "expectancy_bps": result.expectancy_bps,
            "gross_expectancy_bps": result.gross_expectancy_bps,
            "win_rate": result.win_rate,
            "turnover": result.turnover,
            "max_drawdown_usd": result.max_drawdown_usd,
            "total_pnl_usd": result.total_pnl_usd,
            "gross_pnl_usd": result.gross_pnl_usd,
            "total_cost_usd": result.total_cost_usd,
            "trades": float(result.trades),
        }
        artifact.metrics = metrics

        rejection_reasons: list[str] = []
        if result.expectancy_bps < self.config.promotion_min_expectancy_bps:
            rejection_reasons.append("expectancy below minimum")
        if result.win_rate < self.config.promotion_min_win_rate:
            rejection_reasons.append("win rate below minimum")
        if result.max_drawdown_usd > self.config.promotion_max_drawdown_usd:
            rejection_reasons.append("max drawdown above limit")
        if result.turnover > self.config.promotion_max_turnover:
            rejection_reasons.append("turnover above limit")

        accepted = not rejection_reasons
        self.registry.save(artifact)
        if accepted:
            self.registry.promote(artifact)

        return TrainingOutcome(
            accepted=accepted,
            artifact=artifact,
            metrics=metrics,
            rejection_reasons=rejection_reasons,
        )
