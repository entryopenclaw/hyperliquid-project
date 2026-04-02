from __future__ import annotations

from dataclasses import dataclass

from .models import ModelArtifact
from .signal_engine import WeightedFeatureModel


@dataclass(slots=True)
class BacktestResult:
    expectancy_bps: float
    win_rate: float
    turnover: float
    max_drawdown_usd: float
    total_pnl_usd: float
    trades: int


class BacktestEngine:
    def run(self, rows: list[dict], artifact: ModelArtifact) -> BacktestResult:
        model = WeightedFeatureModel.from_artifact(artifact)
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        wins = 0
        trades = 0
        turnover = 0.0
        pnl_samples: list[float] = []

        pnl_bps_samples: list[float] = []

        for row in rows:
            score = artifact.intercept
            for key, value in row["features"].items():
                score += model.weights.get(key, 0.0) * float(value)
            predicted_bps = score * 10.0
            realized_bps = float(row.get("future_return_bps", 0.0))
            if abs(predicted_bps) < 0.1:
                continue
            direction = 1.0 if predicted_bps > 0 else -1.0
            pnl_bps = direction * realized_bps
            notional_usd = float(row.get("notional_usd", 100.0))
            pnl_usd = (pnl_bps / 10_000.0) * notional_usd
            trades += 1
            turnover += abs(direction)
            equity += pnl_usd
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            if pnl_bps > 0:
                wins += 1
            pnl_samples.append(pnl_usd)
            pnl_bps_samples.append(pnl_bps)

        expectancy = (sum(pnl_bps_samples) / trades) if trades else 0.0
        return BacktestResult(
            expectancy_bps=expectancy,
            win_rate=(wins / trades) if trades else 0.0,
            turnover=turnover,
            max_drawdown_usd=max_drawdown,
            total_pnl_usd=sum(pnl_samples),
            trades=trades,
        )
