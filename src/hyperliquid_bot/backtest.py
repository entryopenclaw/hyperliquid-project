from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ModelArtifact
from .signal_engine import WeightedFeatureModel
from .utils import to_jsonable


@dataclass(slots=True)
class BacktestResult:
    expectancy_bps: float
    gross_expectancy_bps: float
    win_rate: float
    turnover: float
    max_drawdown_usd: float
    total_pnl_usd: float
    gross_pnl_usd: float
    total_cost_usd: float
    average_signal_bps: float
    roundtrip_cost_bps: float
    trades: int


class BacktestEngine:
    def run(
        self,
        rows: list[dict[str, Any]],
        artifact: ModelArtifact,
        *,
        notional_usd: float = 100.0,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        min_signal_bps: float = 0.1,
    ) -> BacktestResult:
        model = WeightedFeatureModel.from_artifact(artifact)
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        wins = 0
        trades = 0
        turnover = 0.0
        pnl_samples: list[float] = []
        gross_pnl_samples: list[float] = []
        pnl_bps_samples: list[float] = []
        gross_pnl_bps_samples: list[float] = []
        signal_bps_samples: list[float] = []
        roundtrip_cost_bps = 2.0 * (fee_bps + slippage_bps)

        for row in rows:
            score = artifact.intercept
            for key, value in row["features"].items():
                score += model.weights.get(key, 0.0) * float(value)
            predicted_bps = score * 10.0
            realized_bps = float(row.get("future_return_bps", 0.0))
            if abs(predicted_bps) < min_signal_bps:
                continue
            direction = 1.0 if predicted_bps > 0 else -1.0
            gross_pnl_bps = direction * realized_bps
            pnl_bps = gross_pnl_bps - roundtrip_cost_bps
            row_notional_usd = float(row.get("notional_usd", notional_usd))
            pnl_usd = (pnl_bps / 10_000.0) * row_notional_usd
            gross_pnl_usd = (gross_pnl_bps / 10_000.0) * row_notional_usd
            trades += 1
            turnover += abs(direction)
            equity += pnl_usd
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            if pnl_bps > 0:
                wins += 1
            pnl_samples.append(pnl_usd)
            gross_pnl_samples.append(gross_pnl_usd)
            pnl_bps_samples.append(pnl_bps)
            gross_pnl_bps_samples.append(gross_pnl_bps)
            signal_bps_samples.append(abs(predicted_bps))

        expectancy = (sum(pnl_bps_samples) / trades) if trades else 0.0
        gross_expectancy = (sum(gross_pnl_bps_samples) / trades) if trades else 0.0
        total_pnl = sum(pnl_samples)
        gross_total_pnl = sum(gross_pnl_samples)
        return BacktestResult(
            expectancy_bps=expectancy,
            gross_expectancy_bps=gross_expectancy,
            win_rate=(wins / trades) if trades else 0.0,
            turnover=turnover,
            max_drawdown_usd=max_drawdown,
            total_pnl_usd=total_pnl,
            gross_pnl_usd=gross_total_pnl,
            total_cost_usd=gross_total_pnl - total_pnl,
            average_signal_bps=(sum(signal_bps_samples) / trades) if trades else 0.0,
            roundtrip_cost_bps=roundtrip_cost_bps,
            trades=trades,
        )

    @staticmethod
    def to_report(
        result: BacktestResult,
        artifact: ModelArtifact,
        *,
        rows: int,
        notional_usd: float,
        fee_bps: float,
        slippage_bps: float,
        min_signal_bps: float,
    ) -> dict[str, Any]:
        return {
            "model_version": artifact.version,
            "model_type": artifact.model_type,
            "rows": rows,
            "assumed_notional_usd": notional_usd,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "roundtrip_cost_bps": result.roundtrip_cost_bps,
            "min_signal_bps": min_signal_bps,
            "metrics": to_jsonable(result),
        }
