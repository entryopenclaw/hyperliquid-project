from __future__ import annotations

from .config import ExecutionConfig, RiskConfig, StrategyConfig
from .models import FeatureVector, PortfolioState, SignalPrediction, TradingDecision
from .utils import clamp


class PolicyEngine:
    def __init__(self, strategy: StrategyConfig, risk: RiskConfig, execution: ExecutionConfig):
        self.strategy = strategy
        self.risk = risk
        self.execution = execution

    def decide(self, prediction: SignalPrediction, features: FeatureVector, portfolio: PortfolioState) -> TradingDecision:
        rationale: list[str] = []
        expected = prediction.expected_return_bps
        confidence = prediction.confidence
        current_notional = abs(portfolio.position_size) * features.mid_price

        if confidence < self.strategy.min_confidence:
            rationale.append("confidence below threshold")
            return TradingDecision(
                timestamp=prediction.timestamp,
                symbol=prediction.symbol,
                action="hold",
                side="flat",
                target_notional_usd=current_notional,
                order_type="none",
                limit_price=None,
                reduce_only=False,
                rationale=rationale,
            )

        if portfolio.position_size > 0 and expected < self.strategy.exit_threshold_bps:
            rationale.append("long exit threshold reached")
            return TradingDecision(
                timestamp=prediction.timestamp,
                symbol=prediction.symbol,
                action="exit",
                side="sell",
                target_notional_usd=current_notional,
                order_type="ioc",
                limit_price=None,
                reduce_only=True,
                rationale=rationale,
            )

        if portfolio.position_size < 0 and expected > -self.strategy.exit_threshold_bps:
            rationale.append("short exit threshold reached")
            return TradingDecision(
                timestamp=prediction.timestamp,
                symbol=prediction.symbol,
                action="exit",
                side="buy",
                target_notional_usd=current_notional,
                order_type="ioc",
                limit_price=None,
                reduce_only=True,
                rationale=rationale,
            )

        if expected >= self.strategy.long_entry_bps:
            side = "buy"
            rationale.append("long signal")
        elif expected <= self.strategy.short_entry_bps:
            side = "sell"
            rationale.append("short signal")
        else:
            rationale.append("edge below entry threshold")
            return TradingDecision(
                timestamp=prediction.timestamp,
                symbol=prediction.symbol,
                action="hold",
                side="flat",
                target_notional_usd=current_notional,
                order_type="none",
                limit_price=None,
                reduce_only=False,
                rationale=rationale,
            )

        edge_strength = abs(expected) / max(abs(self.strategy.long_entry_bps), abs(self.strategy.short_entry_bps))
        target_notional = self.risk.base_order_notional_usd * confidence * edge_strength * self.strategy.position_scale
        target_notional = clamp(target_notional, self.risk.base_order_notional_usd, self.risk.max_position_notional_usd)

        urgency = confidence >= self.strategy.urgency_confidence and features.spread_bps <= self.risk.max_spread_bps / 2.0
        order_type = "ioc" if urgency else "limit"
        offset = features.mid_price * (self.execution.limit_offset_bps / 10_000.0)
        limit_price = None
        if order_type == "limit":
            limit_price = features.mid_price - offset if side == "buy" else features.mid_price + offset

        action = "enter"
        if (portfolio.position_size > 0 and side == "buy") or (portfolio.position_size < 0 and side == "sell"):
            action = "add"
        elif portfolio.position_size != 0 and ((portfolio.position_size > 0 and side == "sell") or (portfolio.position_size < 0 and side == "buy")):
            action = "flip"

        return TradingDecision(
            timestamp=prediction.timestamp,
            symbol=prediction.symbol,
            action=action,
            side=side,
            target_notional_usd=target_notional,
            order_type=order_type,
            limit_price=limit_price,
            reduce_only=False,
            rationale=rationale,
        )
