from __future__ import annotations

from .exchange_adapter import HyperliquidAdapter
from .models import ExecutionReport, FeatureVector, PortfolioState, RiskDecision, TradingDecision


class ExecutionEngine:
    def __init__(self, adapter: HyperliquidAdapter):
        self.adapter = adapter

    def reconcile(self, symbol: str) -> PortfolioState:
        return self.adapter.build_portfolio_state(symbol)

    def execute(self, decision: TradingDecision, risk: RiskDecision, features: FeatureVector) -> ExecutionReport:
        if not risk.allowed:
            return ExecutionReport(
                timestamp=decision.timestamp,
                symbol=decision.symbol,
                action=decision.action,
                success=False,
                message="blocked by risk",
                response={"reasons": risk.reasons},
            )

        if decision.action == "hold" or decision.order_type == "none":
            return ExecutionReport(
                timestamp=decision.timestamp,
                symbol=decision.symbol,
                action="hold",
                success=True,
                message="no action",
                response={},
            )

        quantity = 0.0 if not features.mid_price else decision.target_notional_usd / features.mid_price
        if decision.action == "exit":
            response = self.adapter.close_position(decision.symbol)
            return self.adapter.safe_report(decision.symbol, "exit", response)

        if decision.order_type == "ioc":
            response = self.adapter.place_ioc_order(decision.symbol, decision.side, quantity)
            return self.adapter.safe_report(decision.symbol, decision.action, response)

        response = self.adapter.place_limit_order(
            decision.symbol,
            decision.side,
            quantity,
            float(decision.limit_price or features.mid_price),
            reduce_only=decision.reduce_only,
        )
        return self.adapter.safe_report(decision.symbol, decision.action, response)
