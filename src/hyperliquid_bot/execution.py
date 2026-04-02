from __future__ import annotations

from dataclasses import dataclass

from .exchange_adapter import HyperliquidAdapter
from .models import ExecutionReport, FeatureVector, PaperPositionState, PortfolioState, RiskDecision, TradingDecision
from .utils import utc_now


@dataclass(slots=True)
class ExecutionOutcome:
    report: ExecutionReport
    portfolio: PortfolioState


class ExecutionEngine:
    def __init__(
        self,
        adapter: HyperliquidAdapter,
        *,
        paper_starting_balance_usd: float = 10_000.0,
        paper_fee_bps: float = 0.0,
        paper_slippage_bps: float = 0.0,
    ):
        self.adapter = adapter
        self.paper_starting_balance_usd = paper_starting_balance_usd
        self.paper_fee_bps = paper_fee_bps
        self.paper_slippage_bps = paper_slippage_bps
        self.paper_state = PaperPositionState(
            cash_balance_usd=paper_starting_balance_usd,
            position_size=0.0,
            entry_price=0.0,
            realized_pnl_usd=0.0,
            fees_paid_usd=0.0,
        )

    def reconcile(self, symbol: str, *, mark_price: float | None = None, use_exchange: bool = True) -> PortfolioState:
        if use_exchange:
            return self.adapter.build_portfolio_state(symbol)
        return self._paper_portfolio(symbol, mark_price or self.paper_state.last_mark_price)

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
        if quantity <= 0:
            return ExecutionReport(
                timestamp=decision.timestamp,
                symbol=decision.symbol,
                action=decision.action,
                success=False,
                message="zero order quantity",
                response={},
            )

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

    def execute_paper(self, decision: TradingDecision, risk: RiskDecision, features: FeatureVector) -> ExecutionOutcome:
        portfolio = self._paper_portfolio(decision.symbol, features.mid_price)
        if not risk.allowed:
            return ExecutionOutcome(
                report=ExecutionReport(
                    timestamp=decision.timestamp,
                    symbol=decision.symbol,
                    action=decision.action,
                    success=False,
                    message="blocked by risk",
                    response={"reasons": risk.reasons},
                ),
                portfolio=portfolio,
            )

        if decision.action == "hold" or decision.order_type == "none":
            return ExecutionOutcome(
                report=ExecutionReport(
                    timestamp=decision.timestamp,
                    symbol=decision.symbol,
                    action="hold",
                    success=True,
                    message="no action",
                    response={},
                ),
                portfolio=portfolio,
            )

        current_pos = self.paper_state.position_size
        quantity = 0.0 if not features.mid_price else decision.target_notional_usd / features.mid_price
        if decision.action == "exit":
            quantity = abs(current_pos)
        if quantity <= 0:
            return ExecutionOutcome(
                report=ExecutionReport(
                    timestamp=decision.timestamp,
                    symbol=decision.symbol,
                    action=decision.action,
                    success=False,
                    message="zero order quantity",
                    response={},
                ),
                portfolio=portfolio,
            )

        fill_price = self._paper_fill_price(decision, features)
        target_delta = quantity if decision.side == "buy" else -quantity
        if decision.action == "flip":
            target_delta = -current_pos + target_delta
        self._apply_paper_fill(target_delta, fill_price)
        updated = self._paper_portfolio(decision.symbol, features.mid_price)
        return ExecutionOutcome(
            report=ExecutionReport(
                timestamp=utc_now(),
                symbol=decision.symbol,
                action=decision.action,
                success=True,
                message="paper fill",
                response={"fill_price": fill_price, "filled_size": abs(target_delta)},
            ),
            portfolio=updated,
        )

    def _paper_fill_price(self, decision: TradingDecision, features: FeatureVector) -> float:
        if decision.order_type == "limit":
            return float(decision.limit_price or features.mid_price)
        slip = features.mid_price * (self.paper_slippage_bps / 10_000.0)
        return features.mid_price + slip if decision.side == "buy" else features.mid_price - slip

    def _apply_paper_fill(self, delta: float, fill_price: float) -> None:
        if delta == 0:
            return
        current = self.paper_state.position_size
        entry = self.paper_state.entry_price
        fee = abs(delta) * fill_price * (self.paper_fee_bps / 10_000.0)
        self.paper_state.cash_balance_usd -= fee
        self.paper_state.fees_paid_usd += fee

        if current == 0 or current * delta > 0:
            new_position = current + delta
            total_size = abs(current) + abs(delta)
            if total_size > 0:
                if current == 0:
                    self.paper_state.entry_price = fill_price
                else:
                    self.paper_state.entry_price = ((abs(current) * entry) + (abs(delta) * fill_price)) / total_size
            self.paper_state.position_size = new_position
            return

        closing_qty = min(abs(current), abs(delta))
        realized = closing_qty * ((fill_price - entry) if current > 0 else (entry - fill_price))
        self.paper_state.cash_balance_usd += realized
        self.paper_state.realized_pnl_usd += realized

        remaining = current + delta
        self.paper_state.position_size = remaining
        if remaining == 0:
            self.paper_state.entry_price = 0.0
        elif current * remaining < 0:
            self.paper_state.entry_price = fill_price

    def _paper_portfolio(self, symbol: str, mark_price: float) -> PortfolioState:
        self.paper_state.last_mark_price = mark_price or self.paper_state.last_mark_price
        if self.paper_state.position_size == 0 or self.paper_state.entry_price == 0 or mark_price == 0:
            unrealized = 0.0
        else:
            unrealized = abs(self.paper_state.position_size) * (
                (mark_price - self.paper_state.entry_price)
                if self.paper_state.position_size > 0
                else (self.paper_state.entry_price - mark_price)
            )
        account_value = self.paper_state.cash_balance_usd + unrealized
        leverage = (abs(self.paper_state.position_size) * mark_price / account_value) if account_value > 0 and mark_price > 0 else 0.0
        return PortfolioState(
            timestamp=utc_now(),
            symbol=symbol,
            account_value_usd=account_value,
            position_size=self.paper_state.position_size,
            entry_price=self.paper_state.entry_price,
            mark_price=mark_price,
            leverage=leverage,
            unrealized_pnl_usd=unrealized,
            realized_pnl_usd=self.paper_state.realized_pnl_usd,
            daily_pnl_usd=account_value - self.paper_starting_balance_usd,
            open_orders=0,
        )
