from __future__ import annotations

from dataclasses import dataclass

from .exchange_adapter import HyperliquidAdapter
from .models import ExecutionReport, FeatureVector, PaperPositionState, PortfolioState, RiskDecision, TradingDecision
from .utils import clamp, utc_now


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
        paper_latency_bps: float = 0.5,
        paper_max_latency_bps: float = 3.0,
        paper_fill_tolerance_bps: float = 2.5,
        paper_partial_fill_min_fraction: float = 0.25,
    ):
        self.adapter = adapter
        self.paper_starting_balance_usd = paper_starting_balance_usd
        self.paper_fee_bps = paper_fee_bps
        self.paper_slippage_bps = paper_slippage_bps
        self.paper_latency_bps = paper_latency_bps
        self.paper_max_latency_bps = paper_max_latency_bps
        self.paper_fill_tolerance_bps = paper_fill_tolerance_bps
        self.paper_partial_fill_min_fraction = paper_partial_fill_min_fraction
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

        fill_ratio = self._paper_fill_ratio(decision, features)
        filled_quantity = quantity * fill_ratio
        if filled_quantity <= 0:
            return ExecutionOutcome(
                report=ExecutionReport(
                    timestamp=utc_now(),
                    symbol=decision.symbol,
                    action=decision.action,
                    success=False,
                    message="paper no fill",
                    response={
                        "requested_size": quantity,
                        "filled_size": 0.0,
                        "fill_ratio": 0.0,
                        "latency_bps": self._paper_latency_penalty_bps(features),
                    },
                ),
                portfolio=portfolio,
            )

        fill_price = self._paper_fill_price(decision, features)
        target_delta = filled_quantity if decision.side == "buy" else -filled_quantity
        if decision.action == "flip":
            target_delta = -current_pos + target_delta
        self._apply_paper_fill(target_delta, fill_price)
        updated = self._paper_portfolio(decision.symbol, features.mid_price)
        filled_size = abs(target_delta)
        message = "paper fill" if fill_ratio >= 0.999 else "paper partial fill"
        return ExecutionOutcome(
            report=ExecutionReport(
                timestamp=utc_now(),
                symbol=decision.symbol,
                action=decision.action,
                success=True,
                message=message,
                response={
                    "fill_price": fill_price,
                    "requested_size": quantity,
                    "filled_size": filled_size,
                    "fill_ratio": filled_size / quantity if quantity > 0 else 0.0,
                    "unfilled_size": max(0.0, quantity - filled_size),
                    "latency_bps": self._paper_latency_penalty_bps(features),
                },
            ),
            portfolio=updated,
        )

    def _paper_fill_price(self, decision: TradingDecision, features: FeatureVector) -> float:
        adverse_move_bps = self._paper_adverse_move_bps(decision, features)
        if decision.order_type == "limit":
            return float(decision.limit_price or features.mid_price)
        slip_bps = self.paper_slippage_bps + (features.spread_bps / 2.0) + adverse_move_bps
        slip = features.mid_price * (slip_bps / 10_000.0)
        return features.mid_price + slip if decision.side == "buy" else features.mid_price - slip

    def _paper_fill_ratio(self, decision: TradingDecision, features: FeatureVector) -> float:
        if decision.order_type == "ioc" or decision.action == "exit":
            return 1.0

        limit_price = float(decision.limit_price or features.mid_price)
        effective_mid = self._paper_effective_mid_price(decision, features)
        synthetic_touch = self._paper_touch_price(decision.side, effective_mid, features.spread_bps)

        if decision.side == "buy" and limit_price >= synthetic_touch:
            return 1.0
        if decision.side == "sell" and limit_price <= synthetic_touch:
            return 1.0

        distance_bps = self._paper_distance_to_touch_bps(decision.side, limit_price, synthetic_touch)
        tolerance_bps = self.paper_fill_tolerance_bps + (features.spread_bps / 2.0)
        proximity = clamp(1.0 - (distance_bps / tolerance_bps), 0.0, 1.0)
        if proximity <= 0.0:
            return 0.0
        return self.paper_partial_fill_min_fraction + (
            proximity * (1.0 - self.paper_partial_fill_min_fraction)
        )

    def _paper_effective_mid_price(self, decision: TradingDecision, features: FeatureVector) -> float:
        adverse_move_bps = self._paper_adverse_move_bps(decision, features)
        if adverse_move_bps <= 0:
            return features.mid_price
        move = features.mid_price * (adverse_move_bps / 10_000.0)
        return features.mid_price + move if decision.side == "buy" else features.mid_price - move

    def _paper_latency_penalty_bps(self, features: FeatureVector) -> float:
        realized_vol = abs(features.values.get("realized_vol_20", 0.0))
        vol_bps = min(realized_vol * 10_000.0 * 0.2, self.paper_max_latency_bps)
        return clamp(self.paper_latency_bps + vol_bps, 0.0, self.paper_max_latency_bps)

    def _paper_adverse_move_bps(self, decision: TradingDecision, features: FeatureVector) -> float:
        latency_bps = self._paper_latency_penalty_bps(features)
        if latency_bps <= 0:
            return 0.0

        momentum = float(features.values.get("momentum_5", 0.0))
        trade_imbalance = float(features.values.get("trade_imbalance", 0.0))
        depth_imbalance = float(features.values.get("depth_imbalance", 0.0))
        pressure = clamp((momentum * 100.0) + (trade_imbalance * 0.75) - (depth_imbalance * 0.35), -1.0, 1.0)
        adverse_pressure = max(0.0, pressure) if decision.side == "buy" else max(0.0, -pressure)
        return latency_bps * (0.25 + (0.75 * adverse_pressure))

    @staticmethod
    def _paper_touch_price(side: str, mid_price: float, spread_bps: float) -> float:
        half_spread = mid_price * ((spread_bps / 2.0) / 10_000.0)
        return mid_price + half_spread if side == "buy" else mid_price - half_spread

    @staticmethod
    def _paper_distance_to_touch_bps(side: str, limit_price: float, touch_price: float) -> float:
        if touch_price <= 0:
            return 0.0
        if side == "buy":
            return max(0.0, ((touch_price - limit_price) / touch_price) * 10_000.0)
        return max(0.0, ((limit_price - touch_price) / touch_price) * 10_000.0)

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
