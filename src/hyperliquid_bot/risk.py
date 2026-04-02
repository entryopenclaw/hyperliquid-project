from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import RiskConfig
from .models import FeatureVector, PortfolioState, RiskDecision, TradingDecision
from .utils import utc_now


@dataclass(slots=True)
class RiskState:
    peak_equity_usd: float = 0.0
    reject_streak: int = 0
    paused_until: datetime | None = None
    kill_switch_engaged: bool = False


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()

    def update_equity(self, portfolio: PortfolioState) -> None:
        self.state.peak_equity_usd = max(self.state.peak_equity_usd, portfolio.account_value_usd)

    def on_reject(self) -> None:
        self.state.reject_streak += 1
        if self.state.reject_streak >= self.config.max_reject_streak:
            self.state.kill_switch_engaged = True

    def on_success(self) -> None:
        self.state.reject_streak = 0

    def pause_after_stop(self) -> None:
        self.state.paused_until = utc_now() + timedelta(seconds=self.config.cooldown_after_stop_s)

    def evaluate(self, decision: TradingDecision, portfolio: PortfolioState, features: FeatureVector) -> RiskDecision:
        reasons: list[str] = []
        now = utc_now()
        self.update_equity(portfolio)

        if self.state.kill_switch_engaged:
            reasons.append("kill switch engaged")

        if self.state.paused_until and now < self.state.paused_until:
            reasons.append("cooldown active")

        data_age = (now - features.timestamp).total_seconds()
        if data_age > self.config.max_data_age_s:
            reasons.append("market data stale")

        if features.spread_bps > self.config.max_spread_bps:
            reasons.append("spread too wide")

        if portfolio.account_value_usd < self.config.min_account_value_usd:
            reasons.append("account value below minimum")

        if portfolio.daily_pnl_usd <= -self.config.max_daily_loss_usd:
            reasons.append("daily loss limit reached")
            self.state.kill_switch_engaged = True

        drawdown = max(0.0, self.state.peak_equity_usd - portfolio.account_value_usd)
        if drawdown >= self.config.max_drawdown_usd:
            reasons.append("max drawdown exceeded")
            self.state.kill_switch_engaged = True

        resulting_notional = abs(portfolio.position_size) * features.mid_price
        if decision.action in {"enter", "add", "flip"}:
            resulting_notional = max(resulting_notional, decision.target_notional_usd)
        if resulting_notional > self.config.max_position_notional_usd:
            reasons.append("position notional limit exceeded")

        if portfolio.leverage > self.config.max_leverage:
            reasons.append("leverage already exceeds cap")

        kill_switch = self.state.kill_switch_engaged or any(
            reason in {"daily loss limit reached", "max drawdown exceeded"} for reason in reasons
        )
        return RiskDecision(allowed=not reasons, reasons=reasons, kill_switch=kill_switch)
