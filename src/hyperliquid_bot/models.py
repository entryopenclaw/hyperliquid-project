from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class OrderBookLevel:
    price: float
    size: float
    count: int = 0


@dataclass(slots=True)
class OrderBookSnapshot:
    timestamp: datetime
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    mid_price: float
    spread_bps: float


@dataclass(slots=True)
class TradeTick:
    timestamp: datetime
    symbol: str
    price: float
    size: float
    side: str


@dataclass(slots=True)
class CandleBar:
    timestamp: datetime
    symbol: str
    interval: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float


@dataclass(slots=True)
class AssetContext:
    timestamp: datetime
    symbol: str
    mark_price: float
    mid_price: float
    funding_rate: float
    open_interest: float
    premium: float


@dataclass(slots=True)
class FeatureVector:
    timestamp: datetime
    symbol: str
    values: dict[str, float]
    mid_price: float
    spread_bps: float


@dataclass(slots=True)
class SignalPrediction:
    timestamp: datetime
    symbol: str
    model_version: str
    expected_return_bps: float
    adverse_move_bps: float
    confidence: float
    score: float
    feature_values: dict[str, float]


@dataclass(slots=True)
class TradingDecision:
    timestamp: datetime
    symbol: str
    action: str
    side: str
    target_notional_usd: float
    order_type: str
    limit_price: float | None
    reduce_only: bool
    rationale: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PortfolioState:
    timestamp: datetime
    symbol: str
    account_value_usd: float
    position_size: float
    entry_price: float
    mark_price: float
    leverage: float
    unrealized_pnl_usd: float
    realized_pnl_usd: float
    daily_pnl_usd: float
    open_orders: int


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    reasons: list[str]
    kill_switch: bool = False


@dataclass(slots=True)
class ExecutionReport:
    timestamp: datetime
    symbol: str
    action: str
    success: bool
    message: str
    response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExchangeOrderState:
    oid: int | None
    symbol: str
    side: str
    size: float
    limit_price: float
    reduce_only: bool
    order_type: str
    timestamp: datetime


@dataclass(slots=True)
class LiveExecutionState:
    timestamp: datetime
    symbol: str
    status: str
    position_size: float
    open_orders: list[ExchangeOrderState] = field(default_factory=list)
    pending_reconcile: bool = True
    last_action: str = ""
    blocked_reason: str = ""


@dataclass(slots=True)
class PaperPositionState:
    cash_balance_usd: float
    position_size: float
    entry_price: float
    realized_pnl_usd: float
    fees_paid_usd: float
    last_mark_price: float = 0.0


@dataclass(slots=True)
class Incident:
    timestamp: datetime
    severity: str
    title: str
    details: str


@dataclass(slots=True)
class ModelArtifact:
    version: str
    model_type: str
    created_at: datetime
    weights: dict[str, float]
    intercept: float
    metrics: dict[str, float]


@dataclass(slots=True)
class TrainingOutcome:
    accepted: bool
    artifact: ModelArtifact
    metrics: dict[str, float]
    rejection_reasons: list[str]
