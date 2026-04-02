from __future__ import annotations

import logging
import queue
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from .config import BotConfig
from .execution import ExecutionEngine
from .exchange_adapter import HyperliquidAdapter
from .features import FeaturePipeline
from .market_data import MarketDataService
from .model_registry import ModelRegistry
from .models import FeatureVector, Incident, PortfolioState
from .monitoring import MonitoringService
from .policy import PolicyEngine
from .risk import RiskManager
from .signal_engine import SignalEngine
from .signal_engine import WeightedFeatureModel
from .storage import StorageManager
from .trainer import Trainer
from .utils import to_jsonable, utc_now

LOGGER = logging.getLogger(__name__)


class AutonomousBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.storage = StorageManager(config.storage)
        self.registry = ModelRegistry(config.storage.models_dir)
        self.monitoring = MonitoringService(config.monitoring, config.storage.status_path)
        self.adapter = HyperliquidAdapter(config)
        self.market_data = MarketDataService(config.market.symbol)
        self.features = FeaturePipeline(config.market.depth_levels, config.market.feature_window)
        self.signal = SignalEngine(self._load_model())
        self.policy = PolicyEngine(config.strategy, config.risk, config.execution)
        self.risk = RiskManager(config.risk)
        self.execution = ExecutionEngine(self.adapter)
        self.trainer = Trainer(config.training, self.registry)
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.last_training_check_at: datetime | None = None
        self.portfolio = PortfolioState(
            timestamp=utc_now(),
            symbol=config.market.symbol,
            account_value_usd=0.0,
            position_size=0.0,
            entry_price=0.0,
            mark_price=0.0,
            leverage=0.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_usd=0.0,
            daily_pnl_usd=0.0,
            open_orders=0,
        )

    def _load_model(self):
        artifact = self.registry.load_active()
        return WeightedFeatureModel.from_artifact(artifact) if artifact else None

    def bootstrap(self) -> None:
        self.adapter.connect()
        self.monitoring.serve()
        self.portfolio = self.execution.reconcile(self.config.market.symbol)
        self.monitoring.set_metric("account_value_usd", self.portfolio.account_value_usd)
        self.monitoring.set_metric("mode", self.config.execution.mode)
        self.monitoring.set_metric("active_model_version", self.signal.model.version)
        self._persist_health()

    def handle_event(self, stream_type: str, message: Any) -> None:
        try:
            envelope = self.market_data.normalize(stream_type, message)
            self.storage.record_raw_event(
                {"timestamp": utc_now().isoformat(), "stream_type": stream_type, "payload": to_jsonable(envelope.raw)}
            )
            feature = self._route_envelope(envelope)
            if feature is None:
                return
            self._evaluate(feature)
        except Exception as exc:
            self._record_incident("error", "Event handling failure", str(exc))
            LOGGER.exception("failed to handle %s event", stream_type)

    def _route_envelope(self, envelope) -> FeatureVector | None:
        if envelope.book is not None:
            return self.features.ingest_book(envelope.book)
        if envelope.trades:
            feature = None
            for trade in envelope.trades:
                feature = self.features.ingest_trade(trade)
            return feature
        if envelope.candle is not None:
            return self.features.ingest_candle(envelope.candle)
        if envelope.context is not None:
            return self.features.ingest_context(envelope.context)
        return None

    def _evaluate(self, feature: FeatureVector) -> None:
        row = {
            "timestamp": feature.timestamp.isoformat(),
            "symbol": feature.symbol,
            "mid_price": feature.mid_price,
            "spread_bps": feature.spread_bps,
            "features": feature.values,
        }
        self.storage.record_feature_row(row)
        prediction = self.signal.predict(feature)
        self.monitoring.set_metric("latest_expected_return_bps", prediction.expected_return_bps)
        self.portfolio = self.execution.reconcile(self.config.market.symbol)
        decision = self.policy.decide(prediction, feature, self.portfolio)
        risk_decision = self.risk.evaluate(decision, self.portfolio, feature)
        if (
            decision.action in {"enter", "add", "flip"}
            and self.portfolio.open_orders >= self.config.execution.max_open_orders
        ):
            risk_decision.allowed = False
            risk_decision.reasons.append("max open orders reached")

        if risk_decision.kill_switch:
            self.risk.pause_after_stop()
            self._record_incident("critical", "Kill switch engaged", ", ".join(risk_decision.reasons))

        if self.config.execution.mode == "paper":
            report_payload = {
                "timestamp": utc_now().isoformat(),
                "symbol": decision.symbol,
                "action": decision.action,
                "success": risk_decision.allowed,
                "mode": "paper",
                "decision": to_jsonable(decision),
                "risk": to_jsonable(risk_decision),
            }
            self.storage.record_execution(report_payload)
            self.monitoring.set_metric("last_action", decision.action)
            self._persist_health()
            return

        if self.config.execution.mode == "shadow":
            report_payload = {
                "timestamp": utc_now().isoformat(),
                "symbol": decision.symbol,
                "action": decision.action,
                "success": risk_decision.allowed,
                "mode": "shadow",
                "decision": to_jsonable(decision),
                "risk": to_jsonable(risk_decision),
            }
            self.storage.record_execution(report_payload)
            self.monitoring.set_metric("last_action", f"shadow:{decision.action}")
            self._persist_health()
            return

        report = self.execution.execute(decision, risk_decision, feature)
        self.storage.record_execution(to_jsonable(asdict(report)))
        if report.success:
            self.risk.on_success()
        else:
            self.risk.on_reject()
            self._record_incident("warning", "Execution failure", report.message)
        self.monitoring.set_metric("last_action", report.action)
        self._persist_health()

    def run(self) -> None:
        self.bootstrap()
        self.adapter.subscribe_market_streams(self.config.market.symbol, lambda stream, msg: self.queue.put((stream, msg)))
        LOGGER.info("bot running in %s mode for %s", self.config.execution.mode, self.config.market.symbol)
        try:
            while True:
                try:
                    stream_type, message = self.queue.get(timeout=1.0)
                except queue.Empty:
                    self._maybe_train()
                    self._persist_health()
                    continue
                self.handle_event(stream_type, message)
                self._maybe_train()
        finally:
            self.shutdown()

    def train(self) -> dict[str, Any]:
        outcome = self._run_training_cycle()
        if outcome is None:
            return {"accepted": False, "reason": "not enough training rows"}
        return outcome

    def _run_training_cycle(self) -> dict[str, Any] | None:
        outcome = self.trainer.train(self.storage.load_feature_rows())
        self.last_training_check_at = utc_now()
        if outcome is None:
            self.monitoring.set_metric("last_training_status", "not_enough_rows")
            self.monitoring.set_metric("last_training_at", self.last_training_check_at.isoformat())
            self._persist_health()
            return None

        payload = {
            "timestamp": utc_now().isoformat(),
            "accepted": outcome.accepted,
            "model_version": outcome.artifact.version,
            "metrics": outcome.metrics,
            "reasons": outcome.rejection_reasons,
        }
        self.storage.record_training_run(payload)
        self.monitoring.set_metric("last_training_status", "accepted" if outcome.accepted else "rejected")
        self.monitoring.set_metric("last_training_at", payload["timestamp"])
        self.monitoring.set_metric("last_training_model_version", outcome.artifact.version)
        if outcome.accepted:
            self.signal.load_model(outcome.artifact)
            self.monitoring.set_metric("active_model_version", outcome.artifact.version)
        self._persist_health()
        return payload

    def _maybe_train(self) -> None:
        if not self.config.training.enabled:
            return
        now = utc_now()
        interval = timedelta(hours=self.config.training.retrain_interval_hours)
        if self.last_training_check_at is not None and (now - self.last_training_check_at) < interval:
            return
        try:
            self._run_training_cycle()
        except Exception as exc:
            self.last_training_check_at = now
            self._record_incident("error", "Training cycle failure", str(exc))
            LOGGER.exception("scheduled training cycle failed")

    def health(self) -> dict[str, Any]:
        self._persist_health()
        status = self.monitoring.status()
        status["portfolio"] = to_jsonable(self.portfolio)
        active = self.registry.load_active()
        status["active_model"] = to_jsonable(active) if active else None
        return status

    def _persist_health(self) -> None:
        self.monitoring.heartbeat()
        self.storage.record_health(self.monitoring.status())

    def _record_incident(self, severity: str, title: str, details: str) -> None:
        incident = Incident(timestamp=utc_now(), severity=severity, title=title, details=details)
        self.monitoring.report_incident(incident)
        self.storage.record_incident(to_jsonable(incident))

    def shutdown(self) -> None:
        self.monitoring.shutdown()
        self.adapter.close()
