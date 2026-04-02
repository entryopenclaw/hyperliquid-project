from __future__ import annotations

import json
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
from .utils import interval_to_milliseconds, to_jsonable, utc_now
from .backtest import BacktestEngine

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
        self.execution = ExecutionEngine(
            self.adapter,
            paper_starting_balance_usd=config.execution.paper_starting_balance_usd,
            paper_fee_bps=config.backtest.fee_bps,
            paper_slippage_bps=config.backtest.slippage_bps,
            paper_latency_bps=config.execution.paper_latency_bps,
            paper_max_latency_bps=config.execution.paper_max_latency_bps,
            paper_fill_tolerance_bps=config.execution.paper_fill_tolerance_bps,
            paper_partial_fill_min_fraction=config.execution.paper_partial_fill_min_fraction,
        )
        self.trainer = Trainer(config.training, self.registry)
        self.backtester = BacktestEngine()
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.last_training_check_at: datetime | None = None
        self.last_reconcile_at: datetime | None = None
        self.last_market_bootstrap_at: datetime | None = None
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
        self.adapter.connect(require_exchange=self.config.execution.mode == "live")
        self.monitoring.serve()
        self._bootstrap_market_state()
        self.portfolio = self._current_portfolio(mark_price=self.features.last_mid_price())
        self.monitoring.set_metric("account_value_usd", self.portfolio.account_value_usd)
        self.monitoring.set_metric("mode", self.config.execution.mode)
        self.monitoring.set_metric("active_model_version", self.signal.model.version)
        self._update_feature_metrics()
        self._update_execution_metrics()
        self._update_portfolio_metrics(self.portfolio)
        self._persist_health()

    def handle_event(self, stream_type: str, message: Any) -> None:
        try:
            envelope = self.market_data.normalize(stream_type, message)
            self.storage.record_raw_event(
                {"timestamp": utc_now().isoformat(), "stream_type": stream_type, "payload": to_jsonable(envelope.raw)}
            )
            self._handle_execution_envelope(envelope)
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

    def _handle_execution_envelope(self, envelope) -> None:
        if self.config.execution.mode == "paper":
            return

        changed = False
        for update in envelope.order_updates:
            state = self.execution.handle_order_update(update)
            changed = changed or state is not None
        for fill in envelope.user_fills:
            state = self.execution.handle_user_fill(fill)
            changed = changed or state is not None
        for cancel in envelope.user_cancels:
            state = self.execution.handle_user_cancel(cancel)
            changed = changed or state is not None

        if not changed:
            return

        self._update_execution_metrics()
        self.monitoring.set_metric("last_execution_event_at", utc_now().isoformat())
        if self.config.execution.mode == "live" or (
            self.config.execution.mode == "shadow"
            and (self.config.secret_key or self.config.hyperliquid.account_address)
        ):
            self.portfolio = self._current_portfolio(mark_price=self.features.last_mid_price())
            self.last_reconcile_at = utc_now()
            self._update_portfolio_metrics(self.portfolio)
        self._persist_health()

    def _evaluate(self, feature: FeatureVector) -> None:
        self._update_feature_metrics()
        if not self.features.is_ready(self.config.market.min_warm_price_points):
            self.monitoring.set_metric("last_action", "warming")
            self._persist_health()
            return

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
        self.portfolio = self._current_portfolio(mark_price=feature.mid_price)
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
            outcome = self.execution.execute_paper(decision, risk_decision, feature)
            self.portfolio = outcome.portfolio
            report_payload = {
                "timestamp": utc_now().isoformat(),
                "symbol": decision.symbol,
                "action": decision.action,
                "success": outcome.report.success,
                "mode": "paper",
                "decision": to_jsonable(decision),
                "risk": to_jsonable(risk_decision),
                "execution": to_jsonable(outcome.report),
                "portfolio": to_jsonable(self.portfolio),
            }
            self.storage.record_execution(report_payload)
            self.monitoring.set_metric("last_action", decision.action)
            self._update_portfolio_metrics(self.portfolio)
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
            self._update_execution_metrics()
            self._update_portfolio_metrics(self.portfolio)
            self._persist_health()
            return

        report = self.execution.execute(decision, risk_decision, feature)
        self.portfolio = self._current_portfolio(mark_price=feature.mid_price)
        self.storage.record_execution(to_jsonable(asdict(report)))
        if report.success:
            self.risk.on_success()
        else:
            self.risk.on_reject()
            self._record_incident("warning", "Execution failure", report.message)
        self.monitoring.set_metric("last_action", report.action)
        self._update_execution_metrics()
        self._update_portfolio_metrics(self.portfolio)
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
                    self._maybe_refresh_runtime_state()
                    self._maybe_train()
                    continue
                self.handle_event(stream_type, message)
                self._maybe_refresh_runtime_state()
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

    def backtest(self, model_version: str | None = None) -> dict[str, Any]:
        feature_rows = self.storage.load_feature_rows()
        rows = self.trainer.build_training_rows(feature_rows)
        if not rows:
            return {"accepted": False, "reason": "not enough feature rows for backtest"}

        artifact = self.registry.load(model_version) if model_version else self.registry.load_active()
        if artifact is None:
            artifact = self._heuristic_artifact()

        evaluation_rows = [
            {
                "features": row.features,
                "future_return_bps": row.future_return_bps,
                "mid_price": row.mid_price,
                "notional_usd": self.config.backtest.assumed_notional_usd,
            }
            for row in rows
        ]
        result = self.backtester.run(
            evaluation_rows,
            artifact,
            notional_usd=self.config.backtest.assumed_notional_usd,
            fee_bps=self.config.backtest.fee_bps,
            slippage_bps=self.config.backtest.slippage_bps,
            min_signal_bps=self.config.backtest.min_signal_bps,
        )
        report = self.backtester.to_report(
            result,
            artifact,
            rows=len(evaluation_rows),
            notional_usd=self.config.backtest.assumed_notional_usd,
            fee_bps=self.config.backtest.fee_bps,
            slippage_bps=self.config.backtest.slippage_bps,
            min_signal_bps=self.config.backtest.min_signal_bps,
        )
        report_path = (
            self.storage.reports_dir
            / f"backtest-{artifact.version}-{utc_now().strftime('%Y%m%d%H%M%S')}.json"
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        self.monitoring.set_metric("last_backtest_model_version", artifact.version)
        self.monitoring.set_metric("last_backtest_path", str(report_path))
        self.monitoring.set_metric("last_backtest_expectancy_bps", result.expectancy_bps)
        self._persist_health()
        return {"accepted": True, "report_path": str(report_path), **report}

    @staticmethod
    def _heuristic_artifact():
        model = WeightedFeatureModel.heuristic()
        from .models import ModelArtifact

        return ModelArtifact(
            version=model.version,
            model_type="heuristic",
            created_at=utc_now(),
            weights=model.weights,
            intercept=model.intercept,
            metrics={},
        )

    def _persist_health(self) -> None:
        self.monitoring.heartbeat()
        self.storage.record_health(self.monitoring.status())

    def _bootstrap_market_state(self) -> None:
        symbol = self.config.market.symbol
        end_time = utc_now()
        candle_ms = interval_to_milliseconds(self.config.market.candle_interval)
        lookback_ms = candle_ms * self.config.market.startup_candle_lookback
        start_ms = int((end_time - timedelta(milliseconds=lookback_ms)).timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        try:
            candles = self.adapter.get_candles(symbol, self.config.market.candle_interval, start_ms, end_ms)
            for candle in candles[-self.config.market.startup_candle_lookback :]:
                envelope = self.market_data.normalize("candle", {"data": candle})
                self._route_envelope(envelope)

            mids = self.adapter.get_all_mids()
            if mids:
                self._route_envelope(self.market_data.normalize("allMids", {"data": mids}))

            asset_context = self.adapter.get_asset_context(symbol)
            if asset_context:
                self._route_envelope(self.market_data.normalize("activeAssetCtx", {"data": asset_context}))

            book = self.adapter.get_l2_snapshot(symbol)
            self._route_envelope(self.market_data.normalize("l2Book", {"data": book}))

            self.last_market_bootstrap_at = utc_now()
            self.monitoring.set_metric("market_bootstrap_status", "ready" if self.features.is_ready(self.config.market.min_warm_price_points) else "warming")
            self.monitoring.set_metric("last_market_bootstrap_at", self.last_market_bootstrap_at.isoformat())
            self._update_feature_metrics()
        except Exception as exc:
            self.monitoring.set_metric("market_bootstrap_status", "failed")
            self._record_incident("warning", "Market bootstrap failure", str(exc))
            LOGGER.exception("failed to bootstrap market state")

    def _maybe_refresh_runtime_state(self) -> None:
        now = utc_now()
        interval = timedelta(seconds=self.config.execution.reconcile_interval_s)
        if self.last_reconcile_at is not None and (now - self.last_reconcile_at) < interval:
            return

        self.last_reconcile_at = now
        try:
            market_ts = self.features.last_market_timestamp()
            stale_after_s = max(self.config.execution.reconcile_interval_s, int(self.config.risk.max_data_age_s))
            if market_ts is None or (now - market_ts).total_seconds() > stale_after_s:
                self._bootstrap_market_state()

            self.portfolio = self._current_portfolio(mark_price=self.features.last_mid_price())
            self._maybe_cancel_stale_live_orders()
            self._update_feature_metrics()
            self._update_execution_metrics()
            self._update_portfolio_metrics(self.portfolio)
            self._persist_health()
        except Exception as exc:
            self._record_incident("warning", "Runtime refresh failure", str(exc))
            LOGGER.exception("failed to refresh runtime state")

    def _maybe_cancel_stale_live_orders(self) -> None:
        if self.config.execution.mode != "live":
            return
        reports = self.execution.cancel_stale_orders(
            self.config.market.symbol,
            max_order_age_s=self.config.execution.resting_order_max_age_s,
        )
        if not reports:
            return

        for report in reports:
            payload = {
                "timestamp": utc_now().isoformat(),
                "symbol": report.symbol,
                "action": report.action,
                "success": report.success,
                "mode": "live",
                "execution": to_jsonable(report),
            }
            self.storage.record_execution(payload)
            if not report.success:
                self._record_incident("warning", "Stale order cancel failure", report.message)
        self.monitoring.set_metric("last_stale_cancel_at", utc_now().isoformat())

    def _current_portfolio(self, *, mark_price: float) -> PortfolioState:
        if self.config.execution.mode == "paper":
            return self.execution.reconcile(self.config.market.symbol, mark_price=mark_price, use_exchange=False)
        if self.config.execution.mode == "shadow" and not (self.config.secret_key or self.config.hyperliquid.account_address):
            return self.execution.reconcile(self.config.market.symbol, mark_price=mark_price, use_exchange=False)
        return self.execution.reconcile(self.config.market.symbol, use_exchange=True)

    def _update_portfolio_metrics(self, portfolio: PortfolioState) -> None:
        self.monitoring.set_metric("account_value_usd", portfolio.account_value_usd)
        self.monitoring.set_metric("position_size", portfolio.position_size)
        self.monitoring.set_metric("mark_price", portfolio.mark_price)
        self.monitoring.set_metric("daily_pnl_usd", portfolio.daily_pnl_usd)

    def _update_feature_metrics(self) -> None:
        self.monitoring.set_metric("feature_ready", self.features.is_ready(self.config.market.min_warm_price_points))
        self.monitoring.set_metric("warm_price_points", self.features.reference_price_count())
        self.monitoring.set_metric("last_market_data_at", to_jsonable(self.features.last_market_timestamp()))
        self.monitoring.set_metric("last_mid_price", self.features.last_mid_price())

    def _update_execution_metrics(self) -> None:
        if self.config.execution.mode not in {"live", "shadow"}:
            self.monitoring.set_metric("execution_state", "paper")
            self.monitoring.set_metric("execution_open_orders", 0)
            return
        state = self.execution.live_state(self.config.market.symbol)
        self.monitoring.set_metric("execution_state", state.status)
        self.monitoring.set_metric("execution_pending_reconcile", state.pending_reconcile)
        self.monitoring.set_metric("execution_open_orders", len(state.open_orders))
        self.monitoring.set_metric("execution_last_action", state.last_action)
        self.monitoring.set_metric("execution_blocked_reason", state.blocked_reason)

    def _record_incident(self, severity: str, title: str, details: str) -> None:
        incident = Incident(timestamp=utc_now(), severity=severity, title=title, details=details)
        self.monitoring.report_incident(incident)
        self.storage.record_incident(to_jsonable(incident))

    def shutdown(self) -> None:
        self.monitoring.shutdown()
        self.adapter.close()
