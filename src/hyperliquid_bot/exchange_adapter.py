from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from .config import BotConfig
from .models import ExecutionReport, PortfolioState
from .utils import utc_now

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AdapterClients:
    info: Any
    stream: Any
    exchange: Any
    constants: Any


class HyperliquidAdapter:
    """Thin wrapper that isolates all Hyperliquid SDK usage."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.clients: AdapterClients | None = None
        self.account_address = config.hyperliquid.account_address

    def connect(self) -> None:
        if self.clients is not None:
            return

        if not self.config.secret_key:
            raise RuntimeError(
                f"Environment variable {self.config.hyperliquid.secret_key_env} is required to create the exchange client."
            )

        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            from hyperliquid.utils import constants
        except ImportError as exc:
            raise RuntimeError(
                "Required dependencies are missing. Install with `pip install -e .[dev]`."
            ) from exc

        wallet = Account.from_key(self.config.secret_key)
        self.account_address = self.account_address or wallet.address

        if self.config.hyperliquid.api_url:
            base_url = self.config.hyperliquid.api_url
        elif self.config.hyperliquid.network == "testnet":
            base_url = constants.TESTNET_API_URL
        else:
            base_url = constants.MAINNET_API_URL

        info = Info(base_url, skip_ws=True)
        stream = Info(base_url, skip_ws=False)
        exchange = Exchange(
            wallet,
            base_url=base_url,
            account_address=self.account_address or None,
            vault_address=self.config.hyperliquid.vault_address or None,
        )
        self.clients = AdapterClients(info=info, stream=stream, exchange=exchange, constants=constants)

    def close(self) -> None:
        if self.clients is None:
            return
        try:
            self.clients.stream.disconnect_websocket()
        except Exception:
            LOGGER.debug("stream disconnect failed", exc_info=True)
        self.clients = None

    @property
    def _info(self) -> Any:
        self.connect()
        assert self.clients is not None
        return self.clients.info

    @property
    def _stream(self) -> Any:
        self.connect()
        assert self.clients is not None
        return self.clients.stream

    @property
    def _exchange(self) -> Any:
        self.connect()
        assert self.clients is not None
        return self.clients.exchange

    def get_meta(self) -> Any:
        return self._info.meta_and_asset_ctxs()

    def get_user_state(self) -> dict[str, Any]:
        return self._info.user_state(self.account_address)

    def get_open_orders(self) -> list[dict[str, Any]]:
        return self._info.open_orders(self.account_address)

    def query_order(self, oid: int) -> dict[str, Any]:
        return self._info.query_order_by_oid(self.account_address, oid)

    def get_all_mids(self) -> dict[str, str]:
        return self._info.all_mids()

    def get_l2_snapshot(self, symbol: str) -> dict[str, Any]:
        return self._info.l2_snapshot(symbol)

    def get_candles(self, symbol: str, interval: str, start_time: int, end_time: int) -> list[dict[str, Any]]:
        return self._info.candles_snapshot(symbol, interval, start_time, end_time)

    def get_funding_history(self, symbol: str, start_time: int, end_time: int | None = None) -> list[dict[str, Any]]:
        return self._info.funding_history(symbol, start_time, end_time)

    def get_user_fills(self) -> list[dict[str, Any]]:
        return self._info.user_fills(self.account_address)

    def subscribe_market_streams(self, symbol: str, callback: Callable[[str, Any], None]) -> None:
        subscriptions = [
            {"type": "l2Book", "coin": symbol},
            {"type": "trades", "coin": symbol},
            {"type": "candle", "coin": symbol, "interval": self.config.market.candle_interval},
            {"type": "allMids"},
            {"type": "activeAssetCtx", "coin": symbol},
            {"type": "orderUpdates", "user": self.account_address},
            {"type": "userFills", "user": self.account_address},
            {"type": "userEvents", "user": self.account_address},
        ]
        for subscription in subscriptions:
            sub_type = subscription["type"]

            def handler(message: Any, stream_type: str = sub_type) -> None:
                callback(stream_type, message)

            self._stream.subscribe(subscription, handler)

    def place_limit_order(self, symbol: str, side: str, size: float, limit_price: float, reduce_only: bool) -> dict[str, Any]:
        is_buy = side.lower() == "buy"
        return self._exchange.order(
            symbol,
            is_buy,
            size,
            limit_price,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=reduce_only,
        )

    def place_ioc_order(self, symbol: str, side: str, size: float, px: float | None = None) -> dict[str, Any]:
        is_buy = side.lower() == "buy"
        slippage = self.config.execution.ioc_slippage_bps / 10_000.0
        return self._exchange.market_open(symbol, is_buy, size, px=px, slippage=slippage)

    def close_position(self, symbol: str) -> dict[str, Any]:
        slippage = self.config.execution.ioc_slippage_bps / 10_000.0
        return self._exchange.market_close(symbol, slippage=slippage)

    def cancel(self, symbol: str, oid: int) -> dict[str, Any]:
        return self._exchange.cancel(symbol, oid)

    def schedule_cancel_all(self) -> dict[str, Any]:
        cancel_at = utc_now() + timedelta(seconds=self.config.execution.schedule_cancel_after_s)
        return self._exchange.schedule_cancel(int(cancel_at.timestamp() * 1000))

    def update_leverage(self, symbol: str, leverage: int, *, is_cross: bool = True) -> dict[str, Any]:
        return self._exchange.update_leverage(leverage, symbol, is_cross=is_cross)

    def build_portfolio_state(self, symbol: str) -> PortfolioState:
        raw_state = self.get_user_state()
        mark_price = 0.0
        position_size = 0.0
        entry_price = 0.0
        leverage = 0.0
        unrealized = 0.0

        for item in raw_state.get("assetPositions", []):
            position = item.get("position", {})
            if position.get("coin") != symbol:
                continue
            position_size = float(position.get("szi", 0.0))
            entry_price = float(position.get("entryPx", 0.0) or 0.0)
            leverage_value = position.get("leverage", {}).get("value", 0.0)
            leverage = float(leverage_value or 0.0)
            unrealized = float(position.get("unrealizedPnl", 0.0) or 0.0)
            position_value = float(position.get("positionValue", 0.0) or 0.0)
            mark_price = abs(position_value / position_size) if position_size else 0.0

        margin = raw_state.get("marginSummary", {})
        account_value = float(margin.get("accountValue", 0.0) or 0.0)
        open_orders = len(self.get_open_orders())
        return PortfolioState(
            timestamp=utc_now(),
            symbol=symbol,
            account_value_usd=account_value,
            position_size=position_size,
            entry_price=entry_price,
            mark_price=mark_price,
            leverage=leverage,
            unrealized_pnl_usd=unrealized,
            realized_pnl_usd=0.0,
            daily_pnl_usd=0.0,
            open_orders=open_orders,
        )

    def safe_report(self, symbol: str, action: str, response: dict[str, Any], success: bool = True) -> ExecutionReport:
        parsed_success, message = self._parse_response(response)
        return ExecutionReport(
            timestamp=utc_now(),
            symbol=symbol,
            action=action,
            success=success and parsed_success,
            message=message,
            response=response,
        )

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> tuple[bool, str]:
        status = str(response.get("status", "ok")).lower()
        if status not in {"ok", "success"}:
            return False, status

        statuses: list[Any] = []
        response_payload = response.get("response", {})
        if isinstance(response_payload, dict):
            data = response_payload.get("data", {})
            if isinstance(data, dict):
                statuses = data.get("statuses", [])

        for item in statuses:
            if not isinstance(item, dict):
                continue
            if "error" in item:
                return False, str(item["error"])
            if "err" in item:
                return False, str(item["err"])

        if statuses:
            return True, "accepted"
        return True, "ok"
