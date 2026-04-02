from __future__ import annotations

from hyperliquid_bot.market_data import MarketDataService


def test_market_data_normalizes_order_and_fill_streams() -> None:
    service = MarketDataService("BTC")

    order_envelope = service.normalize(
        "orderUpdates",
        {"data": [{"coin": "BTC", "oid": 1, "status": "resting"}, {"coin": "ETH", "oid": 2, "status": "resting"}]},
    )
    fills_envelope = service.normalize(
        "userFills",
        {"data": {"fills": [{"coin": "BTC", "oid": 1, "side": "buy", "sz": "0.5"}]}},
    )
    events_envelope = service.normalize(
        "userEvents",
        {
            "data": {
                "fills": [{"coin": "BTC", "oid": 1, "side": "buy", "sz": "0.5"}],
                "nonUserCancel": [{"coin": "BTC", "oid": 1}],
            }
        },
    )

    assert len(order_envelope.order_updates) == 2
    assert len(fills_envelope.user_fills) == 1
    assert len(events_envelope.user_fills) == 1
    assert len(events_envelope.user_cancels) == 1
