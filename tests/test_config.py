from __future__ import annotations

from pathlib import Path

import pytest

from hyperliquid_bot.config import load_config


def test_load_config_expands_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xabc")
    config_path = tmp_path / "bot.toml"
    config_path.write_text(
        """
        [hyperliquid]
        account_address = "${HL_ACCOUNT_ADDRESS}"

        [execution]
        mode = "paper"
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.hyperliquid.account_address == "0xabc"
    assert config.execution.mode == "paper"
    assert config.market.symbol == "BTC"


def test_invalid_mode_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "bot.toml"
    config_path.write_text(
        """
        [execution]
        mode = "broken"
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_path)


def test_invalid_retrain_interval_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "bot.toml"
    config_path.write_text(
        """
        [training]
        retrain_interval_hours = 0
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_path)


def test_invalid_paper_partial_fill_fraction_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "bot.toml"
    config_path.write_text(
        """
        [execution]
        paper_partial_fill_min_fraction = 0
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_path)


def test_startup_lookback_must_cover_warm_points(tmp_path: Path) -> None:
    config_path = tmp_path / "bot.toml"
    config_path.write_text(
        """
        [market]
        startup_candle_lookback = 10
        min_warm_price_points = 20
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_path)
