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
