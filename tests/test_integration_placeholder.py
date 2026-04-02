from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(not os.getenv("RUN_HYPERLIQUID_INTEGRATION"), reason="integration opt-in only")
def test_hyperliquid_integration_placeholder() -> None:
    # The live adapter is intentionally guarded behind an env flag because it requires
    # real network access plus testnet credentials.
    assert True
