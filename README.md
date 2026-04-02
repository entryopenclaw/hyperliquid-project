# Hyperliquid Autonomous Bot

This repository scaffolds an autonomous futures trading bot for **Hyperliquid testnet** using the official [`hyperliquid-python-sdk`](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) as the exchange layer.

The bot is designed around:

- one market first: `BTC`
- strict risk controls and kill switches
- offline self-learning with gated model promotion
- clear separation between exchange, strategy, risk, execution, training, and monitoring

## What is implemented

- configuration loading from TOML plus environment secrets
- wrapped Hyperliquid exchange adapter
- market data normalization, startup snapshot bootstrapping, and feature pipeline warmup gating
- heuristic/learned signal model support
- policy, risk, execution state reconciliation, monitoring, model registry, and training services
- CLI entrypoints for live loop, training, backtesting, and health inspection
- unit-test coverage for config, features, risk/policy, and model promotion rules

## Quick start

1. Install Python 3.12.
2. Create a virtual environment.
3. Install the package:

```bash
pip install -e .[dev]
```

4. Copy the config template:

```bash
cp config/bot.example.toml config/bot.toml
```

5. Fill `.env` with your Hyperliquid testnet wallet values.
6. Run the bot in paper or shadow mode first:

```bash
hyperliquid-autobot run --config config/bot.toml --mode paper
```

## Recommended rollout

1. `paper`: consume live testnet data with a local simulated futures portfolio, modeled PnL, deterministic latency penalties, partial-fill logic, and startup snapshot warmup, place no exchange orders.
2. `shadow`: validate execution intent and exchange/account reconciliation without sending orders.
3. `live`: allow small **Hyperliquid testnet** orders after stable paper and shadow behavior.
4. `train`: retrain nightly and promote only when validation gates pass.
5. `backtest`: generate a fee-aware report from captured feature data before trusting execution changes.

Live and shadow modes now use a stricter execution guard: the bot reconciles exchange positions and open orders into an internal state machine, consumes `orderUpdates` and `userFills` websocket events to clear ambiguity faster, and refuses to send a new live order while reconciliation is pending or old exchange orders are still working.

## Notes

- The current implementation is **testnet-only by default**.
- Profitability is not guaranteed. The system is structured to discover, validate, and protect an edge rather than assume one exists.
- The machine used for this scaffold did not have Python available in `PATH`, so the code could not be executed locally during generation.
