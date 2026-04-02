from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .orchestrator import AutonomousBot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperliquid autonomous bot")
    parser.add_argument("--config", default="config/bot.toml", help="Path to bot TOML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the bot loop")
    run_parser.add_argument("--mode", choices=["paper", "shadow", "live"], default=None)

    subparsers.add_parser("train", help="Train and evaluate a new model")
    backtest_parser = subparsers.add_parser("backtest", help="Run fee-aware backtest from captured feature data")
    backtest_parser.add_argument("--model-version", default=None, help="Specific model version to evaluate")
    subparsers.add_parser("health", help="Print current health snapshot")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))
    if args.command == "run" and args.mode:
        config.execution.mode = args.mode
    configure_logging(config.monitoring.log_level)

    bot = AutonomousBot(config)
    if args.command == "run":
        bot.run()
        return
    if args.command == "train":
        print(json.dumps(bot.train(), indent=2, sort_keys=True))
        return
    if args.command == "backtest":
        print(json.dumps(bot.backtest(model_version=args.model_version), indent=2, sort_keys=True))
        return
    if args.command == "health":
        print(json.dumps(bot.health(), indent=2, sort_keys=True))
        return

    parser.error("unsupported command")
