"""Command-line interface for the QMT Bridge.

Subcommands:
  serve            Start the read-only HTTP service (default).
  cache            On-demand 落盘 of one symbol's quotes into the loader cache.
  manifest         Print the read-only capability manifest as JSON.
  token generate   Generate and persist a loopback API token.
  token show       Print the active token (only from an env override or vault).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from qmt_bridge.capabilities import manifest_payload


def _serve(args: argparse.Namespace) -> int:
    from qmt_bridge.server import run

    run()
    return 0


def _cache(args: argparse.Namespace) -> int:
    from qmt_bridge.config import Settings, load_settings
    from qmt_bridge.metadata import normalize_symbol
    from qmt_bridge.service import BridgeService
    from qmt_bridge.xtdata_client import XtdataClient

    settings: Settings = load_settings()
    symbol = normalize_symbol(args.symbol)
    service = BridgeService(XtdataClient(), settings=settings)
    result = service.daily(symbol, args.start, args.end, settings.adjust)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _manifest(_args: argparse.Namespace) -> int:
    print(json.dumps(manifest_payload(), ensure_ascii=False, indent=2))
    return 0


def _token_generate(_args: argparse.Namespace) -> int:
    from qmt_bridge.config import TOKEN_FIELD, generate_token
    from qmt_bridge.credentials import SecretVault

    vault = SecretVault()
    token = generate_token()
    vault.set(TOKEN_FIELD, token)
    print(token)
    return 0


def _token_show(_args: argparse.Namespace) -> int:
    from qmt_bridge.config import TOKEN_FIELD
    from qmt_bridge.credentials import SecretVault

    print(SecretVault().get(TOKEN_FIELD) or "")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmt-bridge", description="Read-only miniQMT data bridge.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="start the read-only HTTP service").set_defaults(func=_serve)
    sub.add_parser("manifest", help="print the capability manifest").set_defaults(func=_manifest)

    cache = sub.add_parser("cache", help="on-demand 落盘 one symbol's daily quotes")
    cache.add_argument("--symbol", required=True)
    cache.add_argument("--start", required=True)
    cache.add_argument("--end", required=True)
    cache.set_defaults(func=_cache)

    token = sub.add_parser("token", help="manage the loopback API token")
    token_sub = token.add_subparsers(dest="token_command")
    token_sub.add_parser("generate").set_defaults(func=_token_generate)
    token_sub.add_parser("show").set_defaults(func=_token_show)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
