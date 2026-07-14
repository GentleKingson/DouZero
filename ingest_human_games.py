"""Ingest external human-game records into canonical JSONL (P08).

Two modes:

1. **Synthetic** (``--synthetic``): generate deterministic random-self-play
   records via :mod:`douzero.human_data.synthetic` and write canonical JSONL.
   This is the smoke / CI path when no ``<HUMAN_DATA_PATH>`` exists. No network
   or downloaded weights required.

2. **External** (``--input`` + ``--adapter``): read an external-format JSONL
   (one raw payload per line), convert each via the named adapter, sanitize
   metadata, de-duplicate by ``game_id``, and write canonical JSONL sorted by
   ``game_id``. The adapter is a dotted import path to a callable
   ``adapter(raw: Mapping) -> HumanGameRecord``; no platform format is
   hard-coded here.

Ingest deliberately does NOT validate game legality — run
``validate_human_games.py`` on the output to quarantine invalid games.

Example (CPU smoke)::

    python ingest_human_games.py --synthetic --num_synthetic 16 \
        --output /tmp/games.jsonl

    python ingest_human_games.py --input /tmp/raw.jsonl \
        --adapter mypkg.adapters.PlatformAAdapter \
        --output /tmp/games.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from typing import Any, Iterator, Mapping, Sequence

from douzero.human_data.adapters import Adapter
from douzero.human_data.ingest import ingest_to_jsonl
from douzero.human_data.schema import write_jsonl
from douzero.human_data.synthetic import generate_synthetic_records


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingest_human_games",
        description="Convert external human-game records (or synthetic games) "
                    "into canonical JSONL.",
    )
    p.add_argument("--input", default="",
                   help="external JSONL input (one raw payload per line); "
                        "omitted when --synthetic is set")
    p.add_argument("--output", required=True,
                   help="canonical JSONL output path")
    p.add_argument("--adapter", default="",
                   help="dotted import path to an Adapter callable "
                        "(raw -> HumanGameRecord); required with --input")
    p.add_argument("--synthetic", action="store_true",
                   help="generate deterministic synthetic records (no real data)")
    p.add_argument("--num_synthetic", type=int, default=16)
    p.add_argument("--synthetic_seed", type=int, default=0)
    return p.parse_args(argv)


def _load_adapter(dotted: str) -> Adapter:
    """Import an adapter callable from a ``module.attr`` dotted path."""
    if not dotted:
        raise SystemExit(
            "--adapter is required with --input (a dotted path to a callable "
            "raw -> HumanGameRecord)"
        )
    if "." not in dotted:
        raise SystemExit(f"--adapter {dotted!r} must be a 'module.attr' path")
    module_name, attr = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import adapter module {module_name!r}: {exc}")
    try:
        adapter = getattr(module, attr)
    except AttributeError as exc:
        raise SystemExit(
            f"adapter {attr!r} not found in module {module_name!r}: {exc}"
        )
    if inspect.isclass(adapter):
        try:
            adapter = adapter()
        except TypeError as exc:
            raise SystemExit(
                "adapter classes must have a zero-argument constructor; "
                "use an adapter function that closes over configuration"
            ) from exc
    if not callable(adapter):
        raise SystemExit(f"adapter {dotted!r} is not callable")
    return adapter  # type: ignore[return-value]


def _iter_raw_external(path: str) -> Iterator[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}")
            yield payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.synthetic:
        records = list(
            generate_synthetic_records(
                num_games=args.num_synthetic,
                base_seed=args.synthetic_seed,
            )
        )
        n = write_jsonl(records, args.output)
        print(
            f"[ingest_human_games] synthetic: wrote {n} canonical records to "
            f"{args.output}",
            file=sys.stderr,
        )
        return 0

    if not args.input:
        raise SystemExit("either --synthetic or --input is required")
    adapter = _load_adapter(args.adapter)
    n = ingest_to_jsonl(_iter_raw_external(args.input), adapter, args.output)
    print(
        f"[ingest_human_games] ingested {n} canonical records to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
