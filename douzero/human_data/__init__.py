"""P08: human-gameplay data pipeline and canonical record format.

This package implements the AGENTS.md "Human-game data and strategy priors"
rules for *offline, lawfully-obtained* human DouDizhu game records. It does
NOT contain any web-scraping, account-automation, anti-detection, or
platform-ToS-bypass code (those are explicitly out of scope and prohibited).

Public modules:

- :mod:`schema`    — the canonical :class:`HumanGameRecord` (JSONL format) and
                      JSON-Schema validation.
- :mod:`adapters`  — the :class:`Adapter` protocol for converting external
                      platform formats into canonical records (no platform
                      format is hard-coded in the training code).
- :mod:`synthetic` — a deterministic synthetic game generator used for tests
                      and pipeline smoke when no ``<HUMAN_DATA_PATH>`` exists.
- :mod:`validate`  — full replay validation through the rule engine + quarantine.
- :mod:`ingest`    — parse / anonymize / de-duplicate external records.
- :mod:`split`     — by-``game_id`` dataset splitting with no overlap.
- :mod:`sample`    — build listwise BC samples (public obs + human action index).
- :mod:`weights`   — sample-weight computation (skill, integrity, rule match).

Imperfect-information boundary
------------------------------
The raw :class:`HumanGameRecord` carries ``initial_hands`` (the true deal) and
the recorded human actions. These are **privileged training-only data**,
analogous to :class:`~douzero.observation.privileged.PrivilegedObservation`. The
BC *student* model only ever receives the public
:class:`~douzero.observation.encode_v2.ObservationV2` produced by replaying the
record; the recorded human action becomes the ``human_action_index`` label,
which is carried in a separately-stamped :class:`~douzero.human_data.sample.BCSample`
and never reaches the deployment ``DeepAgentV2.act``.
"""

from __future__ import annotations

from .schema import (
    CANONICAL_FORMAT_VERSION,
    HUMAN_RECORD_KIND,
    HUMAN_RECORD_SCHEMA_VERSION,
    ACTION_ROLES,
    FINAL_RESULT_KEYS,
    HumanGameRecord,
    JsonlLineResult,
    RecordValidationError,
    iter_jsonl_resilient,
    record_from_dict,
    record_from_jsonl_line,
    read_jsonl,
    write_jsonl,
)
from .validate import assert_legacy_ruleset

__all__ = [
    "CANONICAL_FORMAT_VERSION",
    "HUMAN_RECORD_KIND",
    "HUMAN_RECORD_SCHEMA_VERSION",
    "ACTION_ROLES",
    "FINAL_RESULT_KEYS",
    "HumanGameRecord",
    "JsonlLineResult",
    "RecordValidationError",
    "iter_jsonl_resilient",
    "record_from_dict",
    "record_from_jsonl_line",
    "read_jsonl",
    "write_jsonl",
    "assert_legacy_ruleset",
]
