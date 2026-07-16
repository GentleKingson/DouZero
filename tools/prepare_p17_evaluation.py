#!/usr/bin/env python3
"""Validate and collate P17 evaluation inputs into the fixed artifact layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from douzero.evaluation.checkpoint_inputs import (
    CheckpointIdentityError,
    require_explicit_matrix_checkpoint_digests,
)
from douzero.evaluation.p17 import (
    ABLATION_NAMES,
    P17MatrixError,
    empty_matrix,
    load_result,
    normalize_matrix,
    write_p17_artifacts,
)
from douzero.evaluation.provenance import (
    AttestedEvaluationInput,
    AttestationPolicy,
    ProvenanceError,
)


def _assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ABLATION=/path/to/result.json")
    name, path = value.split("=", 1)
    if name not in ABLATION_NAMES or not path:
        raise argparse.ArgumentTypeError("unknown ablation name or empty path")
    return name, path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", help="P17 model matrix JSON")
    parser.add_argument("--write-matrix-template", help="write an unavailable template and exit")
    parser.add_argument("--cardplay-result", help="P15 cardplay_only result JSON")
    parser.add_argument("--full-game-result", help="P15 full_game result JSON")
    parser.add_argument(
        "--ablation-result", action="append", type=_assignment, default=[]
    )
    parser.add_argument(
        "--ablation-attestation", action="append", type=_assignment, default=[]
    )
    parser.add_argument(
        "--expected-evaluator-git-sha",
        action="append",
        default=[],
        help=(
            "approved full evaluator Git SHA; repeat for an explicit cross-version "
            "allowlist"
        ),
    )
    parser.add_argument(
        "--expected-cardplay-deal-set-id",
        help="pre-approved cardplay_only deal-set SHA-256",
    )
    parser.add_argument(
        "--expected-full-game-deal-set-id",
        help="pre-approved full_game deal-set SHA-256",
    )
    parser.add_argument(
        "--approved-cardplay-eval-data",
        help="strict formal JSON cardplay deals used for deterministic replay",
    )
    parser.add_argument(
        "--approved-full-game-eval-data",
        help="strict formal JSON full-game deals used for deterministic replay",
    )
    parser.add_argument("--cardplay-attestation", help="detached GitHub attestation bundle")
    parser.add_argument("--full-game-attestation", help="detached GitHub attestation bundle")
    parser.add_argument("--attestation-repository", help="exact owner/repository signer scope")
    parser.add_argument(
        "--attestation-signer-workflow", help="exact protected signer workflow path"
    )
    parser.add_argument(
        "--attestation-signer-digest", help="full Git OID of the signer workflow"
    )
    parser.add_argument(
        "--attestation-source-ref", help="fully qualified evaluated source ref"
    )
    parser.add_argument("--output", default="artifacts/evaluation/p17")
    args = parser.parse_args(argv)

    if args.write_matrix_template:
        path = Path(args.write_matrix_template)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(empty_matrix(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if not args.matrix:
        parser.error("--matrix is required unless --write-matrix-template is used")
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    try:
        require_explicit_matrix_checkpoint_digests(matrix, kind="p17")
    except CheckpointIdentityError as exc:
        parser.error(f"P17 matrix checkpoint identity is invalid: {exc}")
    normalized_matrix = normalize_matrix(matrix)
    ablation_protocols = {
        name: normalized_matrix["ablations"][name]["protocol"]
        for name, _path in args.ablation_result
    }
    if (
        args.cardplay_result
        or args.full_game_result
        or args.ablation_result
    ) and not args.expected_evaluator_git_sha:
        parser.error(
            "--expected-evaluator-git-sha is required when collating results"
        )
    if args.cardplay_result and not args.cardplay_attestation:
        parser.error("--cardplay-attestation is required for a cardplay result")
    if args.full_game_result and not args.full_game_attestation:
        parser.error("--full-game-attestation is required for a full-game result")
    ablation_attestations = dict(args.ablation_attestation)
    if set(ablation_attestations) != {name for name, _path in args.ablation_result}:
        parser.error(
            "every --ablation-result requires exactly one matching "
            "--ablation-attestation"
        )
    if (
        args.cardplay_result or args.full_game_result or args.ablation_result
    ) and not all((
        args.attestation_repository,
        args.attestation_signer_workflow,
        args.attestation_signer_digest,
        args.attestation_source_ref,
    )):
        parser.error(
            "formal results require repository, signer workflow/digest, and source ref"
        )
    needs_cardplay_set = bool(args.cardplay_result) or any(
        ablation_protocols[name] == "cardplay_only"
        for name, _path in args.ablation_result
    )
    needs_full_game_set = bool(args.full_game_result) or any(
        ablation_protocols[name] == "full_game"
        for name, _path in args.ablation_result
    )
    if needs_cardplay_set and not args.expected_cardplay_deal_set_id:
        parser.error(
            "--expected-cardplay-deal-set-id is required for cardplay results"
        )
    if needs_full_game_set and not args.expected_full_game_deal_set_id:
        parser.error(
            "--expected-full-game-deal-set-id is required for full-game results"
        )
    if needs_cardplay_set and not args.approved_cardplay_eval_data:
        parser.error(
            "--approved-cardplay-eval-data is required for deterministic replay"
        )
    if needs_full_game_set and not args.approved_full_game_eval_data:
        parser.error(
            "--approved-full-game-eval-data is required for deterministic replay"
        )

    from douzero.env.rules import RuleSet
    from douzero.evaluation.formal_eval_data import (
        FormalEvalDataError,
        load_formal_eval_data,
    )

    try:
        approved_cardplay_deals = (
            tuple(load_formal_eval_data(
                args.approved_cardplay_eval_data,
                expected_mode="cardplay_only",
                expected_ruleset=RuleSet.legacy(),
            ))
            if args.approved_cardplay_eval_data else None
        )
        approved_full_game_deals = (
            tuple(load_formal_eval_data(
                args.approved_full_game_eval_data,
                expected_mode="full_game",
                expected_ruleset=RuleSet.standard(),
            ))
            if args.approved_full_game_eval_data else None
        )
    except (FormalEvalDataError, OSError) as exc:
        parser.error(f"approved formal evaluation data is invalid: {exc}")

    approved_shas = set(args.expected_evaluator_git_sha)

    def attested_input(path: str, bundle: str, mode: str):
        raw = load_result(path, mode)
        runtime = raw.get("runtime_identity", {})
        source_sha = runtime.get("source_git_sha")
        if source_sha not in approved_shas:
            parser.error(f"{path} source_git_sha is not approved")
        artifact_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        try:
            return AttestedEvaluationInput(
                result_path=path,
                bundle_path=bundle,
                policy=AttestationPolicy(
                    repository=args.attestation_repository,
                    signer_workflow=args.attestation_signer_workflow,
                    signer_digest=args.attestation_signer_digest,
                    source_digest=source_sha,
                    source_ref=args.attestation_source_ref,
                    artifact_sha256=artifact_sha,
                ),
            )
        except ProvenanceError as exc:
            parser.error(f"invalid attestation policy: {exc}")

    cardplay = (
        attested_input(
            args.cardplay_result, args.cardplay_attestation, "cardplay_only"
        )
        if args.cardplay_result else None
    )
    full_game = (
        attested_input(
            args.full_game_result, args.full_game_attestation, "full_game"
        )
        if args.full_game_result else None
    )
    ablations = {
        name: attested_input(path, ablation_attestations[name], ablation_protocols[name])
        for name, path in args.ablation_result
    }
    try:
        paths = write_p17_artifacts(
            args.output,
            matrix=matrix,
            cardplay_result=cardplay,
            full_game_result=full_game,
            ablation_results=ablations,
            expected_evaluator_git_shas=args.expected_evaluator_git_sha,
            expected_cardplay_deal_set_id=args.expected_cardplay_deal_set_id,
            expected_full_game_deal_set_id=args.expected_full_game_deal_set_id,
            approved_cardplay_deals=approved_cardplay_deals,
            approved_full_game_deals=approved_full_game_deals,
        )
    except P17MatrixError as exc:
        parser.error(str(exc))
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
