#!/usr/bin/env bash
# P17 target-hardware validation. Artifacts are local and ignored by Git.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${DOUZERO_GPU_VALIDATION_OUTPUT:-${ROOT}/artifacts/gpu-validation}"
PROBE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --probe-only)
      PROBE_ONLY=1
      shift
      ;;
    --help|-h)
      printf '%s\n' \
        "usage: scripts/validate_gpu_training.sh [--output DIR] [--probe-only]" \
        "" \
        "Environment overrides:" \
        "  DOUZERO_PYTHON                 Python executable" \
        "  DOUZERO_GPU_EPISODES           Episodes per smoke (default: 8)" \
        "  DOUZERO_GPU_OPTIMIZER_STEPS    Learner steps per smoke (default: 2)" \
        "  DOUZERO_GPU_BATCH_SIZE         Replay/bidding batch size (default: 4)" \
        "  DOUZERO_GPU_BUFFER_CAPACITY    Replay capacity (default: 64)" \
        "  DOUZERO_GPU_BELIEF_CONFIG      Belief-enabled standard V2 YAML" \
        "  DOUZERO_GPU_BELIEF_CHECKPOINT  Compatible standard belief checkpoint"
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${DOUZERO_PYTHON:-}" ]]; then
  PYTHON="${DOUZERO_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  printf 'No Python executable found. Set DOUZERO_PYTHON.\n' >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" "${ROOT}/tools/gpu_validation_probe.py" \
  --output "${OUTPUT_DIR}/environment.json" >/dev/null

write_status() {
  local name="$1"
  local status="$2"
  local reason="$3"
  "${PYTHON}" -c \
    'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); payload={"schema_version":"p17-gpu-run-v1","status":sys.argv[2],"reason":sys.argv[3],"metrics":{"peak_memory_mib":None,"peak_reserved_memory_mib":None,"samples_per_second":None,"decisions_per_second":None,"learner_steps_per_second":None}}; t=p.with_name("."+p.name+".tmp"); t.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); t.replace(p)' \
    "${OUTPUT_DIR}/${name}.json" "${status}" "${reason}"
}

CUDA_COUNT="$(${PYTHON} -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if [[ "${PROBE_ONLY}" -eq 1 ]]; then
  printf 'GPU environment probe written to %s/environment.json\n' "${OUTPUT_DIR}"
  exit 0
fi

if [[ "${CUDA_COUNT}" -lt 1 ]]; then
  for name in single_gpu_fp32 single_gpu_fp16 single_gpu_bf16 \
    amp_nonfinite_fallback belief_frozen belief_joint checkpoint_resume; do
    write_status "${name}" "not_run" "CUDA device unavailable"
  done
  write_status "ddp_2gpu" "blocked_implementation" \
    "standard learned-bidding DDP intentionally fails closed"
  cat >"${OUTPUT_DIR}/summary.md" <<'EOF'
# P17 GPU Validation

Status: **NOT RUN**

Reason: PyTorch did not detect a CUDA device. FP32, FP16, BF16, NCCL DDP,
checkpoint resume, peak memory, throughput, and torch.compile remain pending
external target-hardware validation.

Independent implementation blocker: standard learned-bidding training does
not support DDP and fails closed before collection. `ddp_2gpu.json` records
that blocker rather than presenting the missing hardware as the only reason.
EOF
  printf 'CUDA unavailable; empirical GPU validation was not run.\n' >&2
  exit 3
fi

EPISODES="${DOUZERO_GPU_EPISODES:-8}"
OPTIMIZER_STEPS="${DOUZERO_GPU_OPTIMIZER_STEPS:-2}"
BATCH_SIZE="${DOUZERO_GPU_BATCH_SIZE:-4}"
BUFFER_CAPACITY="${DOUZERO_GPU_BUFFER_CAPACITY:-64}"

run_case() {
  local name="$1"
  shift
  local started ended status report_status
  write_status "${name}" "running" "validation command is in progress"
  started="$(date +%s)"
  set +e
  "$@" >"${OUTPUT_DIR}/${name}.log" 2>&1
  status=$?
  set -e
  ended="$(date +%s)"
  if [[ "${status}" -eq 0 ]]; then
    report_status="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("status",""))' "${OUTPUT_DIR}/${name}.json")"
    if [[ "${report_status}" != "passed" ]]; then
      status=70
    fi
  fi
  "${PYTHON}" -c \
    'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); rc=int(sys.argv[2]); payload=json.loads(p.read_text(encoding="utf-8")); payload["status"]="passed" if rc == 0 else "failed"; payload["exit_code"]=rc; payload["command_wall_seconds"]=int(sys.argv[3]); payload.pop("reason",None) if rc == 0 else payload.update(reason="validation command failed; inspect the adjacent local log"); t=p.with_name("."+p.name+".tmp"); t.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); t.replace(p)' \
    "${OUTPUT_DIR}/${name}.json" "${status}" "$((ended - started))"
  return "${status}"
}

COMMON=(
  "${PYTHON}" "${ROOT}/train_v2.py"
  --config "${ROOT}/configs/standard_v2.yaml"
  --episodes "${EPISODES}"
  --optimizer_steps "${OPTIMIZER_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --buffer_capacity "${BUFFER_CAPACITY}"
  --device cuda
)

overall=0
FP32_CHECKPOINT="${OUTPUT_DIR}/single_gpu_fp32.checkpoint.pt"
: >"${FP32_CHECKPOINT}"
run_case single_gpu_fp32 "${COMMON[@]}" --no-amp_enabled \
  --checkpoint_path "${FP32_CHECKPOINT}" \
  --metrics_path "${OUTPUT_DIR}/single_gpu_fp32.json" || overall=1
run_case single_gpu_fp16 "${COMMON[@]}" --amp_enabled --amp_dtype float16 \
  --checkpoint_path "${OUTPUT_DIR}/single_gpu_fp16.checkpoint.pt" \
  --metrics_path "${OUTPUT_DIR}/single_gpu_fp16.json" || overall=1

run_case amp_nonfinite_fallback \
  "${PYTHON}" "${ROOT}/tools/validate_amp_fallback.py" \
  --output "${OUTPUT_DIR}/amp_nonfinite_fallback.json" \
  --device cuda --dtype float16 || overall=1

BF16_SUPPORTED="$(${PYTHON} -c 'import torch; print(int(torch.cuda.is_bf16_supported()))')"
if [[ "${BF16_SUPPORTED}" -eq 1 ]]; then
  run_case single_gpu_bf16 "${COMMON[@]}" --amp_enabled --amp_dtype bfloat16 \
    --checkpoint_path "${OUTPUT_DIR}/single_gpu_bf16.checkpoint.pt" \
    --metrics_path "${OUTPUT_DIR}/single_gpu_bf16.json" || overall=1
else
  write_status "single_gpu_bf16" "unsupported_hardware" \
    "attached CUDA device does not support BF16"
fi

write_status "ddp_2gpu" "blocked_implementation" \
  "standard learned-bidding DDP intentionally fails closed"
overall=1

BELIEF_CONFIG="${DOUZERO_GPU_BELIEF_CONFIG:-}"
BELIEF_CHECKPOINT="${DOUZERO_GPU_BELIEF_CHECKPOINT:-}"
if [[ -n "${BELIEF_CONFIG}" && -n "${BELIEF_CHECKPOINT}" ]]; then
  if [[ ! -f "${BELIEF_CONFIG}" || ! -f "${BELIEF_CHECKPOINT}" ]]; then
    write_status "belief_frozen" "failed_precondition" \
      "configured belief config or checkpoint is unavailable"
    write_status "belief_joint" "failed_precondition" \
      "configured belief config or checkpoint is unavailable"
    overall=1
  else
    BELIEF_COMMON=(
      "${PYTHON}" "${ROOT}/train_v2.py"
      --config "${BELIEF_CONFIG}"
      --belief_checkpoint "${BELIEF_CHECKPOINT}"
      --episodes "${EPISODES}"
      --optimizer_steps "${OPTIMIZER_STEPS}"
      --batch_size "${BATCH_SIZE}"
      --buffer_capacity "${BUFFER_CAPACITY}"
      --device cuda
      --no-amp_enabled
    )
    run_case belief_frozen "${BELIEF_COMMON[@]}" \
      --belief_training_mode frozen \
      --checkpoint_path "${OUTPUT_DIR}/belief_frozen.checkpoint.pt" \
      --metrics_path "${OUTPUT_DIR}/belief_frozen.json" || overall=1
    run_case belief_joint "${BELIEF_COMMON[@]}" \
      --belief_training_mode joint \
      --checkpoint_path "${OUTPUT_DIR}/belief_joint.checkpoint.pt" \
      --metrics_path "${OUTPUT_DIR}/belief_joint.json" || overall=1
  fi
else
  write_status "belief_frozen" "not_run" \
    "compatible belief config and checkpoint were not supplied"
  write_status "belief_joint" "not_run" \
    "compatible belief config and checkpoint were not supplied"
  overall=1
fi

if [[ -s "${FP32_CHECKPOINT}" ]]; then
  run_case checkpoint_resume "${COMMON[@]}" --no-amp_enabled \
    --resume_checkpoint "${FP32_CHECKPOINT}" \
    --checkpoint_path "${OUTPUT_DIR}/checkpoint_resume.checkpoint.pt" \
    --metrics_path "${OUTPUT_DIR}/checkpoint_resume.json" || overall=1
else
  write_status "checkpoint_resume" "not_run" \
    "FP32 checkpoint was not produced by this invocation"
  overall=1
fi

cat >"${OUTPUT_DIR}/summary.md" <<'EOF'
# P17 GPU Validation

See the adjacent JSON files and per-case logs. A passing short run establishes
only execution compatibility; it does not establish throughput, stability, or
playing strength. Rates in the training reports use the measured `trainer.train`
wall time; command wall time additionally includes startup and checkpoint I/O.

Standard learned-bidding DDP is an explicit implementation blocker and is not
attempted. Frozen and joint belief cases require both
`DOUZERO_GPU_BELIEF_CONFIG` and `DOUZERO_GPU_BELIEF_CHECKPOINT`. The script
returns nonzero while any required case is failed, unavailable, or blocked.
EOF

exit "${overall}"
