#!/usr/bin/env bash
# Test runner used by both the Docker image and CI.
#
# Runs the deterministic, offline, CPU-only checks required by P00.
# Failures in any stage stop the script (set -e).

set -euo pipefail

# The Docker build context intentionally excludes .git. Deployment package
# tests still require an explicit source identity, so use a fixed test-only
# object ID unless CI injects the real commit through DOUZERO_GIT_SHA.
export DOUZERO_GIT_SHA="${DOUZERO_GIT_SHA:-0000000000000000000000000000000000000000}"

echo "=== environment ==="
python -c "import sys, platform, torch, numpy; \
print('python', sys.version.split()[0]); \
print('platform', platform.platform()); \
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); \
print('numpy', numpy.__version__)"

echo
echo "=== compileall ==="
python -m compileall -q douzero tools benchmarks *.py

echo
echo "=== CLI --help ==="
python train.py --help >/dev/null
python train_v2.py --help >/dev/null
python train_coach.py --help >/dev/null
python evaluate.py --help >/dev/null
python evaluate_paired.py --help >/dev/null
python generate_eval_data.py --help >/dev/null
python train_belief.py --help >/dev/null
python evaluate_belief.py --help >/dev/null
python ingest_human_games.py --help >/dev/null
python validate_human_games.py --help >/dev/null
python pretrain_bc.py --help >/dev/null
python tools/package_model.py --help >/dev/null
python tools/prepare_p17_evaluation.py --help >/dev/null
python tools/rebuild_human_dataset.py --help >/dev/null
python tools/gpu_validation_probe.py --help >/dev/null
python tools/validate_amp_fallback.py --help >/dev/null
bash scripts/validate_gpu_training.sh --help >/dev/null

echo
echo "=== pytest ==="
python -m pytest "$@"
