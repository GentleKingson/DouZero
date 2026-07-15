#!/usr/bin/env bash
set -euo pipefail

bash .docker/run_tests.sh
python -m compileall -q tools/package_model.py tools/prepare_p17_evaluation.py tools/rebuild_human_dataset.py tools/gpu_validation_probe.py tools/validate_amp_fallback.py
python tools/package_model.py --help >/dev/null
python tools/prepare_p17_evaluation.py --help >/dev/null
python tools/rebuild_human_dataset.py --help >/dev/null
python tools/validate_amp_fallback.py --help >/dev/null
bash scripts/validate_gpu_training.sh --help >/dev/null
python tools/capture_baseline.py --num_deals 2 --output artifacts/baseline/p16_release_gate.json
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check
else
  echo "git diff --check: skipped (.git metadata is not present in this image)"
fi
