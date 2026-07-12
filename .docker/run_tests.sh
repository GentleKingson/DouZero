#!/usr/bin/env bash
# Test runner used by both the Docker image and CI.
#
# Runs the deterministic, offline, CPU-only checks required by P00.
# Failures in any stage stop the script (set -e).

set -euo pipefail

echo "=== environment ==="
python -c "import sys, platform, torch, numpy; \
print('python', sys.version.split()[0]); \
print('platform', platform.platform()); \
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); \
print('numpy', numpy.__version__)"

echo
echo "=== compileall ==="
python -m compileall -q douzero train.py evaluate.py generate_eval_data.py train_belief.py evaluate_belief.py

echo
echo "=== CLI --help ==="
python train.py --help >/dev/null
python evaluate.py --help >/dev/null
python generate_eval_data.py --help >/dev/null
python train_belief.py --help >/dev/null
python evaluate_belief.py --help >/dev/null

echo
echo "=== pytest ==="
python -m pytest "$@"
