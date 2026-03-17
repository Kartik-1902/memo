#!/usr/bin/env bash
# MEMO-MODIFICATION: small wrapper to set PYTHONPATH for ImageNet experiments.

set -euo pipefail

PYTHONPATH="${PYTHONPATH:-}:$(pwd)/imagenet-exps" \
python "$@"
