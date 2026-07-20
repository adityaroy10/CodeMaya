#!/usr/bin/env bash
# End-to-end smoke run of the whole pipeline on tiny/mock components.
# No GPU, no Maya, no API key, no big downloads. Verifies wiring, not quality.
set -euo pipefail
cd "$(dirname "$0")/.."

D="${1:-outputs/smoke}"
COMMON="--set paths.raw_docs=$D/raw --set paths.cache=$D/cache --set paths.dataset=$D/dataset --set paths.renders=$D/renders --set paths.results=$D/results --set paths.checkpoints=$D/ckpt"

echo "== scrape ==";           python -m codemaya.cli scrape --smoke $COMMON
echo "== build-dataset ==";    python -m codemaya.cli build-dataset --smoke $COMMON
echo "== analyze-dataset ==";  python -m codemaya.cli analyze-dataset --smoke $COMMON
echo "== train-sft ==";        python -m codemaya.cli train-sft --smoke $COMMON
echo "== train-contrastive =="; python -m codemaya.cli train-contrastive --smoke $COMMON
echo "== predict ==";          python -m codemaya.cli predict --smoke $COMMON
# eval stages read results/generations.jsonl (syntax + geometry need no big models)
echo "== eval-syntax ==";      python -m codemaya.cli eval-syntax $COMMON
echo "== eval-geometry ==";    python -m codemaya.cli eval-geometry $COMMON
echo "== eval-detection ==";   python -m codemaya.cli eval-detection --smoke $COMMON
echo "== eval-siamese ==";     python -m codemaya.cli eval-siamese --smoke $COMMON
echo "== aggregate ==";        python -m codemaya.cli aggregate $COMMON
echo "== DONE -> $D/results/summary.md =="
