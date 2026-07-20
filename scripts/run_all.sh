#!/usr/bin/env bash
# Full real pipeline. Requires: GPU (training), GOOGLE_API_KEY (dataset),
# optionally Autodesk Maya (rendering; else set render.backend=fallback).
# Downloads CLIP / DINOv2 / ViT weights on first eval use.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. scrape Autodesk Maya docs =="
python -m codemaya.cli scrape

echo "== 2. build dataset (Gemini backward prompts) =="
python -m codemaya.cli build-dataset
python -m codemaya.cli analyze-dataset

echo "== 3. stage-1 SFT LoRA (GPU) =="
python -m codemaya.cli train-sft

echo "== 4. stage-2 CLIP contrastive alignment (GPU) =="
python -m codemaya.cli train-contrastive

echo "== 5. batch predict over the test set (generate + render) =="
python -m codemaya.cli predict --model-tag finetuned
# baseline for comparison:
# python -m codemaya.cli predict --model-tag base --set model.base=codellama/CodeLlama-7b-Instruct-hf

echo "== 6. evaluation =="
python -m codemaya.cli eval-syntax
python -m codemaya.cli eval-semantic
python -m codemaya.cli eval-geometry
python -m codemaya.cli eval-detection
python -m codemaya.cli eval-siamese

echo "== 7. aggregate + figures =="
python -m codemaya.cli aggregate
echo "Open notebooks/results.ipynb for the plots."
