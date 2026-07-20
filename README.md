# CodeMaya — LLM-Driven VFX

Reference implementation of **"LLM Driven VFX: Automating Script Generation and Model Validation with LLMs"** (A. Roy Chowdhury). CodeMaya turns a natural-language prompt into an executable **Autodesk Maya MEL script**, renders the 3D scene, and **validates the render against the prompt** with vision models.

```
prompt ──▶ fine-tuned Code LLaMA ──▶ MEL script ──▶ Maya render ──▶ .obj + image
                                                             │
                                    validation ◀─────────────┘
             (CLIP semantic · DINOv2 visual · Chamfer/Hausdorff/volume/area · ViT detection)
```

> **The paper (`report.pdf`) is in this repo.** This codebase reproduces its full pipeline: scraping → dataset construction → LoRA SFT → CLIP contrastive alignment → rendering → inference → evaluation → visualization.

## Objective
- **Generate:** fine-tune Code LLaMA-Instruct 7B (LoRA) to produce MEL from natural language.
- **Align:** a second contrastive stage uses frozen CLIP encoders to pull the rendered output toward the prompt.
- **Validate:** an automatic, multi-metric evaluation of the rendered 3D asset (semantic, visual, geometric, and prompt-guided object detection).

## Environment & Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Requirements by stage:
- **Dataset construction** needs a Gemini API key: `export GOOGLE_API_KEY=...` (model `gemini-2.0-flash`).
- **Training** needs a CUDA GPU (QLoRA via bitsandbytes). On CPU/macOS it falls back to full-precision LoRA (slow — not recommended).
- **Rendering** needs Autodesk Maya (`mayapy`) for real MEL renders. Without Maya, set `render.backend: fallback` to render geometric primitives via trimesh (covers the cube/sphere eval set).
- **Evaluation** downloads CLIP / DINOv2 / ViT weights from the HuggingFace Hub on first use.

## Execution
Everything runs through one CLI; every stage reads `configs/default.yaml` (override with `--set key.sub=value`).
```bash
python -m codemaya.cli scrape                     # 1. scrape Autodesk Maya docs
python -m codemaya.cli build-dataset              # 2. Gemini backward prompt-gen -> JSONL
python -m codemaya.cli analyze-dataset            # 2b. dataset stats (paper Fig.3 / Tables 2-3)
python -m codemaya.cli train-sft                  # 3. stage-1 LoRA SFT
python -m codemaya.cli train-contrastive          # 4. stage-2 CLIP contrastive alignment
python -m codemaya.cli infer --prompt "..."       # 5. generate MEL for one prompt
python -m codemaya.cli render                      # 6. render a MEL -> obj + image
python -m codemaya.cli predict                     # 6b. batch generate+render test set
                                                   #     -> results/generations.jsonl
python -m codemaya.cli eval-syntax                 # 7a. syntax validity (Table 1)
python -m codemaya.cli eval-semantic               # 7b. CLIP + DINOv2 similarity
python -m codemaya.cli eval-geometry               # 7c. Chamfer/Hausdorff/volume/area
python -m codemaya.cli eval-detection              # 7d. prompt-guided ViT detection
python -m codemaya.cli eval-siamese                # 7e. Siamese-ViT pairwise similarity
python -m codemaya.cli aggregate                   # 8. build results tables + figure data
```
Then open `notebooks/results.ipynb` for the figures.

Add `--smoke` to assemble/verify a stage without heavy execution.

## Architecture
```
codemaya/
  config.py            # YAML config loader (attribute access + CLI overrides)
  cli.py               # single entrypoint dispatching to every stage
  utils/               # io (jsonl/json), logging
  data/                # scrape_maya_docs, build_dataset (Gemini), analyze_dataset
  training/            # sft_lora (stage 1), contrastive_align (stage 2)
  rendering/           # render (mayapy backend + trimesh fallback)
  inference/           # generate (base + LoRA adapter -> MEL)
  evaluation/          # syntax_validity, semantic_visual, geometry,
                       # object_detection, siamese_similarity, aggregate
configs/default.yaml   # all knobs
notebooks/             # results.ipynb — reproduces paper figures/tables
tests/                 # pytest suite (pure-python logic)
scripts/               # thin shell wrappers for each stage
```

## Testing
```bash
pytest -q                       # pure-logic unit tests (no GPU/Maya/API needed)
bash scripts/smoke.sh           # offline end-to-end pipeline on tiny/mock components
```
`--smoke` on any stage assembles and exercises the code path with a tiny random
model / mock encoders / fallback renderer — no big downloads, no training.

## Status
Full pipeline implemented and wired (`codemaya-implementation` branch): scraping,
dataset construction, stage-1 SFT, stage-2 contrastive alignment, rendering,
inference, the complete evaluation stack, aggregation, and the results notebook.
Every stage is smoke-verified and the pure-logic cores are unit-tested. **Real
training is intentionally not run here** — this repo ships the code; launch the
runs yourself with `scripts/run_all.sh` (needs GPU + Gemini key; Maya optional).

## License
See `LICENSE`.
