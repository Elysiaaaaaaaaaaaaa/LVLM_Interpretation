# AGENTS.md

## What This Repo Does
Generates saliency heatmaps to interpret where Large Vision-Language Models (LVLMs) look when answering questions. Associated paper: "Where do Large Vision-Language Models Look at when Answering Questions?" (arXiv:2503.13891)

## Supported Models
- `llava` (llava-v1.5-7b)
- `llava_next` (llava-onevision-qwen2-72b-ov)
- `cambrian` (nyu-visionx/cambrian-8b)
- `mgm` (requires local path `mgm/work_dirs/mgm13bhd`)

## Setup
```bash
pip install --upgrade pip
bash install.sh
```
Note: `install.sh` overrides `pyproject.toml` timm version (installs 0.9.16, not 0.6.13). Flash-attn requires separate install with `--no-build-isolation`.

## Running
```bash
python main.py --method iGOS+ --model llava --dataset <name> --data_path <questions> --image_folder <images> --output_dir <output> [hyperparams]
```
Hyperparams: `--size`, `--L1`, `--L2`, `--L3`, `--ig_iter`, `--iterations`, `--momentum`, `--lr`, `--gamma`

## Data Formats
Supports CSV, JSON, JSONL, PKL, and HuggingFace datasets (`load_dataset` with "val" split).

## Architecture
- Model-specific code in subdirs: `llava/`, `llava_next/`, `cambrian/`, `mgm/`
- Each subdir mirrors structure: `model/builder.py` (load model), `mm_utils.py` (image processing), `conversation.py` (templates)
- `main.py` routes to correct model via `--model` flag (line 217-232)
- Core methods: `methods.py` (iGOS+, iGOS++), `methods_helper.py`, `utils.py`

## Dependencies Worth Knowing
- `yake` for keyword extraction (optional: `--use_yake`)
- `flash-attn` (reinstalled by install.sh)
- `mamba-ssm`, `causal-conv1d` (for some model variants)
- `open_clip_torch`, `diffusers` (for vision encoders)

## No Tests
This repo has no test suite, lint config, or CI. The `pyproject.toml` excludes "tests*" from package.
