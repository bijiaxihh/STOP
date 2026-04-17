<div align="center">
  <img src="./submitssion/STOP/figures/stop.png" width="220" alt="STOP Logo">
</div>

# STOP: Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning

Official implementation for the paper **"Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning"**.

<p align="center">
<img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python 3.12+">
<a href="./Prefix_Rejection____DFT___ACL26_-61.pdf">
<img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper PDF"></a>
<a href="./submitssion/STOP/index.html">
<img src="https://img.shields.io/badge/Project-Page-black.svg" alt="Project Page"></a>
</p>

<p align="center">
<a href="./submitssion/STOP/index.html">Project Page</a> •
<a href="./Prefix_Rejection____DFT___ACL26_-61.pdf">Paper</a> •
<a href="#getting-started">Getting Started</a> •
<a href="#training">Training</a> •
<a href="#inference">Inference</a> •
<a href="#citation">Citation</a>
</p>

## Abstract

Parallel reasoning enhances Large Reasoning Models (LRMs) but incurs prohibitive costs due to futile paths caused by early errors. To mitigate this, path pruning at the prefix level is essential, yet existing research remains fragmented without a standardized framework. We propose the first systematic taxonomy of path pruning, categorizing methods by their signal source and learnability. This classification reveals the unexplored potential of learnable internal methods, motivating our proposal of **STOP** (Super TOken for Pruning). Extensive evaluations across LRMs ranging from 1.5B to 20B parameters demonstrate that STOP achieves superior effectiveness and efficiency compared to existing baselines. Furthermore, we validate the scalability of STOP under varying compute budgets and distill our findings into practical deployment guidelines.

## Key Features

* **Type IV pruning**: STOP is an instantiation of learnable internal pruning.
* **Launch-Check-Resume pipeline**: cache prefixes, score them, and continue only the best candidates.
* **Internal-state based pruning**: STOP reads KV-cache states instead of relying only on surface text.
* **Effective and efficient**: STOP improves accuracy while reducing token usage by over **70%** in many settings.
* **Scalable deployment**: STOP remains strong across model sizes from **1.5B to 20B** and varying compute budgets.

## Why STOP?

In standard parallel reasoning, every sampled path is typically generated to completion and then aggregated. However, many trajectories are already doomed by early mistakes and still consume the full decoding budget. STOP prunes such futile paths at the prefix level, saving computation and improving the candidate pool used for final aggregation.

<p align="center">
  <img src="./submitssion/STOP/figures/motivation.png" width="88%" alt="STOP Motivation">
</p>

## A Unified Taxonomy of Path Pruning

We categorize existing pruning methods by two dimensions: whether the pruning signal comes from **internal** or **external** states, and whether the signal generator is **learnable** or **non-learnable**. This taxonomy highlights the missing but desirable quadrant of **learnable internal pruning**, which STOP instantiates as a Type IV method.

<p align="center">
  <img src="./submitssion/STOP/figures/main.png" width="88%" alt="STOP Taxonomy">
</p>

## Method

STOP follows a simple three-stage workflow:

1. **Launch**: generate short prefixes and cache their internal states.
2. **Check**: append special tokens and score each prefix with a lightweight classifier.
3. **Resume**: keep only the top-ranked prefixes and continue generation.

The key implementation idea is that STOP reuses the heavy computation already stored in the KV cache, which makes the pruning step lightweight.

<p align="center">
  <img src="./submitssion/STOP/figures/method.png" width="88%" alt="STOP Method">
</p>

## Results

STOP consistently improves both effectiveness and efficiency across benchmarks and model scales.

* On **AIME24 (1.5B)**, performance improves from `30.10` to `37.92`.
* Under fixed compute budgets, STOP boosts **GPT-OSS-20B** accuracy on **AIME25** from `84%` to nearly `90%`.
* In many setups, STOP reduces token consumption by **over 70%**.
* In the **AIMO3** tool-use setting, STOP improves the score from `39` to `42` and `43` in the reported settings.

<p align="center">
  <img src="./submitssion/STOP/figures/scaling_main.png" width="88%" alt="STOP Scaling">
</p>

## Repository Structure

```text
.
├── src/
│   ├── classification_dataset_harmony.py
│   ├── evaluate_harmony_vllm.py
│   ├── finetuning_harmony.py
│   ├── inference.py
│   └── modeling_harmony.py
├── scripts/
│   └── run_train-harmony.sh
├── Prefix-Generation./
├── vllm/
├── vllm.tar.gz
├── Prefix_Rejection____DFT___ACL26_-61.pdf
└── submitssion/STOP/
```

## Getting Started

This repository depends on a local customized `vllm` source tree and external model/checkpoint assets. Before running training or inference, prepare:

* base model weights
* tokenizer files
* STOP classifier checkpoints or output directory
* local `artifacts/` paths used by the scripts

A trimmed dependency list is available in [requirements.txt](/data/250010180/bjx/STOP/STOP-github/submitssion/STOP/requirements.txt).

## Training

The main training entrypoint is:

[finetuning_harmony.py](/data/250010180/bjx/STOP/STOP-github/src/finetuning_harmony.py)

The recommended launcher script is:

[run_train-harmony.sh](/data/250010180/bjx/STOP/STOP-github/scripts/run_train-harmony.sh)

Example:

```bash
bash scripts/run_train-harmony.sh
```

Direct Python usage:

```bash
python src/finetuning_harmony.py \
  --model_path /path/to/model \
  --data_path /path/to/train.jsonl \
  --val_data_path /path/to/val.jsonl \
  --output_dir /path/to/output
```

The training code learns a lightweight classifier on top of a frozen base model, using LoRA adapters and a classifier head to score prefixes.

## Inference

The main inference entrypoint is:

[inference.py](/data/250010180/bjx/STOP/STOP-github/src/inference.py)

Example:

```bash
python src/inference.py
```

This script expects local runtime assets under configurable paths such as:

* `artifacts/models/`
* `artifacts/checkpoints/`
* `artifacts/tiktoken_encodings/`
* `artifacts/reference/`

It also uses the local `vllm/` source tree bundled in this repository.

## Evaluation

For scoring prefix records with a vLLM classify endpoint:

```bash
python src/evaluate_harmony_vllm.py \
  --input-jsonl INPUT.jsonl \
  --output-jsonl OUTPUT.jsonl \
  --summary-json SUMMARY.json \
  --num-assess-tokens N \
  --assess-special-token-id TOKEN_ID
```

## Citation

If you find this repository useful, please cite:

```bibtex
@article{bi2026cut,
  title={Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning},
  author={Bi, Jiaxi and Luo, Tongxu and Du, Wenyu and Tang, Zhengyang and Wang, Benyou},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
