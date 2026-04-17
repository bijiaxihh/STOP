<div align="center">
  <img src="./figures/main.png" width="78%" alt="STOP Overview">
</div>

# STOP: Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning

Official implementation for **"Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning"**.

<p align="center">
<img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python 3.12+">
<a href="./index.html">
<img src="https://img.shields.io/badge/Project-Page-black.svg" alt="Project Page"></a>
</p>

<p align="center">
<a href="./index.html">Project Page</a> •
<a href="#overview">Overview</a> •
<a href="#getting-started">Getting Started</a> •
<a href="#citation">Citation</a>
</p>

## Overview

Parallel reasoning improves the performance of Large Reasoning Models (LRMs), but it is also expensive: many sampled paths are already unpromising from their early prefixes and still consume the full decoding budget. STOP addresses this problem with a lightweight internal pruning module that reads prefix KV-cache states, predicts whether a path is promising, and resumes only the most valuable candidates.

## Abstract

Parallel reasoning enhances Large Reasoning Models (LRMs) but incurs prohibitive costs due to futile paths caused by early errors. To mitigate this, path pruning at the prefix level is essential, yet existing research remains fragmented without a standardized framework. We propose the first systematic taxonomy of path pruning, categorizing methods by their signal source and learnability. This reveals the unexplored potential of **learnable internal pruning**, which we instantiate with **STOP** (Super TOken for Pruning). Extensive evaluations across LRMs ranging from 1.5B to 20B parameters demonstrate that STOP improves both effectiveness and efficiency. We further validate its scalability under varying compute budgets and distill these observations into empirical deployment guidelines.

## Key Features

* **First systematic taxonomy of path pruning**: We organize prior methods by signal source and learnability.
* **Type IV pruning**: STOP is the first instantiation of the learnable internal pruning paradigm.
* **Early path pruning**: STOP identifies low-value trajectories from their prefixes instead of waiting for full completion.
* **Internal and lightweight**: STOP works directly on cached internal states, avoiding expensive prefix recomputation.
* **Strong effectiveness and efficiency**: STOP improves reasoning quality while reducing token usage by over **70%** in many settings.
* **Practical deployment guideline**: STOP provides an empirical rule for choosing retention ratios under different compute budgets.

## STOP Framework

STOP follows a simple three-stage pipeline:

1. **Launch**: generate short reasoning prefixes and cache their internal states.
2. **Check**: append a special token and score each prefix with a lightweight classifier.
3. **Resume**: keep only the most promising candidates and continue generation.

Because STOP reuses the KV cache, it avoids re-encoding the full prefix and adds only a small amount of extra overhead.

<p align="center">
  <img src="./figures/method.png" width="88%" alt="STOP Method">
</p>

## Why Prune Early?

In standard parallel reasoning, every sampled path is usually generated to completion and then aggregated. However, many paths fail because of mistakes made very early in the reasoning process. Continuing these trajectories wastes compute and can even hurt the final answer when poor paths are mixed into aggregation.

<p align="center">
  <img src="./figures/motivation.png" width="88%" alt="STOP Motivation">
</p>

## Results

STOP consistently improves both effectiveness and efficiency.

* On **AIME24 (1.5B)**, performance improves from `30.10` to `37.92`.
* Under fixed compute budgets, STOP boosts **GPT-OSS-20B** accuracy on **AIME25** from `84%` to nearly `90%`.
* In many setups, STOP reduces token consumption by **over 70%**.
* In the **AIMO3** tool-use setting, STOP improves the score from `39` to `42` and `43` in the reported configurations.

<p align="center">
  <img src="./figures/scaling_main.png" width="88%" alt="STOP Scaling">
</p>

## Getting Started

Install the core Python dependencies:

```bash
pip install -r requirements.txt
```

This repository uses a **modified local vLLM source tree** instead of a plain PyPI `vllm` package. Unpack it before running:

```bash
cd STOP
tar -xzf vllm/vllm.tar.gz -C .
```

You will also need to prepare your own base model weights, tokenizer files, and STOP checkpoints for local experiments.

## Usage

### Evaluate Prefix Records

```bash
python STOP/src/evaluate_harmony_vllm.py \
  --input-jsonl INPUT.jsonl \
  --output-jsonl OUTPUT.jsonl \
  --summary-json SUMMARY.json \
  --num-assess-tokens N \
  --assess-special-token-id TOKEN_ID
```

### Train the STOP Classifier

```bash
python STOP/src/finetuning_harmony.py --help
```

or

```bash
bash STOP/scripts/run_train-harmony.sh
```

### Local Inference

```bash
python STOP/src/inference.py
```

### Distributed Prefix Generation

```bash
bash STOP/scripts/run-distribute-inference.sh
```

## Repository Structure

```text
.
├── STOP/
│   ├── Prefix-Generation./
│   ├── scripts/
│   ├── src/
│   └── vllm/
│       └── vllm.tar.gz
├── data/
│   └── benchmark/
├── figures/
├── index.html
├── requirements.txt
└── README.md
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
