# IndiaMediaLens: LLM-Driven Stance Detection

This repository contains the core training, evaluation, and deployment scripts for **IndiaMediaLens**, a platform designed to monitor editorial stance and regulatory accountability in Indian digital print media.

## Project Overview
The project evaluates the effectiveness of Large Language Models (LLMs) in performing Aspect-Based Sentiment Analysis (ABSA) and Stance Detection on complex political news discourse. By comparing zero-shot foundation models against domain-specific fine-tuned models, this research highlights the "Generalization Gap" when shifting from standard product sentiment benchmarks to political stance evaluation.

## Repository Structure
* `/src/`: Core Python scripts.
  * `train.py`: Fine-tuning script utilizing PEFT/LoRA.
  * `eval.py`: Inference and evaluation script for In-Distribution (ID) testing.
  * `eval_news.py`: Adapted inference script for Out-of-Distribution (OOD) Stance Detection.
* `/jobs/`: PBS shell scripts for execution on the Ashoka University HPC cluster.
* `/logs/`: Output logs capturing inference metrics, F1-scores, and confusion matrices.
* `/utils/`: Shell scripts for monitoring HPC hardware and GPU utilization.

## Methodology
* **Base Architecture:** `Mistral-7B-v0.1`
* **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT) via **LoRA** (Rank = 16) to prevent catastrophic forgetting while adapting to ABSA tasks.
* **Infrastructure:** Trained on the Ashoka University HPC cluster (NVIDIA RTX 4090 equivalent).
* **Target Task:** Adapting ABSA to identify stance toward complex political entities (Pro/Anti mapped to Positive/Negative).

## Experimental Results

The models were first evaluated on the SemEval-2014 Laptop dataset (In-Distribution) and subsequently tested on the IndiaMediaLens news dataset (Out-of-Distribution) to measure domain robustness.

| Metric | Mistral-7B (In-Distribution) | Mistral-7B (Out-of-Distribution) |
| :--- | :--- | :--- |
| **Overall Accuracy** | 79.27% | 33.00% |
| **Weighted F1-Score**| 0.79 | 0.40 |
| **Positive / Pro F1**| 0.84 | 0.31 |
| **Negative / Anti F1**| 0.86 | 0.32 |
| **Neutral F1** | 0.58 | 0.45 |

### Key Findings
1. **Domain Sensitivity:** The significant performance decay (79% → 33%) indicates that sentiment logic learned from product reviews does not natively translate to structural political reporting.
2. **The Conflict/Neutral Ambiguity:** Human annotators frequently categorize mixed arguments (Conflict) as Neutral. The model, lacking multi-premise reasoning capabilities, fails to resolve these hidden conflicts.

## Future Work: Argument Mining
To resolve the Generalization Gap and the Conflict/Neutral ambiguity, the next phase of IndiaMediaLens will integrate **Argument-Grounded Stance Detection**. By linking ABSA outputs to argument mining frameworks, the system will extract specific premises that justify media stances, moving beyond sentiment pattern-matching into structural logical analysis.
