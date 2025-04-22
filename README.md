# STATS 507 Final Project — Instance Segmentation on ARMBench

This repository contains the final project for **STATS 507: Data Science Analytics Using Python** at the University of Michigan. The project applies two modern deep learning models—**Mask R-CNN** and **Mask2Former**—to a robotic instance segmentation benchmark.

> If you're interested in applying deep learning to real-world segmentation tasks using Python, PyTorch, and Hugging Face—this project is for you!

---

## Dataset

- **ARMBench**: A real-world, object-centric benchmark for robotic manipulation.
- **Subset used**: `mix-object-tote` with COCO-style annotations.
- **Download link**: [http://armbench.s3-website-us-east-1.amazonaws.com/](http://armbench.s3-website-us-east-1.amazonaws.com/)

After downloading, structure your dataset like this:
armbench-segmentation-0.1/
└── mix-object-tote/
├── images/
├── train.json
└── val.json

---

## Project Overview

- **Goal**: Compare the performance of two segmentation architectures in a domain with strong distributional shift (COCO → ARMBench).
- ⚙**Models**:
  - `Mask R-CNN` (ResNet-50 FPN, PyTorch)
  - `Mask2Former` (Swin-B, Hugging Face)
- **Metrics**: COCO-style mAP, loss curves, object-size-based AP, recall

---

## Results Snapshot

| Model        | mAP@[.5:.95] | mAP@0.5 | mAP@0.75 |
|--------------|-------------|--------|----------|
| Mask R-CNN   | 0.5422      | 0.6971 | 0.5389   |
| Mask2Former  | 0.0001      | 0.0007 | 0.0000   |

- Mask R-CNN shows reliable generalization to ARMBench.
- Mask2Former converged in training loss but failed to produce valid masks, likely due to domain mismatch and limited tuning.

---

## Key Lessons

- Pretrained models require adaptation in new domains.
- Loss convergence ≠ mAP improvement — evaluation config and architecture matter.
- PyTorch + Hugging Face + Optuna makes model experimentation modular and reproducible.

---

## Environment

Make sure the following packages are installed:

- `torch`, `torchvision`, `transformers`, `optuna`, `pycocotools`

Install via:

```bash
pip install -r requirements.txt



