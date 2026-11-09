# Clean Enough to Recycle?

**A computer vision model that sorts waste, spots contamination, and outlines where it is.**

Stanford CS 231N: Deep Learning for Computer Vision · Team of 3 · [Poster](CS_231N_Poster.pdf)

## Highlights

- Raised contamination recall on real photos from 0% to 48% by adding synthetic training images (recall is the share of contaminated items the model catches)
- Trained a ResNet-50 (a standard image recognition network) in PyTorch to predict 3 things at once: waste type, whether contamination is present, and which pixels it covers
- Traced the remaining gap with Grad-CAM heatmaps, which show where the model looks: it learned the look of synthetic stains, not contamination itself

## Built with

Python · PyTorch · NumPy · Grad-CAM

---

## Results

| Condition | Test set | Contamination recall | Contamination F1 | Localization mIoU |
|---|---|---|---|---|
| No synthetic data | 1,403 real TACO images | 0% | 0.000 | 0.000 |
| 50% synthetic data | 1,403 real TACO images | **48%** | **0.138** | 0.017 |
| 100% synthetic data | 1,403 real TACO images | 18% | 0.094 | 0.016 |
| 100% synthetic data | Synthetic validation set | n/a | 1.000 | **0.759** |

Recall is the share of contaminated items caught. F1 balances catching items against false alarms (0 to 1). mIoU measures how well the predicted outline overlaps the real one (0 to 1). Condition names follow our final report. A model trained only to sort waste type reaches 60% accuracy on the ZeroWaste validation set.

## What it does

- **Waste classification:** ResNet-50 backbone maps images to one of four waste categories (ZeroWaste labels)
- **Contamination presence detection:** binary head with asymmetric loss (false negatives penalized 5×) predicts whether an item is contaminated
- **Contamination localization:** decoder upsamples backbone features to a pixel-level segmentation mask
- **Synthetic augmentation:** blob-based compositor overlays grease, residue, and liquid textures onto clean images at configurable probability
- **GradCAM visualization:** gradient-weighted activation maps for the presence head highlight what drives contamination predictions
- **WandB logging:** optional experiment tracking with per-step loss and metric curves

## Experimental conditions

| Condition | Active heads | Synthetic contamination |
|-----------|-------------|------------------------|
| B0 | classification | no |
| B1 | classification · presence · localization | no |
| B2 | classification · presence · localization | yes |
| B3 | classification · presence · localization | yes |

B2 and B3 are identical in head configuration; use them for replicated runs.

## Architecture

| Layer | Stack |
|-------|-------|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Classification head | GAP → Linear(2048, 4) |
| Presence head | GAP → Linear(2048, 1), weighted BCE |
| Localization head | 5-stage bilinear upsampler → Conv(32, 1), BCE + Dice |
| Loss | weighted sum of active head losses |
| Optimizer | AdamW + cosine annealing with linear warmup |
| Training data | ZeroWaste (clean) + optional synthetic contamination |
| Evaluation data | TACO (real-world contaminated waste) |

## Repository layout

```
datasets.py     ZeroWaste and TACO dataset loaders
models.py       MultiHeadResNet with segmentation decoder
losses.py       multi-task loss (CE + weighted BCE + Dice)
metrics.py      presence, segmentation, and classification meters
train.py        training loop with cosine warmup and evaluation
gradcam.py      GradCAM for the presence head
synthetic.py    synthetic contamination compositor
TACO/           TACO dataset submodule (pedropro/TACO)
```

## Quick start

Prerequisites: Python 3.10+, PyTorch, ZeroWaste dataset

```bash
# Install dependencies
pip install torch torchvision pillow numpy

# Classification only (B0)
python train.py --condition B0 --zerowaste_root /path/to/zerowaste

# Full multi-task with synthetic augmentation (B2)
python train.py --condition B2 --zerowaste_root /path/to/zerowaste

# Add TACO evaluation
python train.py --condition B3 \
  --zerowaste_root /path/to/zerowaste \
  --taco_test_index /path/to/taco_index.json

# With WandB logging
python train.py --condition B3 --zerowaste_root /path/to/zerowaste --wandb
```

## Training options

| Flag | Default | Purpose |
|------|---------|---------|
| `--condition` | required | B0 / B1 / B2 / B3 |
| `--zerowaste_root` | required | path to ZeroWaste dataset |
| `--taco_test_index` | none | JSON index of TACO images for evaluation |
| `--epochs` | 30 | training epochs |
| `--batch_size` | 32 | |
| `--lr` | 1e-4 | peak learning rate |
| `--lambda_cls` | 1.0 | classification loss weight |
| `--lambda_pres` | 2.0 | presence loss weight |
| `--lambda_loc` | 1.0 | localization loss weight |
| `--fn_pos_weight` | 5.0 | false-negative penalty for presence head |
| `--wandb` | off | enable WandB logging |

## Datasets

ZeroWaste and TACO are not included in this repository.

- **ZeroWaste:** download from the ZeroWaste project; point `--zerowaste_root` at the directory containing `splits_final_deblurred/`
- **TACO:** cloned as a submodule at `TACO/`; run `python TACO/download.py` to fetch images, then build a JSON index with `image_path`, `contam`, and optional `mask_path` per entry

## Notes

The presence head uses an asymmetric positive weight (`fn_pos_weight`, default 5×) because sending a contaminated item through is worse than a false alarm. Adjust this if your deployment cost model differs.

Synthetic contamination runs only for B2/B3 and fires with 50% probability per sample. The compositor blends grease, residue, or liquid textures using Gaussian blob alpha masks.

GradCAM is computed against the presence head only (`PresenceGradCAM` in `gradcam.py`); localization head outputs can be visualized directly as sigmoid maps.
