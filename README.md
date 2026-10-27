# Removing Interference and Recovering Content Imaginatively for Visible Watermark Removal

**AAAI 2024**

Yicheng Leng, Chaowei Fang, Gen Li, Yixiang Fang, Guanbin Li

## Overview

We propose **RIRCI**, a two-stage framework for visible watermark removal that
separates *watermark component exclusion* from *background content restoration*.
Earlier methods either treat the task as a direct image-to-image translation or
mix watermark segmentation and background recovery in a single branch, which
leaves watermark residuals behind.

RIRCI instead splits the problem:

1. **Watermark component exclusion.** A U-Net with a shared decoder and two
   prediction branches estimates the watermark mask `M̂` and the background
   component `Ĉ_b = (1 - A) ⊙ I`.
2. **Background content restoration.** A dual-path network restores the
   background from two complementary sources: a *content restoration* path
   that uses the intrinsic background visible beneath a semi-transparent
   watermark (`M̂`, `Ĉ_b`), and a *content imagination* path that uses the
   unaffected peripheral context (`(1 - M̂) ⊙ J`, `M̂`). A non-local fusion
   network combines the two predictions `I_r` and `I_i` into the final
   reconstruction `Î`.

Both paths are built from a **Global and Local Context Interaction (GLCI)**
module, which extracts local features with a block MLP, global features with a
spectral (FFT) transform, and exchanges information between the two with
channel attention.

## Results

Reported in the paper (Table 1). RMSE / RMSE_w are on the 0-255 scale, IoU is a
percentage.

| Dataset | PSNR | SSIM | RMSE | RMSE_w | F1 | IoU |
|---|---|---|---|---|---|---|
| HWVOC | 39.52 | 0.9855 | 3.8647 | 14.7919 | 0.8802 | 79.82 |
| PW | 41.89 | 0.9866 | 4.5423 | 13.5077 | 0.8528 | 78.31 |

## Repository structure

```
train.py                 # training entry point
test.py                  # evaluation on a dataset split
infer.py                 # single-image / folder inference
options.py               # command line options
evaluation.py            # PSNR / SSIM / RMSE / IoU / F1 helpers
datasets/
  rirci_dataset.py       # dataset reader
src/
  models/                # training wrapper and losses
  networks/              # stage-1 backbone, stage-2 restoration, GLCI blocks
  modules/               # FFC / spectral transform and gMLP layers
  utils/                 # losses, checkpointing and small helpers
pytorch_ssim/            # vendored SSIM implementation
scripts/train.sh
scripts/test.sh
```

## Installation

```bash
pip install -r requirements.txt
```

The code targets Python 3.8+ and a CUDA GPU. The perceptual loss uses a
torchvision VGG16 backbone, whose weights are downloaded on first use.

## Dataset

**The HWVOC and PW datasets are not released with this repository.** The reader
below documents the layout used in the paper so that a compatible split can be
built from your own background images and watermarks.

```
RIRCI/
├── train/
│   ├── watermarked/     # J = A * W + (1 - A) * I
│   ├── gt/              # I, the watermark-free image
│   ├── mask/            # M = 255 * (A > 0)
│   ├── cleancomp/       # C_b = (1 - A) * I
│   └── alpha/           # A, saved as 0-255 grayscale
├── valid/
│   └── (the same five folders)
└── test/                # optional
    └── (the same five folders)
```

- All five folders are required; `alpha/` is needed because the training loss
  is split by watermark opacity.
- A sample is identified by its file stem, and the stem must match across all
  five folders. `.png` is recommended.
- The paper uses 256x256 images. HWVOC contains 60,000 training and 2,500 test
  images built from PASCAL VOC2012 backgrounds and 858 watermark logos; PW
  contains 60,000 training and 1,045 test images built from 2,435 private
  backgrounds.

## Training

```bash
python train.py \
    --dataset_dir /path/to/RIRCI \
    --checkpoint checkpoints \
    --name rirci \
    --epochs 100 --schedule 65 --lr 1e-3 \
    --train-batch 8 --test-batch 1 \
    --lambda_l1 2 --lambda_content 1 --lambda_mask 3
```

or `DATASET_DIR=/path/to/RIRCI bash scripts/train.sh`.

The defaults in `options.py` match the configuration above. Checkpoints,
TensorBoard logs and the resolved command line are written under
`checkpoints/<name>/`.

## Evaluation

```bash
python test.py \
    --evaluate \
    --dataset_dir /path/to/RIRCI \
    --checkpoint checkpoints \
    --name rirci \
    --resume checkpoints/rirci/model_best.pth.tar \
    --eval_split val
```

`test.py` reports PSNR, SSIM, RMSE, RMSE_w, mask IoU and mask F1, and saves
`[input | ground truth | prediction | I_r | I_i | Ĉ_b]` panels under
`checkpoints/<name>/results/`.

## Inference on your own images

```bash
python infer.py \
    --resume checkpoints/rirci/model_best.pth.tar \
    --input path/to/image_or_folder \
    --output inference_output
```

Each output is an `[input | restored | predicted mask]` panel. Inputs are
resized to `--crop_size` (256 by default).

## Notes

- **Trained weights are not released.**
- The training objective follows Eq. (2)-(10) of the paper: masked L1, VGG16
  perceptual loss, mask BCE, with `lambda_1 = 2`, `lambda_2 = 1`,
  `lambda_3 = 3`, `gamma = 1.5` and opacity threshold `alpha = 0.75`.
- There is no built-in data generator: the paper's datasets were produced by
  randomly scaling, rotating and placing watermark logos on background images,
  then saving the five folders described above.
- Stage 1 follows the shared-decoder design of SLBR (Liang et al., 2021); the
  two-stage decomposition, the dual-path restoration network and the GLCI
  module are this paper's contribution. See `NOTICE` for upstream projects.

## Citation

```bibtex
@inproceedings{leng2024removing,
  title={Removing Interference and Recovering Content Imaginatively for Visible Watermark Removal},
  author={Leng, Yicheng and Fang, Chaowei and Li, Gen and Fang, Yixiang and Li, Guanbin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Acknowledgements

This code builds on [SLBR](https://github.com/bcmi/SLBR-Visible-Watermark-Removal),
[MAXIM](https://github.com/google-research/maxim) (gMLP layers),
[FFC](https://github.com/pkumivision/FFC) and
[pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).
See `NOTICE` for details. The project is released under the MIT license
(see `LICENSE`).
