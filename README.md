# AI-PVI: Deep Learning for Patient-Specific Pulmonary Vein Isolation

This repository contains the deep learning models for predicting patient-specific PVI (Pulmonary Vein Isolation) regions and lines from pre-ablation voltage maps.

## Overview

- **AI Model 1 (PVI Area)**: Segments the candidate PVI region from the pre-ablation voltage map
- **AI Model 2 (PVI Line)**: Depicts the PVI lines within the predicted region

## Model Architecture

Both models use a 7-level U-Net with ConvMixer blocks:
- Encoder-decoder with skip connections and attention gates
- Channel widths: [16, 32, 48, 72, 104, 144, 192]
- ConvMixer blocks with 7×7 depthwise convolution
- Parameters: ~1.43M

## Training

### Two-Phase Curriculum Learning
1. **Phase 1**: Pre-training on all available data
2. **Phase 2**: Fine-tuning on recurrence-free cases only

### Ensemble
- 5-seed ensemble with test-time augmentation (TTA)
- TTA: horizontal/vertical flip + 90°/180°/270° rotation

## Performance

| Model | IoU (mean) | 95% CI | Dice | AUROC |
|-------|-----------|--------|------|-------|
| AI Model 1 (PVI Area) | 0.87 | 0.86-0.88 | 0.93 | 0.99 |
| AI Model 2 (PVI Line) | 0.80 | 0.79-0.81 | 0.89 | 0.99 |

### Per-Projection Performance (AI Model 1)
| View | IoU |
|------|-----|
| Anterior-Posterior (AP) | 0.86 |
| Posterior-Anterior (PA) | 0.88 |
| Superior (SUP) | 0.87 |

## Requirements

```
tensorflow>=2.17.0
numpy
pillow
scikit-learn
```

## Usage

### Training
```bash
# Model 1: PVI Area
python model1_pvi_area_train.py <seed>

# Model 2: PVI Line
python model2_pvi_line_train.py <seed>
```

### Evaluation
```bash
# Model 1: PVI Area (5-ensemble + TTA)
python model1_pvi_area_eval.py

# Model 2: PVI Line (5-ensemble + TTA)
python model2_pvi_line_eval.py
```

## Citation

If you use this code, please cite:

```
[Citation to be added upon publication]
```

## License

This project is licensed under the MIT License.

## Contact

For questions, please contact the corresponding author.
