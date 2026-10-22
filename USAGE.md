For the code that reproduces the results of the paper, see release/
and release/README.md.

# Normalizing Flow Training and Testing Guide

## File Structure

```
SageNetR/
├── train.py           # Unified training script
├── test.py            # Unified testing script
├── summarize.py       # Experiment summary generator
├── verify_lisa_noise.py  # LISA noise model verification
├── configs/           # Configuration files
│   ├── config_package_mock.json      # Mock noise experiment package
│   └── config_package_observed.json  # Observed noise experiment package
├── noise/             # Noise data files
│   ├── NANOGrav15yr/  # PTA noise data
│   ├── AplusDesign.txt   # LIGO A+ design sensitivity
│   └── C_O1_O2_O3.dat.txt  # LIGO O3 cross-correlation spectrum
├── dataset/           # Dataset directory
│   ├── joint/         # Joint multi-band datasets
│   └── single/        # Single-band datasets
└── models/            # Model save directory
```

---

## Training (train.py)

### Using Configuration Files (Recommended)

```bash
# Run experiment package (1 joint + 3 single)
python train.py --config configs/config_package_mock.json
python train.py --config configs/config_package_observed.json
```

### Configuration File Reference

```json
{
    "name": "Experiment name",
    "description": "Experiment description",
    "dataset": {
        "type": "concat/conti/single",
        "path": "Dataset path",
        "bands": ["PTA", "LISA", "LIGO"],
        "data_dir": "Single-band data directory"
    },
    "features": {
        "use_slope": true,
        "use_curvature": true
    },
    "training": {
        "epochs": 1500,
        "batch_size": 256,
        "learning_rate": 0.0002,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "scheduler_T_max": 300
    },
    "model": {
        "num_inputs": 9,
        "num_hidden": 128,
        "num_layers": 10,
        "context_dim": 64
    },
    "noise": {
        "use_complex": false,
        "level": 1.0,
        "glitch_prob": 0.5
    },
    "output": {
        "save_dir": "models",
        "save_name": "model_name.pt"
    }
}
```

### Command-Line Arguments (Alternative)

```bash
python train.py \
    --dataset-type concat \
    --json-path dataset/joint/Dataset_PTA_LISA_LIGO.json \
    --save-dir models \
    --save-name my_model.pt \
    --epochs 2000 \
    --no-curvature
```

---

## Testing (test.py)

### Basic Usage

```bash
# Evaluate model (primary 5 parameters)
python test.py \
    --model models/flow_concat_slope.pt \
    --data dataset/joint/Dataset_PTA_LISA_LIGO.json

# Evaluate all 9 parameters
python test.py \
    --model models/flow_concat_slope.pt \
    --data dataset/joint/Dataset_PTA_LISA_LIGO.json \
    --params all

# Evaluate secondary parameters only (5-8)
python test.py \
    --model models/flow_concat_slope.pt \
    --data dataset/joint/Dataset_PTA_LISA_LIGO.json \
    --params secondary
```

### Single-Band Model Testing

```bash
# Test a single band
python test.py \
    --model models/single/pta_flow.pt \
    --data dataset/single/PTA_fixed_14pts.json \
    --mode single \
    --band PTA

python test.py \
    --model models/single/lisa_flow.pt \
    --data dataset/single/LISA_fixed_500pts.json \
    --mode single \
    --band LISA
```

### Model Comparison

```bash
# Compare multiple models
python test.py --compare \
    --models models/model1.pt models/model2.pt models/model3.pt \
    --labels "Baseline" "With Slope" "Conti" \
    --data dataset/joint/Dataset_PTA_LISA_LIGO.json

# Compare single-band vs joint model
python test.py --compare \
    --models models/single/pta_flow.pt models/single/lisa_flow.pt models/flow_joint.pt \
    --labels "PTA Only" "LISA Only" "Joint" \
    --data dataset/joint/Dataset_PTA_LISA_LIGO.json
```

### Additional Options

```bash
--figpath eval          # Output directory for figures
--num-samples 10000     # Number of posterior samples
--no-corner             # Skip corner plot
```

---

## Output Description

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| R² | Coefficient of determination (closer to 1 is better) |
| RMSE | Root mean square error |
| MAE | Mean absolute error |
| NMAE (%) | Normalized MAE (relative to data range) |
| Rel. CI Width | Relative confidence interval width |
| KS | Kolmogorov-Smirnov statistic (from P-P plot) |

### Output Figures

- `PP_Plot.pdf`: P-P plot, evaluates posterior calibration
- `Scatter_Plot.pdf`: True vs Predicted scatter plot
- `Corner_Plot.pdf`: Single-sample posterior distribution (optional)
- `model_comparison.pdf`: Multi-model comparison bar chart

---

## Dataset Description

| Dataset | Description | PTA Points | LISA Points | LIGO Points |
|---------|-------------|------------|-------------|-------------|
| concat | Non-uniform concatenation | 14 | 500 | 546 |
| conti | Continuous interpolation | 78 | 136 | 133 |
| single | Single-band independent | 14 | 500 | 546 |

---

## 9 Cosmological Parameters

| Index | Parameter | Transform | Description |
|-------|-----------|-----------|-------------|
| 0 | r | log10 | Tensor-to-scalar ratio |
| 1 | n_t | - | Tensor spectral index |
| 2 | κ₁₀ | log10 | Reheating parameter |
| 3 | T_re | log10 | Reheating temperature |
| 4 | ΔN_re | - | Effective degrees of freedom change |
| 5 | Ω_bh² | - | Baryon density |
| 6 | Ω_ch² | - | Dark matter density |
| 7 | H₀ | - | Hubble constant |
| 8 | A_s | log(10¹⁰x) | Scalar spectral amplitude |

Primary parameters: 0-4 (r, n_t, κ₁₀, T_re, ΔN_re)
Secondary parameters: 5-8 (cosmological parameters)

---

## Frequently Asked Questions

### Q: How to create a new experiment?

Copy an existing configuration file and modify it:
```bash
cp configs/config_package_mock.json configs/my_experiment.json
# Edit my_experiment.json
python train.py --config configs/my_experiment.json
```

### Q: How to resume training?

Checkpoint resumption is not currently supported. Consider reducing epochs or saving intermediate models.

### Q: Out of memory?

Adjust in the configuration file:
- `batch_size`: Reduce to 128 or 64
- `num_samples`: Reduce sampling count during testing

### Q: How to use GPU?

GPU is automatically detected. If available, training runs on GPU; otherwise, CPU is used.
