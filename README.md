# SageNet-Reverse

Code for **"Simulation-based Inference in Searches for the Inflationary
Stochastic Gravitational-Wave Backgrounds using Multi-band Conditional
Normalizing Flows"** (AAS76432, ApJS).

- **`release/`** — frozen, parameterized code that reproduces the paper's
  results (dataset production → training the two configurations → main
  table → figures). Start at [`release/README.md`](release/README.md).
- Data, checkpoints, noise files, and posterior samples are distributed on
  Zenodo (see the paper's Software section).
- The repository root additionally keeps the original 2026-04 normalizing
  flow library (`train.py`, `test.py`, `configs/`, `noise/`).

## Quick start

```bash
cd release
pip install -r requirements.txt
# point paths.json at the Zenodo files, then:
python r55_t2_main.py   # regenerate the main-table npz files
```

## Citation

If you use this code, please cite the paper and the Zenodo data record.
