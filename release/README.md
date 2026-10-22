# SageNet-Reverse — release/ (paper reproduction set)

This directory contains the frozen, parameterized code that reproduces the
results of the paper "Simulation-based Inference in Searches for the
Inflationary Stochastic Gravitational-Wave Backgrounds using Multi-band
Conditional Normalizing Flows" (AAS76432, ApJS).

The dataset, checkpoints, noise files, and posterior npz files referenced
below are distributed on Zenodo (see the paper's Software section); this
directory is the code snapshot (identical content is packaged there as
`code/SageNet-Reverse_release.zip`).

## File map (paper section → file)

| File | Role | Paper section |
|---|---|---|
| crop_joint.py | Dataset production v2 (coverage check → interpolate on the full curve → band crop; no extrapolation) | §2.1.2, Fig. 4 |
| train.py | Shared training library (datasets, noise models, encoders, flows) | §2 (methods) |
| r47_t1_u1b_train.py | U1b driver (availability-conditioned training lineage) | §2.3 |
| r51_u1c_train.py | U1c driver, current configuration (joint model) | §2.3, Tables 3–4 |
| r55_u1cfut_train.py | U1c driver, future configuration | §2.3, Tables 3–4 |
| r55_t2_main.py | Main-table evaluation (52,115-spectrum validation union, six noise realizations) | Tables 3–4 |
| r55_union_eval.py | Availability-group evaluation (appendix tables) | Appendix |
| r40_snr_table.py | SNR definitions and band-SNR tables (√Σ(Ω/σ)²) | §2.2 |
| figure_materials.py | Generates the six deluxetable .tex files | Tables 3–4, appendix |
| r56_figures_second_revision.py | Generates the three final PDF figures (scatter, coverage vs SNR, PP) | Figs. 5–7 |
| prepare_inputs.py | Helper: one spectrum → noise injection → 4-channel features → scaler standardization (composes existing functions only) | §2.2 |
| paths.json | Path overrides (relative to this directory); absent → scripts use their built-in defaults | — |
| freqs_downsampled.csv | LIGO 546-bin frequency grid (input to crop_joint.py) | §2.1.2 |
| requirements.txt | Python dependencies (versions as used for the paper runs) | — |

## Reproduction pipeline

1. **Dataset production** — `crop_joint.py`
   - Requires: MongoDB at `mongodb://localhost:27017/`, database
     `solve_plus`, collection `main` (the raw simulation database);
     `freqs_downsampled.csv` in the same directory.
   - Coverage levels: f_max ≥ LIGO_max → PTA+LISA+LIGO+joint; ≥ LISA_max →
     PTA+LISA; ≥ PTA_max → PTA only; else dropped. Interpolation is done on
     the full curve with `bounds_error=True` (no extrapolation).
   - Output: per-band JSON datasets plus a joint dataset whose records are
     concatenated spectra with a `band_structure` index field (name →
     [start, end) slice into the concatenated arrays).
   - The produced JSONs are already on Zenodo; this step is only needed to
     regenerate them from the raw database.

2. **Training the two configurations** — `r51_u1c_train.py` (current) and
   `r55_u1cfut_train.py` (future)
   - Inputs: the three single-band JSONs + noise files (Zenodo `noise/`);
     both scripts verify the training recipe against a reference checkpoint
     (`J1_flow_joint.pt`, included on Zenodo) by assertion before training.
   - Output: `U1c_flow.pt` per configuration (equal-weight best epoch by
     three real-availability-group held-out NLLs, from epoch 300).

3. **Main table** — `r55_t2_main.py`
   - Inputs: the trained checkpoint, the three single-band JSONs, noise
     files; six noise realizations (seeds 20265921, 11, 22, 33, 44, 55),
     5,000 posterior samples per spectrum.
   - Output: per-realization npz (posterior_mean, q05/q50/q95, SNRs, ids) —
     identical to Zenodo `npz/`.

4. **Figures and tables** — `figure_materials.py` (six .tex tables) and
     `r56_figures_second_revision.py` (three PDF figures).

## Notes

- All scripts accept path overrides via `paths.json` next to this file; with
  no `paths.json`, the scripts fall back to their built-in defaults.
- This release/ snapshot is pushed as a single commit; the repository root
  keeps the original 2026-04 library.
