#!/usr/bin/env python
"""One-call input preparation for U1c checkpoints (composition only).

Composes, without changing any algorithm:
  - r51_u1c_train.U1cDataset  (four-channel features, physical noise-injection
    machinery, per-band StandardScaler application)
  - the noise configuration and ALL scalers carried inside the checkpoint

The sampling step stays explicit (r47_t1_u1b_train.U1Flow.sample) so the
number of posterior draws and the sampling seed remain caller-controlled.
Nothing here alters model code or the noise models; this file only wires
existing release/ functions together in the same order used by
release/r55_t2_main.py.
"""
from collections import namedtuple

import json

import numpy as np
import torch

from r51_u1c_train import U1cDataset, ZERO_OM  # noqa: F401  (ZERO_OM re-export)

Inputs = namedtuple('Inputs', ['pta', 'lisa', 'ligo', 'avail',
                               'params_raw', 'params_std'])


def load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def make_item(pta_rec, lisa_rec=None, ligo_rec=None, fill_grids=None):
    """Assemble a U1c three-band item dict from per-band records.

    Bands without a real record are filled with a zero-signal spectrum
    (log10OmegaGW = ZERO_OM = -300 on that band's frequency grid); after
    noise injection those bands contain pure detector noise -- the same
    convention used throughout training and inference in the paper.

    fill_grids: optional dict {'lisa': f_list, 'ligo': f_list} supplying the
    band frequency grids for the zero fill (e.g. the 'f' field of any record
    of that band, such as json.load(open(LISA_JSON))[0]['f']). Needed only
    for bands where lisa_rec/ligo_rec is None.
    """
    fill_grids = fill_grids or {}

    def _fill(band, rec):
        if rec is not None:
            return rec
        if band not in fill_grids:
            raise ValueError('band %s: no record and no fill grid given' % band)
        return dict(f=fill_grids[band],
                    log10OmegaGW=[ZERO_OM] * len(fill_grids[band]))

    return dict(_id=pta_rec.get('_id'), params=pta_rec, pta=pta_rec,
                lisa=_fill('lisa', lisa_rec), ligo=_fill('ligo', ligo_rec))


def prepare(items, ck, noise_seed=0, noise_scale=1.0, paths=None):
    """Noise injection + features + standardization for a list of items.

    ONE call yields ONE noise realization (numpy global RNG seeded with
    noise_seed immediately before the injection pass); repeat with different
    noise_seed values to average over realizations (the paper's main table
    uses six seeds).

    paths: optional dict overriding the noise-data locations stored in the
    checkpoint config: {'noise_pta_dir': ..., 'ligo_noise_file': ...}.
    The checkpoint config is deep-copied; nothing is modified in place.

    Returns an Inputs namedtuple:
      pta/lisa/ligo : torch.float32 (N, 4, L) standardized encoder inputs
                      (channels: log10 Omega, log10 f, slope, curvature;
                      slope/curvature computed on the NOISY spectrum)
      avail         : torch.float32 (N, 3), all ones (U1c data layer:
                      every band has an input; unavailable bands carry
                      pure noise, see make_item)
      params_raw    : np.float32 (N, 9) physical parameters of the records
      params_std    : np.float32 (N, 9) checkpoint-param_scaler space
    """
    noise_config = json.loads(json.dumps(ck['config']['noise']))
    if paths:
        if paths.get('noise_pta_dir'):
            noise_config['pta']['noise_dir'] = str(paths['noise_pta_dir'])
        if paths.get('ligo_noise_file'):
            noise_config['ligo']['noise_file'] = str(paths['ligo_noise_file'])

    ds = U1cDataset(list(items), noise_config, True, True, train=False)
    ds.pta_scalers = ck['pta_scalers']
    ds.lisa_scalers = ck['lisa_scalers']
    ds.ligo_scalers = ck['ligo_scalers']
    ds.param_scaler = ck['param_scaler']
    ds.params = ck['param_scaler'].transform(ds.params_raw)

    np.random.seed(noise_seed)
    ds.refresh_noisy_cache(noise_scale)

    return Inputs(
        pta=torch.tensor(ds._cache['PTA'], dtype=torch.float32),
        lisa=torch.tensor(ds._cache['LISA'], dtype=torch.float32),
        ligo=torch.tensor(ds._cache['LIGO'], dtype=torch.float32),
        avail=torch.ones((len(ds), 3), dtype=torch.float32),
        params_raw=ds.params_raw,
        params_std=ds.params,
    )
