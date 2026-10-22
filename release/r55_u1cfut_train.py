#!/usr/bin/env python
# R55 T1: U1c_fut_s1 训练 driver — 由 r51_u1c_train.py 复制，唯一改动=噪声换未来档
#   (PTA density_mock / LISA 同现在档 cbe / LIGO CE40 aplus_asd)；纯噪声填充随之用未来档噪声
#   输出 outputs/r55；TAG U1c_fut_s1；ckpt 名 r55_t1_*；其余逐字同 U1c_s1
# ---- 以下为 r51 原头注释 ----
# R51 T1: U1c_s1 训练 driver — 缺失语义拆分（基于 r48_t1_u1bfix_train.py 3a77daf3）
# 只改两处（结构/加噪函数不动）：
#   1. 谱延伸不到的波段改填纯噪声：items 组装时给每个样本三波段全输入
#      （真实不存在的波段信号=0，加噪流程与其他样本完全相同）；avail=1（数据层）
#   2. 探测器组合按固定分布抽样（0.40/0.15/0.15/0.15/0.05/0.05/0.05），
#      替代 U1bfix 的逐段独立丢弃 drop_p=0.15 —— 在 __getitem__ 层实现
# 其余与 U1bfix_s1 完全相同（1500ep curriculum 0.2 / 过采样×2 / scaler 引用训练 ds /
# 选模 ep≥300 四 ckpt / ep300+900+1500 保险件 / 种子 20261008）
# 开训前五核对 (a)-(e)；报告点 ep20/ep320 口径同 R49
import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# --- release path wiring: optional sibling paths.json (absent = original behavior) ---
_DIR = Path(__file__).resolve().parent
_PFILE = _DIR / 'paths.json'
_P = json.loads(_PFILE.read_text(encoding='utf-8')) if _PFILE.is_file() else {}

def _rp(key, default):
    v = _P.get(key, default)
    return str((_DIR / str(v)).resolve())
R = Path(_rp('root', '/root/SageNetR'))
WS = R / 'workspace'
sys.path.insert(0, str(WS / 'new'))
sys.path.insert(0, str(R))

from r47_t1_u1b_train import (  # noqa: E402
    U1Flow, BaseDataset, BAND_CONFIG_CONCAT, NOISE, StandardScaler,
    _build_param_weights)

PTA_JSON = WS / 'dataset/single/PTA_fixed_14pts.json'
LISA_JSON = WS / 'dataset/single/LISA_fixed_500pts.json'
LIGO_JSON = WS / 'dataset/single/LIGO_fixed_546pts.json'
JOINT_JSON = WS / 'dataset/Dataset_PTA_LISA_LIGO.json'
J1_CKPT = Path(_rp('r40_refs', '/root/SageNetR/r40_refs')) / 'J1_flow_joint.pt'
J1_SHA = '3be078d15a'

OUT = WS / 'outputs/r55'
PREP = OUT / 't1_prep'
OUT.mkdir(parents=True, exist_ok=True)
PREP.mkdir(parents=True, exist_ok=True)

SEED = 20261008
TAG = 'U1c_fut_s1'
PARAM_EMPHASIS = dict(enabled=True, aux_weight=0.1,
                      weights={'T_re': 3.0, 'DN_re': 0.3})
HP = dict(epochs=1500, batch_size=512, learning_rate=8e-4, weight_decay=1e-5,
          grad_clip=5.0, scheduler_T_max=300, num_inputs=9, num_hidden=128,
          num_layers=10, context_dim=64, use_slope=True, use_curvature=True)
CURRICULUM = dict(final_noise=1.0, warmup_frac=0.2)  # 前 300 轮 0.05→1.0

# ★ R51 改动 2: 探测器组合固定分布（替代逐段独立丢弃）
COMBO_PROBS = {  # (pta,lisa,ligo) -> prob
    (1, 1, 1): 0.40, (1, 0, 0): 0.15, (0, 1, 0): 0.15, (0, 0, 1): 0.15,
    (1, 1, 0): 0.05, (1, 0, 1): 0.05, (0, 1, 1): 0.05}
COMBO_KEYS = list(COMBO_PROBS.keys())
COMBO_P = np.array([COMBO_PROBS[k] for k in COMBO_KEYS], dtype=np.float64)
COMBO_P /= COMBO_P.sum()
COMBO_ARR = np.array(COMBO_KEYS, dtype=np.float32)

EXPECT = dict(full=166127, val_union=52073, train=114054, only_pta=69201,
              pta_lisa=20746, all3=24065, pta_val=33226, lisa_val=15927,
              ligo_val=9420)

LOG = OUT / 'u1c_train.log'


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '+00:00'


def log(msg):
    line = f'[{utc()}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsons():
    with open(PTA_JSON) as f:
        pta = json.load(f)
    with open(LISA_JSON) as f:
        lisa = json.load(f)
    with open(LIGO_JSON) as f:
        ligo = json.load(f)
    return pta, lisa, ligo


def build_split(pta, lisa, ligo):
    pta_ids = [str(x['_id']) for x in pta]
    lisa_ids = [str(x['_id']) for x in lisa]
    ligo_ids = [str(x['_id']) for x in ligo]
    _, pv = train_test_split(np.arange(len(pta)), test_size=0.2,
                             random_state=42)
    _, lv = train_test_split(np.arange(len(lisa)), test_size=0.2,
                             random_state=42)
    _, gv = train_test_split(np.arange(len(ligo)), test_size=0.2,
                             random_state=42)
    val_union = (set(pta_ids[i] for i in pv) | set(lisa_ids[i] for i in lv)
                 | set(ligo_ids[i] for i in gv))
    sp, sl, sg = set(pta_ids), set(lisa_ids), set(ligo_ids)
    full = sp | sl | sg
    train_ids = full - val_union
    counts = dict(full=len(full), val_union=len(val_union),
                  train=len(train_ids),
                  only_pta=len(train_ids & (sp - sl - sg)),
                  pta_lisa=len(train_ids & ((sp & sl) - sg)),
                  all3=len(train_ids & (sp & sl & sg)),
                  pta_val=len(pv), lisa_val=len(lv), ligo_val=len(gv))
    log(f'split: 全集 {counts["full"]} / val并集 {counts["val_union"]} / '
        f'训练 {counts["train"]}')
    return dict(train_ids=sorted(train_ids), val_ids=sorted(val_union)), counts


# ★ R51 改动 1: 三波段全输入 —— 真实不存在的波段用纯噪声条目
# （信号=0 的占位谱：log10OmegaGW 全 0 = Ω=1e0 的平线？不行 —— 必须"信号=0"。
#   U1Dataset 特征计算吃 log10OmegaGW 数组；信号=0 的正确表示 = log10(0+noise)，
#   即让 _inject_batch 在 omega=10**om 上加噪时 om=log10(0)=-inf —— 不可行。
#   实现：raw 特征层直接给"零信号"占位数组 om = 全 -300（10^-300≈0），
#   _inject_batch: obs = |10^-300 + noise| = |noise| → log10(max(|noise|,1e-300))
#   = 判词 (b) 要求的口径，逐位成立。）
ZERO_OM = -300.0  # log10OmegaGW 占位：10^-300 ≈ 0（信号=0）


def make_union_items_v2(pta, lisa, ligo, id_list):
    """U1c 条目组装：三波段全给。真实存在的波段用真数据；
    不存在的波段用零信号占位条目（频格取该波段 json 首行；Ω 占位 -300）。
    返回 items + real_avail（真实可用性，用于留出分组/过采样；数据 avail 全 1）。"""
    pta_by = {str(x['_id']): x for x in pta}
    lisa_by = {str(x['_id']): x for x in lisa}
    ligo_by = {str(x['_id']): x for x in ligo}
    # 零信号占位模板（频格=各自 json 首行，500/546 点；PTA 恒真给但保险同款）
    def _zero_entry(json_path):
        with open(json_path) as f:
            first = json.load(f)[0]
        return dict(f=first['f'], log10OmegaGW=[ZERO_OM] * len(first['f']))
    z_lisa = _zero_entry(LISA_JSON)
    z_ligo = _zero_entry(LIGO_JSON)
    items, real_av = [], []
    for oid in id_list:
        p = pta_by[oid]
        l = lisa_by.get(oid)
        g = ligo_by.get(oid)
        items.append(dict(_id=oid, params=p, pta=p,
                          lisa=l if l else z_lisa,
                          ligo=g if g else z_ligo))
        real_av.append([1.0, 1.0 if l else 0.0, 1.0 if g else 0.0])
    return items, np.array(real_av, dtype=np.float32)


class U1cDataset(torch.utils.data.Dataset):
    """U1c 数据集：三波段全输入（真实或零信号+噪声）；
    训练期 __getitem__ 按固定分布抽探测器组合（R51 改动 2）。"""

    def __init__(self, items, noise_config, use_slope=True,
                 use_curvature=True, train=True):
        self.use_slope = use_slope
        self.use_curvature = use_curvature
        self.train = train
        base = BaseDataset(noise_config, use_slope, use_curvature)
        self.compute_features = base.compute_features
        self.extract_params = base.extract_params

        bc = BAND_CONFIG_CONCAT
        N = len(items)
        # 三波段 raw 特征全量预计算（无缺失占位 —— 全部样本都有三波段输入）
        self.pta_raw = np.zeros((N, 4, bc['PTA']['len']), dtype=np.float32)
        self.lisa_raw = np.zeros((N, 4, bc['LISA']['len']), dtype=np.float32)
        self.ligo_raw = np.zeros((N, 4, bc['LIGO']['len']), dtype=np.float32)
        for i, it in enumerate(items):
            om = np.array(it['pta']['log10OmegaGW'], dtype=np.float32)
            fr = np.array(it['pta']['f'], dtype=np.float32)
            self.pta_raw[i] = base.compute_features(om, fr)
            om = np.array(it['lisa']['log10OmegaGW'], dtype=np.float32)
            fr = np.array(it['lisa']['f'], dtype=np.float32)
            self.lisa_raw[i] = base.compute_features(om, fr)
            om = np.array(it['ligo']['log10OmegaGW'], dtype=np.float32)
            fr = np.array(it['ligo']['f'], dtype=np.float32)
            self.ligo_raw[i] = base.compute_features(om, fr)
        self.num_channels = 4
        # 数据层 avail 全 1（三波段都有输入）；组合抽样发生在 __getitem__
        self.avail = np.ones((N, 3), dtype=np.float32)

        # 噪声模型（频格从 items 实值取 —— 三波段全有，必取得到）
        pta_freq = np.array(items[0]['pta']['f'], dtype=np.float32)[
            :bc['PTA']['len']]
        self.pta_noise = base._build_pta_noise_model(noise_config)
        if self.pta_noise is not None and hasattr(self.pta_noise,
                                                  'validate_freqs'):
            self.pta_noise.validate_freqs(pta_freq)
        self.lisa_freq_grid = np.array(
            items[0]['lisa']['f'], dtype=np.float32)[:bc['LISA']['len']]
        self.lisa_noise = base._build_lisa_noise_model(
            noise_config, self.lisa_freq_grid)
        self.ligo_freq_grid = np.array(
            items[0]['ligo']['f'], dtype=np.float32)[:bc['LIGO']['len']]
        self.ligo_noise = base._build_ligo_noise_model(
            noise_config, self.ligo_freq_grid)

        # scaler：三波段全部训练样本（含零信号填充）的加噪参考拟合
        # ★ R51: 重新拟合 —— 与 U1bfix 的"仅可用样本"不同
        self.pta_scalers = [StandardScaler() for _ in range(4)]
        self.lisa_scalers = [StandardScaler() for _ in range(4)]
        self.ligo_scalers = [StandardScaler() for _ in range(4)]
        self.param_scaler = StandardScaler()
        self.params_raw = np.array([base.extract_params(it['params'])
                                    for it in items], dtype=np.float32)
        if train:
            self._fit_scalers()
            self.params = self.param_scaler.fit_transform(self.params_raw)
        else:
            self.params = np.zeros_like(self.params_raw)
        self._cache = {}
        # 留出组合（固定种子抽一次，永不变）—— 仅 val 用；train 每次抽样
        self.hold_combo = None
        print(f'U1cDataset: {N} samples; 三波段全输入 (真实或零信号+噪声)')

    def _noisy_ref(self, raw, model, band, max_n=4000, seed=12345):
        rng = np.random.default_rng(seed)
        if len(raw) > max_n:
            sub = raw[rng.choice(len(raw), size=max_n, replace=False)]
        else:
            sub = raw
        out = np.empty_like(sub)
        for i in range(len(sub)):
            om = sub[i, 0, :].copy()
            fr = sub[i, 1, :]
            if band == 'pta':
                om_n = model.inject(om)
            else:
                om_n = model.inject(om, fr)
            out[i] = self.compute_features(om_n, fr)
        return out

    def _fit_scalers(self):
        for band, raw, model, scalers in (
                ('pta', self.pta_raw, self.pta_noise, self.pta_scalers),
                ('lisa', self.lisa_raw, self.lisa_noise, self.lisa_scalers),
                ('ligo', self.ligo_raw, self.ligo_noise, self.ligo_scalers)):
            print(f'[{band.upper()}] fitting scalers on {len(raw)} samples '
                  f'(全样本含零信号填充, noise-injected ref)...')
            ref = self._noisy_ref(raw, model, band)
            for c in range(4):
                scalers[c].fit(ref[:, c, :].reshape(-1, 1))

    def refresh_noisy_cache(self, noise_scale_override=1.0):
        for band, raw, model, scalers in (
                ('PTA', self.pta_raw, self.pta_noise, self.pta_scalers),
                ('LISA', self.lisa_raw, self.lisa_noise, self.lisa_scalers),
                ('LIGO', self.ligo_raw, self.ligo_noise, self.ligo_scalers)):
            om = raw[:, 0, :].copy()
            fr = raw[0, 1, :]
            omega_noisy = self._inject_batch(om, model, band.lower(),
                                             noise_scale_override)
            feats = self._features_batch(omega_noisy, fr)
            arr = np.stack([scalers[c].transform(
                feats[:, c, :].reshape(-1, 1)).reshape(
                feats.shape[0], -1) for c in range(4)], axis=1)
            self._cache[band] = arr.astype(np.float32)

    def _inject_batch(self, om, model, band, scale):
        if band == 'pta':
            dev = model.sample_omega_noise(om.shape[0])
            omega = 10.0 ** om.astype(np.float64)
            obs = omega + dev * scale
            return np.log10(np.maximum(obs, 1e-300)).astype(np.float32)
        sig = np.asarray(model.sigma_omega, dtype=np.float64) * scale
        omega = 10.0 ** om.astype(np.float64)
        noise = np.random.normal(0.0, 1.0, om.shape) * sig[None, :]
        obs = np.abs(omega + noise)
        return np.log10(np.maximum(obs, 1e-300)).astype(np.float32)

    def _features_batch(self, om_noisy, freq):
        out = np.empty((om_noisy.shape[0], self.num_channels,
                        om_noisy.shape[1]), dtype=np.float32)
        for i in range(om_noisy.shape[0]):
            out[i] = self.compute_features(om_noisy[i], freq)
        return out

    def __len__(self):
        return len(self.params)

    def sample_combo(self, idx):
        """训练：每条样本独立按固定分布抽组合；留出：hold_combo 固定。"""
        if self.hold_combo is not None:
            return self.hold_combo[idx].copy()
        k = np.random.choice(len(COMBO_KEYS), p=COMBO_P)
        return COMBO_ARR[k].copy()

    def __getitem__(self, idx):
        avail = self.avail[idx].copy()  # 全 1
        if self.train:
            avail = self.sample_combo(idx)  # ★ R51 改动 2
        pta = self._cache['PTA'][idx].copy()
        lisa = self._cache['LISA'][idx].copy()
        ligo = self._cache['LIGO'][idx].copy()
        if avail[0] == 0:
            pta[:] = 0
        if avail[1] == 0:
            lisa[:] = 0
        if avail[2] == 0:
            ligo[:] = 0
        return (torch.tensor(pta, dtype=torch.float32),
                torch.tensor(lisa, dtype=torch.float32),
                torch.tensor(ligo, dtype=torch.float32),
                torch.tensor(self.params[idx], dtype=torch.float32),
                torch.tensor(avail, dtype=torch.float32))


def pretrain_checks(ds, val_dss, keep_items, real_avail):
    """开训前核对 (a)(b)(c)(e)。(d) 在烟测后做。返回 dict 落盘。"""
    rep = {}
    # (a) 加噪核对 9 项：三波段 × curriculum 0.05/0.5/1.0
    #     逐位一致判据 = 同种子下 _inject_batch 与参考实现完全相同
    checks = []
    np.random.seed(999)
    for band, raw, model in (
            ('pta', ds.pta_raw, ds.pta_noise),
            ('lisa', ds.lisa_raw, ds.lisa_noise),
            ('ligo', ds.ligo_raw, ds.ligo_noise)):
        idx = np.random.choice(len(raw), size=50, replace=False)
        for scale in (0.05, 0.5, 1.0):
            np.random.seed(7000 + int(scale * 100))
            a = ds._inject_batch(raw[idx, 0, :].copy(), model, band, scale)
            if band == 'pta':
                np.random.seed(7000 + int(scale * 100))
                b = ref_pta_inject(raw[idx, 0, :].copy(), model, scale)
            else:
                np.random.seed(7000 + int(scale * 100))
                b = ref_lisa_ligo_inject(raw[idx, 0, :].copy(), model,
                                         band, scale)
            d = float(np.abs(a.astype(np.float64)
                             - b.astype(np.float64)).max())
            checks.append((band, scale, d))
            assert d == 0, f'(a) 加噪核对 FAIL {band} {scale}: {d}'
    rep['a_noise'] = checks
    log('(a) 加噪核对 9 项逐位一致 PASS')

    # (b) 纯噪声填充核对：100 条真实 only-PTA 样本
    only_pta_mask = (real_avail[:, 1] == 0) & (real_avail[:, 2] == 0)
    cand = np.where(only_pta_mask)[0]
    rng = np.random.default_rng(777)
    sel = rng.choice(cand, size=100, replace=False)
    for arr, nm in ((ds.lisa_raw, 'LISA'), (ds.ligo_raw, 'LIGO')):
        v = arr[sel, 0, :]
        assert np.all(v == ZERO_OM), f'(b) {nm} 零信号占位 FAIL: {v.min()}'
    for nm, raw_arr, noise_model, sseed in (
            ('LISA', ds.lisa_raw, ds.lisa_noise, 31337),
            ('LIGO', ds.ligo_raw, ds.ligo_noise, 31338)):
        np.random.seed(sseed)
        om_z = raw_arr[sel, 0, :].copy()
        got = ds._inject_batch(om_z, noise_model, nm.lower(), 1.0)
        np.random.seed(sseed)
        sig = np.asarray(noise_model.sigma_omega, dtype=np.float64)
        noise = np.random.normal(0.0, 1.0, om_z.shape) * sig[None, :]
        want = np.log10(np.maximum(np.abs(noise), 1e-300)).astype(np.float32)
        d = float(np.abs(got.astype(np.float64)
                         - want.astype(np.float64)).max())
        assert d == 0, f'(b) {nm} 纯噪声输出 FAIL: {d}'
    rep['b_purenoise_maxdiff'] = 0.0
    log('(b) 纯噪声填充核对: 100 条 only-PTA 样本 LISA/LIGO 信号全 0、'
        '输出=log10(max(|noise|,1e-300)) 最大差=0 PASS')

    # (c) scaler 一致性：留出 ds（同一对象引用）vs 训练 ds
    v0 = val_dss['only_pta']
    for nm, a, b in (('pta', v0.pta_scalers, ds.pta_scalers),
                     ('lisa', v0.lisa_scalers, ds.lisa_scalers),
                     ('ligo', v0.ligo_scalers, ds.ligo_scalers),
                     ('param', [v0.param_scaler], [ds.param_scaler])):
        for s1, s2 in zip(a, b):
            assert np.abs(s1.mean_ - s2.mean_).max() == 0, f'(c) {nm} mean_'
            assert np.abs(s1.scale_ - s2.scale_).max() == 0, f'(c) {nm} scale_'
    rep['c_scaler_maxdiff'] = 0.0
    log('(c) scaler 一致性 mean_/scale_ max差=0 PASS')

    # (e) 泄露核对（一整轮规模的组合抽样统计）
    n = len(ds)
    rng = np.random.default_rng(20261008 + 51)
    combos = rng.choice(len(COMBO_KEYS), size=n, p=COMBO_P)
    freqs = {str(COMBO_KEYS[k]): float((combos == k).mean())
             for k in range(len(COMBO_KEYS))}
    for k, key in enumerate(COMBO_KEYS):
        want_p = COMBO_PROBS[key]
        got_p = freqs[str(key)]
        assert abs(got_p - want_p) < 0.01, \
            f'(e) 组合频率 FAIL {key}: {got_p} vs {want_p}'
    tre = ds.params_raw[:, 3]  # log10 T_re (ACTIVE 序 idx3)
    nt = ds.params_raw[:, 1]   # n_t
    av_mat = COMBO_ARR[combos]
    corrs = {}
    for bi, bn in enumerate(('pta', 'lisa', 'ligo')):
        for pn, pv in (('log10T_re', tre), ('n_t', nt)):
            c = float(np.corrcoef(av_mat[:, bi], pv)[0, 1])
            corrs[f'{bn}-{pn}'] = c
            assert abs(c) < 0.02, f'(e) 相关性 FAIL {bn}-{pn}: {c}'
    rep['e_combo_freqs'] = freqs
    rep['e_corr'] = corrs
    log('(e) 泄露核对: 七组合频率 |Δp|<0.01 + 六个相关 |r|<0.02 PASS '
        f'(max|r|={max(abs(v) for v in corrs.values()):.4f})')
    return rep


# ★ R55 唯一改动：未来档噪声（逐项取自现有未来档模型 ckpt config，开训前 assert）
NOISE_PTA_DIR = WS / 'new/noise/NANOGrav15yr'
CE_ASD = Path(_rp('ce40_inputs', '/root/SageNetR/ce40_inputs')) / 'CE' / 'CE40km_v009_strain_ASD_LIGO546.txt'
NOISE_FUT = dict(
    pta=dict(mode='density_mock', noise_dir=str(NOISE_PTA_DIR),
             per_sample_noise=True, noise_scale=1.0, n_bins_use=14),
    lisa=dict(enabled=True, noise_source='analytical', noise_preset='cbe',
              noise_scale=1.0, injection_mode='physical', logscale_base=1.0,
              T_obs=4.0, include_confusion=True),
    ligo=dict(injection_mode='physical', noise_type='aplus_asd',
              noise_file=str(CE_ASD), noise_scale=1.0, logscale_base=1.0),
)
FUT_SOURCES = dict(
    pta=WS / 'outputs/r37/a3_train/mock_s1/PTA_flow.pt',
    lisa=WS / 'outputs/r41/t1_train/Jfut_fix_s1/flow_joint.pt',
    ligo=WS / 'outputs/r40/t2_train/L3_CE40_s1/LIGO_flow.pt',
    joint=WS / 'outputs/r41/t1_train/Jfut_fix_s1/flow_joint.pt')
DENSITY_MOCK_SHA = '718e5b40fbcdbd21'


def check_fut_noise_sources():
    """config diff：三波段噪声配置逐项 vs 未来档来源 ckpt（豁免=noise_dir 换机路径）。"""
    rows = []
    for band in ('pta', 'lisa', 'ligo'):
        for srcname in (band, 'joint'):
            ck = torch.load(FUT_SOURCES[srcname], map_location='cpu',
                            weights_only=False)
            n = ck['config']['noise']
            got = dict(n[band]) if band in n else None
            if got is None:
                continue
            mine = dict(NOISE_FUT[band])
            for k in ('noise_dir',):
                got.pop(k, None); mine.pop(k, None)
            assert got == mine, f'config diff FAIL {band} vs {srcname}: {got} != {mine}'
            rows.append(f'{band}<-{FUT_SOURCES[srcname].parent.name}')
    dsha = sha256_file(NOISE_PTA_DIR / 'density_mock.npy')
    assert dsha.startswith(DENSITY_MOCK_SHA), dsha
    assert CE_ASD.is_file()
    log(f'config diff 未来档噪声来源逐项一致 PASS: {rows}; '
        f'density_mock sha {dsha[:16]}; CE40 ASD {CE_ASD.name}')
    return rows


def NoiseCfgForCheck():
    return {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0, 'glitch_prob': 0.0,
            'pta': dict(NOISE_FUT['pta']), 'lisa': dict(NOISE_FUT['lisa']),
            'ligo': dict(NOISE_FUT['ligo'])}


def ref_pta_inject(om, model, scale):
    dev = model.sample_omega_noise(om.shape[0])
    omega = 10.0 ** om.astype(np.float64)
    obs = omega + dev * scale
    return np.log10(np.maximum(obs, 1e-300)).astype(np.float32)


def ref_lisa_ligo_inject(om, model, band, scale):
    sig = np.asarray(model.sigma_omega, dtype=np.float64) * scale
    omega = 10.0 ** om.astype(np.float64)
    noise = np.random.normal(0.0, 1.0, om.shape) * sig[None, :]
    obs = np.abs(omega + noise)
    return np.log10(np.maximum(obs, 1e-300)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    j1_sha = sha256_file(J1_CKPT)
    assert j1_sha.startswith(J1_SHA), j1_sha
    # J1 母本 config diff 声明（与 U1bfix 同款核对思路；逐项 assert 略——
    # 噪声/超参与 U1bfix 完全一致已在其 driver 验过，此处声明差异清单）
    log(f'[R55 U1c_fut_s1 = U1c_s1 配方, 唯一改动=未来档噪声 NGmock/cbe/CE40] '
        f'J1 母本 ({J1_SHA}) — U1c 与 U1bfix_s1 差异全清单: '
        f'(1) ★ 谱延伸不到的波段填纯噪声 om=-300 占位, avail=1, '
        f'三波段全输入; (2) ★ 探测器组合固定分布抽样 '
        f'{COMBO_PROBS} 替代 drop_p=0.15; (3) ★ scaler 重新拟合 '
        f'(三波段全样本含零信号填充的加噪参考); 其余逐项同 U1bfix_s1 '
        f'(1500ep curriculum 0.2/过采样×2/选模 ep≥300 四 ckpt/'
        f'ep300+900+1500 保险件/留出 R47 3,000 ids/种子 20261008)')
    check_fut_noise_sources()
    set_seed(SEED)
    t0 = time.time()
    pta, lisa, ligo = load_jsons()
    split, counts = build_split(pta, lisa, ligo)

    train_items, real_avail = make_union_items_v2(pta, lisa, ligo,
                                                  split['train_ids'])
    del pta, lisa, ligo
    gc.collect()

    noise_config = NoiseCfgForCheck()
    # 留出：沿用 R47 3,000 条 ids 不变（rng 20261008 按真实可用性三组各 1,000）
    rng = np.random.default_rng(20261008)
    grp_idx = {'only_pta': np.where((real_avail[:, 1] == 0)
                                    & (real_avail[:, 2] == 0))[0],
               'pta_lisa': np.where((real_avail[:, 1] > 0)
                                    & (real_avail[:, 2] == 0))[0],
               'all3': np.where(real_avail[:, 2] > 0)[0]}
    hold_idx = {g: rng.choice(ix, size=1000, replace=False)
                for g, ix in grp_idx.items()}
    hold_all = np.concatenate(list(hold_idx.values()))
    assert len(np.unique(hold_all)) == 3000
    hold_mask = np.zeros(len(train_items), dtype=bool)
    hold_mask[hold_all] = True
    keep_items = [it for it, m in zip(train_items, ~hold_mask) if m]
    keep_real_avail = real_avail[~hold_mask]
    log(f'留出: 沿用 R47 3,000 条 (rng 20261008); 实际训练 {len(keep_items)}')

    ds = U1cDataset(keep_items, noise_config, HP['use_slope'],
                    HP['use_curvature'], train=True)

    # 留出 ds：组合固定种子抽一次永不复变（任务单）；scaler 引用训练 ds
    val_dss = {}
    hold_rng = np.random.default_rng(20265151)  # 留出组合固定种子
    for g, ix in hold_idx.items():
        v = U1cDataset([train_items[i] for i in ix], noise_config,
                       HP['use_slope'], HP['use_curvature'], train=False)
        v.pta_scalers = ds.pta_scalers
        v.lisa_scalers = ds.lisa_scalers
        v.ligo_scalers = ds.ligo_scalers
        v.param_scaler = ds.param_scaler
        v.params = v.param_scaler.transform(v.params_raw)
        v.hold_combo = COMBO_ARR[hold_rng.choice(
            len(COMBO_KEYS), size=len(ix), p=COMBO_P)]
        val_dss[g] = v
    # 三组里各组合条数（任务单要求报）
    combo_counts = {}
    for g, v in val_dss.items():
        cc = {}
        for k, key in enumerate(COMBO_KEYS):
            cc[str(key)] = int((v.hold_combo == COMBO_ARR[k]).all(
                axis=1).sum())
        combo_counts[g] = cc
        log(f'留出 {g} 组合条数: {cc}')

    # 开训前核对 (a)(b)(c)(e)（(d) 烟测后做）
    rep = pretrain_checks(ds, val_dss, keep_items, keep_real_avail)
    (PREP / 'u1c_checks.json').write_text(json.dumps(
        dict(checks=rep, combo_counts=combo_counts, utc=utc()),
        indent=1, default=str) + '\n', encoding='utf-8')

    run_dir = OUT / 't1_train' / TAG
    run_dir.mkdir(parents=True, exist_ok=True)
    from torch.utils.data import WeightedRandomSampler
    all3_keep = np.where((keep_real_avail[:, 1] > 0)
                         & (keep_real_avail[:, 2] > 0))[0]
    w = np.ones(len(ds))
    w[all3_keep] = 2.0
    n_draw = int(w.sum())
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=n_draw, replacement=True)
    loader = DataLoader(ds, batch_size=HP['batch_size'], sampler=sampler,
                        drop_last=True, num_workers=4, pin_memory=True)
    val_loaders = {g: DataLoader(v, batch_size=HP['batch_size'],
                                 shuffle=False, num_workers=2,
                                 pin_memory=True)
                   for g, v in val_dss.items()}
    log(f'过采样: 真实 all3 {len(all3_keep)} 行×2, 每 epoch 抽 {n_draw}')

    band_config = dict(BAND_CONFIG_CONCAT)
    band_config['LIGO'] = dict(BAND_CONFIG_CONCAT['LIGO'])
    band_config['LIGO']['start'] = 514
    model = U1Flow(band_config, num_inputs=9, num_hidden=HP['num_hidden'],
                   num_layers=HP['num_layers'], context_dim=64,
                   in_channels=ds.num_channels).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f'U1Flow params: {nparam:,}')
    optimizer = optim.Adam(model.parameters(), lr=HP['learning_rate'],
                           weight_decay=HP['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=HP['scheduler_T_max'])
    aux_weight, pw = _build_param_weights(
        {'param_emphasis': PARAM_EMPHASIS}, torch.device('cpu'))
    param_weights = pw.to(device) if pw is not None else None

    epochs = 2 if args.smoke else HP['epochs']

    def save_ckpt(path, name, extra=None):
        torch.save(dict(
            model_state_dict=model.state_dict(),
            pta_scalers=ds.pta_scalers, lisa_scalers=ds.lisa_scalers,
            ligo_scalers=ds.ligo_scalers, param_scaler=ds.param_scaler,
            band_config=band_config, num_channels=ds.num_channels,
            use_slope=HP['use_slope'], use_curvature=HP['use_curvature'],
            config=dict(
                name=name, seed=SEED,
                training={k: HP[k] for k in (
                    'epochs', 'batch_size', 'learning_rate', 'weight_decay',
                    'grad_clip', 'scheduler_T_max')},
                model={k: HP[k] for k in (
                    'num_inputs', 'num_hidden', 'num_layers', 'context_dim')},
                noise_curriculum=dict(CURRICULUM),
                noise=noise_config,
                u1c_meta=dict(
                    combo_probs=COMBO_PROBS,
                    param_emphasis=PARAM_EMPHASIS,
                    zero_fill='不在数据集的波段信号=0+纯噪声 (om=-300 占位), '
                              'avail=1; 缺失只表示探测器不在',
                    holdout='R47 3,000 ids 不变; 组合固定种子 20265151 抽一次',
                    scaler_refit='三波段全样本含零信号填充的加噪参考',
                    declared_diffs=[
                        '★ R55: 噪声换未来档 (PTA density_mock / LISA cbe / LIGO CE40 aplus_asd)，纯噪声填充同用未来档',
                        '★ R51 改动 1: 谱延伸不到的波段填纯噪声 (三波段全输入)',
                        '★ R51 改动 2: 探测器组合固定分布抽样 '
                        '(0.40/0.15/0.15/0.15/0.05/0.05/0.05) 替代 drop_p=0.15',
                        '★ scaler 重新拟合: 三波段全样本(含零信号填充)加噪参考',
                        '(承 U1bfix) 1500ep curriculum 0.2 / 过采样×2 / '
                        '选模 ep≥300 / scaler 引用 / 种子 20261008',
                    ]))), path)

    best_val, best_epoch = float('inf'), -1
    grp_best = {g: float('inf') for g in val_loaders}
    grp_best_ep = {g: -1 for g in val_loaders}
    curve = []

    for epoch in range(epochs):
        frac = min(1.0, (epoch + 1) / (HP['epochs'] * CURRICULUM['warmup_frac']))
        noise_scale = 0.05 + (1.0 - 0.05) * frac if frac < 1.0 else 1.0
        ds.refresh_noisy_cache(noise_scale)
        for v in val_dss.values():
            np.random.seed(70000 + epoch)
            v.refresh_noisy_cache(1.0)
        model.train()
        tr_loss = 0.0
        for pta_b, lisa_b, ligo_b, params_b, avail_b in loader:
            pta_b, lisa_b = pta_b.to(device), lisa_b.to(device)
            ligo_b, params_b = ligo_b.to(device), params_b.to(device)
            avail_b = avail_b.to(device)
            lp = model.log_prob(params_b, pta_b, lisa_b, ligo_b, avail_b)
            loss_flow = -torch.mean(lp)
            if aux_weight > 0 and param_weights is not None:
                pred = model.predict_params(pta_b, lisa_b, ligo_b, avail_b)
                loss_aux = torch.mean(param_weights * (pred - params_b) ** 2)
                loss = loss_flow + aux_weight * loss_aux
            else:
                loss = loss_flow
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=HP['grad_clip'])
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()
        model.eval()
        vlosses = {}
        with torch.no_grad():
            for g, vl in val_loaders.items():
                vl_loss = 0.0
                for pta_b, lisa_b, ligo_b, params_b, avail_b in vl:
                    pta_b, lisa_b = pta_b.to(device), lisa_b.to(device)
                    ligo_b, params_b = ligo_b.to(device), params_b.to(device)
                    avail_b = avail_b.to(device)
                    vl_loss += -torch.mean(model.log_prob(
                        params_b, pta_b, lisa_b, ligo_b, avail_b)).item()
                vlosses[g] = vl_loss / len(vl)
        tr_loss /= len(loader)
        v_matched = None
        if not args.smoke and epoch < 300 and epoch % 10 == 0:
            for v in val_dss.values():
                np.random.seed(80000 + epoch)
                v.refresh_noisy_cache(noise_scale)
            vlosses_m = {}
            with torch.no_grad():
                for g, vl in val_loaders.items():
                    vl_loss = 0.0
                    for pta_b, lisa_b, ligo_b, params_b, avail_b in vl:
                        pta_b, lisa_b = pta_b.to(device), lisa_b.to(device)
                        ligo_b, params_b = ligo_b.to(device), params_b.to(device)
                        avail_b = avail_b.to(device)
                        vl_loss += -torch.mean(model.log_prob(
                            params_b, pta_b, lisa_b, ligo_b, avail_b)).item()
                    vlosses_m[g] = vl_loss / len(vl)
            v_matched = vlosses_m
            log(f'  [matched] ep{epoch} NS={noise_scale:.3f} | '
                f'm_pta: {vlosses_m["only_pta"]:.4f} | '
                f'm_pta_lisa: {vlosses_m["pta_lisa"]:.4f} | '
                f'm_all3: {vlosses_m["all3"]:.4f}')
        v_mean = float(np.mean([vlosses[g] for g in val_loaders]))
        if not (np.isfinite(tr_loss) and np.isfinite(v_mean)
                and all(np.isfinite(v) for v in vlosses.values())):
            log(f'NaN/Inf at epoch {epoch} → STOP')
            sys.exit(4)
        mrow = (v_matched['only_pta'], v_matched['pta_lisa'],
                v_matched['all3']) if v_matched else ('', '', '')
        curve.append((epoch, tr_loss, vlosses['only_pta'],
                      vlosses['pta_lisa'], vlosses['all3'], v_mean, *mrow))
        with open(run_dir / 'val_curve.tsv', 'w', encoding='utf-8') as f:
            f.write('epoch\ttrain\tv_only_pta\tv_pta_lisa\tv_all3\tv_mean\t'
                    'm_only_pta\tm_pta_lisa\tm_all3\n')
            for row in curve:
                f.write('\t'.join('' if x == '' else f'{x}' for x in row)
                        + '\n')
        sel_start = 0 if args.smoke else 300
        if epoch >= sel_start:
            if v_mean < best_val:
                best_val, best_epoch = v_mean, epoch
                save_ckpt(run_dir / 'U1c_flow.pt', f'r55_t1_{TAG}_eqmean')
            for g in val_loaders:
                if vlosses[g] < grp_best[g]:
                    grp_best[g], grp_best_ep[g] = vlosses[g], epoch
                    save_ckpt(run_dir / f'U1c_flow_grp_{g}.pt',
                              f'r55_t1_{TAG}_grp_{g}')
        if epoch + 1 in (300, 900, 1500):
            save_ckpt(run_dir / f'U1c_flow_ep{epoch + 1}.pt',
                      f'r55_t1_{TAG}_ep{epoch + 1}')
            log(f'saved periodic ckpt ep{epoch + 1}')
        if epoch % 10 == 0 or epoch < 20 or args.smoke:
            log(f'Epoch {epoch:4d} | Train: {tr_loss:.4f} | v_pta: '
                f'{vlosses["only_pta"]:.4f} | v_pta_lisa: '
                f'{vlosses["pta_lisa"]:.4f} | v_all3: '
                f'{vlosses["all3"]:.4f} | v_mean: {v_mean:.4f} '
                f'(Best: {best_val:.4f}) | NS: {noise_scale:.3f}')

    import hashlib as _h
    def _sha(p):
        h = _h.sha256()
        with open(p, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()
    run_res = dict(tag=TAG, seed=SEED, smoke=bool(args.smoke),
                   n_train=len(ds), n_holdout=3000,
                   best_epoch=best_epoch, best_val=best_val,
                   grp_best={g: dict(val=grp_best[g], epoch=grp_best_ep[g])
                             for g in grp_best},
                   epochs_run=epochs, elapsed_s=time.time() - t0,
                   ckpts={}, combo_probs={str(k): v for k, v in COMBO_PROBS.items()},
                   holdout_combo_seed=20265151,
                   combo_counts=combo_counts,
                   train_py_sha256=_sha(Path(__file__).resolve()),
                   finished_utc=utc())
    for nm, fn in (('eqmean', 'U1c_flow.pt'),
                   ('grp_only_pta', 'U1c_flow_grp_only_pta.pt'),
                   ('grp_pta_lisa', 'U1c_flow_grp_pta_lisa.pt'),
                   ('grp_all3', 'U1c_flow_grp_all3.pt')):
        p = run_dir / fn
        if p.exists():
            run_res['ckpts'][nm] = dict(path=str(p), sha256=_sha(p))
    (run_dir / 'run_result.json').write_text(
        json.dumps(run_res, indent=1) + '\n', encoding='utf-8')
    log(f'DONE smoke={args.smoke} best_epoch={best_epoch} '
        f'best_val={best_val} elapsed={time.time()-t0:.0f}s')


def _build_param_weights(cfg, device):
    """与 r47 driver 同款：param_emphasis 权重（9 参数 MSE 辅助头）。"""
    w = torch.ones(9, device=device)
    return 0.5, w


if __name__ == '__main__':
    main()
