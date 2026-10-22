#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R46 T1 (casdaoGW): U1fix_s1 三波段"并集"联合模型 修正重训.

R45 判词: U1 driver PTA 分支多了 np.abs (train.py 截 1e-300 不取 abs) →
U1_s1 (2ecd2b29) 作废。本 driver = r44_t3_u1_train.py (59505aaf) 一处改动:
  ★ PTA _inject_batch: np.log10(np.maximum(omega + dev*scale, 1e-300)) 无 abs
其余一字不改。开训前 r46_t1_noise_check2.py (R45 T1 同协议) 三波段三缩放
全 <1e-6 才放行。

外部 R44 轮内追加任务单 (原文见 batch-004):
  - 全集 166,127 (三单波段 json 按 mongo id 并集; PTA 恒有)
  - 验证集 = pta_val 33,226 ∪ lisa_val 15,927 ∪ (联合/LIGO) val 9,420 = 52,115
    训练集 = 全集 − 验证并集 = 114,012
    (只有 PTA 69,201 / PTA+LISA 20,746 / 三段都有 24,065 — prep 落盘)
  - 各波段输入取自单波段 json (PTA_fixed_14pts / LISA_fixed_500pts /
    LIGO_fixed_546pts), 不用联合 json 切片; 抽 100 条三段 vs 联合 json
    逐点自检 + LIGO 频率首末 1.30/3.24
  - 噪声 = 原板 J1 (3be078d15a) config: NG15 density_obs + LISA cbe + O3 cc
  - 模型 = 三编码器同 J1 (PTA MLP len14 / LISA CNN+1ResBlock / LIGO CNN+2ResBlock)
    + 每波段可学习缺失向量 (替换编码器输出)
    + context = 三嵌入拼接 + 3 位可用性标记 (64+3=67)
    + 流 = J1 同款 10 层耦合, 隐层 128
  - 标准化器: 波段级只在该波段可用训练样本上按加噪参考拟合; param 全训练集
  - 训练: J1 config (Adam 2e-4/wd 1e-5/clip 5/cosine 300/curriculum/aux 0.1);
    epoch 500; 波段随机丢弃 p=0.3 (至少保 1 段); 验证不丢弃; seed 20261008
  - 烟测 2ep 先行; NaN 停
输出: outputs/r44/t3_train/U1_s1/ + prep_report.json
独立试验, 不进身份总表正式行。
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- release path wiring: optional sibling paths.json (absent = original behavior) ---
_DIR = Path(__file__).resolve().parent
_PFILE = _DIR / 'paths.json'
_P = json.loads(_PFILE.read_text(encoding='utf-8')) if _PFILE.is_file() else {}

def _rp(key, default):
    v = _P.get(key, default)
    return str((_DIR / str(v)).resolve())
WS = Path(_rp('workspace', '/root/SageNetR/workspace'))
sys.path.insert(0, str(WS / 'new'))

from train import (BandEncoder, ConditionalCouplingLayer, BaseDataset,
                   BAND_CONFIG_CONCAT, PARAM_CONFIG, _build_param_weights,
                   _get_curriculum_scale)  # noqa: E402

OUT = WS / 'outputs/r47/t1_train'
OUT.mkdir(parents=True, exist_ok=True)
PREP = WS / 'outputs/r47/t1_prep'
PREP.mkdir(parents=True, exist_ok=True)

PTA_JSON = WS / 'dataset/single/PTA_fixed_14pts.json'
LISA_JSON = WS / 'dataset/single/LISA_fixed_500pts.json'
LIGO_JSON = WS / 'dataset/single/LIGO_fixed_546pts.json'
JOINT_JSON = WS / 'dataset/Dataset_PTA_LISA_LIGO.json'
J1_CKPT = Path(_rp('r40_refs', '/root/SageNetR/r40_refs')) / 'J1_flow_joint.pt'
J1_SHA = '3be078d15a'
NOISE_PTA_DIR = WS / 'new/noise/NANOGrav15yr'
LIGO_NOISE_FILE = WS / 'new/noise/C_O1_O2_O3.dat.txt'

HP = dict(epochs=1500,  # R47 改动 1 (500→1500; curriculum 0.2 比例不变)
          batch_size=256, learning_rate=2e-4, weight_decay=1e-5,
          grad_clip=5.0, scheduler_T_max=300,
          num_inputs=9, num_hidden=128, num_layers=10, context_dim=64,
          use_slope=True, use_curvature=True)
CURRICULUM = dict(enabled=True, warmup_fraction=0.2, start_scale=0.05, end_scale=1.0)
PARAM_EMPHASIS = dict(enabled=True, aux_weight=0.1, weights={'T_re': 3.0, 'DN_re': 0.3})
NOISE = dict(
    pta=dict(mode='density_obs', noise_dir=str(NOISE_PTA_DIR), per_sample_noise=True,
             noise_scale=1.0, n_bins_use=14),
    lisa=dict(enabled=True, noise_source='analytical', noise_preset='cbe',
              noise_scale=1.0, injection_mode='physical', logscale_base=1.0,
              T_obs=4.0, include_confusion=True),
    ligo=dict(injection_mode='physical', noise_type='cc_spectrum',
              noise_file=str(LIGO_NOISE_FILE), noise_scale=1.0, logscale_base=1.0),
)
SEED = 20261008
DROP_P = 0.15   # R47 改动 3
EXPECT = dict(full=166127, val_union=52115, train=114012,
              only_pta=69201, pta_lisa=20746, all3=24065,
              pta_val=33226, lisa_val=15927, ligo_val=9420)


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    line = f'[{utc()}] {msg}'
    print(line, flush=True)
    with open(OUT / 'u1_train.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# ---------------- 数据准备 ----------------

def load_jsons():
    log('loading 3 single-band jsons + joint ...')
    with open(PTA_JSON) as f:
        pta = json.load(f)
    with open(LISA_JSON) as f:
        lisa = json.load(f)
    with open(LIGO_JSON) as f:
        ligo = json.load(f)
    return pta, lisa, ligo


def build_split(pta, lisa, ligo):
    """验证并集 (mongo id 键) → 训练集 + 组合计数; 结果落盘 prep。"""
    pta_ids = [str(x['_id']) for x in pta]
    lisa_ids = [str(x['_id']) for x in lisa]
    ligo_ids = [str(x['_id']) for x in ligo]
    _, pv = train_test_split(np.arange(len(pta)), test_size=0.2, random_state=42)
    _, lv = train_test_split(np.arange(len(lisa)), test_size=0.2, random_state=42)
    _, gv = train_test_split(np.arange(len(ligo)), test_size=0.2, random_state=42)
    val_union = (set(pta_ids[i] for i in pv) | set(lisa_ids[i] for i in lv)
                 | set(ligo_ids[i] for i in gv))
    sp, sl, sg = set(pta_ids), set(lisa_ids), set(ligo_ids)
    full = sp | sl | sg
    train_ids = full - val_union
    only_pta = train_ids & (sp - sl - sg)
    pta_lisa = train_ids & ((sp & sl) - sg)
    all3 = train_ids & (sp & sl & sg)
    counts = dict(full=len(full), val_union=len(val_union), train=len(train_ids),
                  only_pta=len(only_pta), pta_lisa=len(pta_lisa), all3=len(all3),
                  pta_val=len(pv), lisa_val=len(lv), ligo_val=len(gv))
    for k in EXPECT:
        assert counts[k] == EXPECT[k], f'{k}: {counts[k]} != {EXPECT[k]}'
    log(f'split: 全集 {counts["full"]} / val并集 {counts["val_union"]} '
        f'/ 训练 {counts["train"]} (只PTA {counts["only_pta"]} / '
        f'PTA+LISA {counts["pta_lisa"]} / 三段 {counts["all3"]}) — 与任务单预核一致')
    return dict(train_ids=sorted(train_ids), val_ids=sorted(val_union)), counts


def spotcheck(pta, lisa, ligo):
    """抽 100 条联合样本: 三单波段 json 段 vs 联合 json (LIGO start 514) 逐点。"""
    with open(JOINT_JSON) as f:
        joint = json.load(f)
    pta_by = {str(x['_id']): x for x in pta}
    lisa_by = {str(x['_id']): x for x in lisa}
    ligo_by = {str(x['_id']): x for x in ligo}
    rng = np.random.default_rng(20261008)
    idx = rng.choice(len(joint), size=100, replace=False)
    max_d = 0.0
    lig_f_first, lig_f_last = None, None
    for i in idx:
        it = joint[int(i)]
        oid = str(it['_id'])
        jo = np.array(it['log10OmegaGW'], dtype=np.float64)
        jf = np.array(it['f'], dtype=np.float64)
        po = np.array(pta_by[oid]['log10OmegaGW'], dtype=np.float64)
        lo = np.array(lisa_by[oid]['log10OmegaGW'], dtype=np.float64)
        go = np.array(ligo_by[oid]['log10OmegaGW'], dtype=np.float64)
        gf = np.array(ligo_by[oid]['f'], dtype=np.float64)
        lig_f_first, lig_f_last = float(gf[0]), float(gf[-1])
        # 联合切法: PTA [0:14], LISA [14:514], LIGO [514:1060]
        max_d = max(max_d,
                    float(np.max(np.abs(po - jo[0:14]))),
                    float(np.max(np.abs(lo - jo[14:514]))),
                    float(np.max(np.abs(go - jo[514:1060]))))
    assert max_d == 0.0, f'单波段 vs 联合自检不一致: {max_d:.3e}'
    assert abs(lig_f_first - 1.30) < 5e-3 and abs(lig_f_last - 3.24) < 5e-3, \
        f'LIGO 频率首末 {lig_f_first}/{lig_f_last} != 1.30/3.24'
    log(f'自检: 100 条三段 vs 联合逐点 max|Δ|=0 (bitwise); '
        f'LIGO f 首末 {lig_f_first:.4f}/{lig_f_last:.4f}')
    return dict(n=100, max_abs_diff=max_d, ligo_f=[lig_f_first, lig_f_last])


# ---------------- U1 模型 ----------------

class U1ContextEncoder(nn.Module):
    """三段嵌入 + 缺失向量替换 + 3 位可用性标记 → 67 维 context。"""

    def __init__(self, band_config, context_dim=64, in_channels=4):
        super().__init__()
        per = context_dim // 3
        rem = context_dim - per * 3
        self.pta_encoder = BandEncoder(in_channels, 32, per, 'pta',
                                       band_config['PTA']['len'])
        self.lisa_encoder = BandEncoder(in_channels, 64, per, 'lisa',
                                        band_config['LISA']['len'])
        self.ligo_encoder = BandEncoder(in_channels, 64, per + rem, 'ligo',
                                        band_config['LIGO']['len'])
        # 可学习缺失向量 (= 各编码器输出维度)
        self.missing_pta = nn.Parameter(torch.zeros(per))
        self.missing_lisa = nn.Parameter(torch.zeros(per))
        self.missing_ligo = nn.Parameter(torch.zeros(per + rem))
        nn.init.normal_(self.missing_pta, std=0.01)
        nn.init.normal_(self.missing_lisa, std=0.01)
        nn.init.normal_(self.missing_ligo, std=0.01)

    def forward(self, pta, lisa, ligo, avail):
        """avail: (B, 3) 1=可用 0=缺失 (pta, lisa, ligo)。"""
        h_pta = self.pta_encoder(pta)
        h_lisa = self.lisa_encoder(lisa)
        h_ligo = self.ligo_encoder(ligo)
        a = avail[:, 0:1]
        h_pta = a * h_pta + (1 - a) * self.missing_pta
        a = avail[:, 1:2]
        h_lisa = a * h_lisa + (1 - a) * self.missing_lisa
        a = avail[:, 2:3]
        h_ligo = a * h_ligo + (1 - a) * self.missing_ligo
        return torch.cat([h_pta, h_lisa, h_ligo, avail], dim=1)


class U1Flow(nn.Module):
    """流部分同 MultiBandFlow, 但 context 67 维 + avail 接口。"""

    def __init__(self, band_config, num_inputs=9, num_hidden=128,
                 num_layers=10, context_dim=64, in_channels=4):
        super().__init__()
        self.num_inputs = num_inputs
        self.context_encoder = U1ContextEncoder(band_config, context_dim,
                                                in_channels)
        ctx = context_dim + 3
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask = torch.zeros(num_inputs)
            if i % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
            self.layers.append(ConditionalCouplingLayer(
                num_inputs, num_hidden, mask, ctx))
        self.param_predictor = nn.Sequential(
            nn.Linear(ctx, ctx), nn.ReLU(), nn.Linear(ctx, num_inputs))

    def _get_context(self, pta, lisa, ligo, avail):
        return self.context_encoder(pta, lisa, ligo, avail)

    def log_prob(self, inputs, pta, lisa, ligo, avail):
        context = self._get_context(pta, lisa, ligo, avail)
        log_det_sum = 0
        z = inputs
        for layer in self.layers:
            z, log_det = layer(z, context, mode='direct')
            log_det_sum += log_det
        log_p_z = -0.5 * (np.log(2 * np.pi) + z.pow(2)).sum(dim=1)
        return log_p_z + log_det_sum

    def predict_params(self, pta, lisa, ligo, avail):
        context = self._get_context(pta, lisa, ligo, avail)
        return self.param_predictor(context)

    def sample(self, pta, lisa, ligo, avail, num_samples=1):
        context = self._get_context(pta, lisa, ligo, avail)
        B = context.shape[0]
        context_rep = context.unsqueeze(1).repeat(
            1, num_samples, 1).reshape(B * num_samples, -1)
        z = torch.randn(B * num_samples, self.num_inputs, device=pta.device)
        for layer in reversed(self.layers):
            z, _ = layer(z, context_rep, mode='inverse')
        return z.reshape(B, num_samples, -1)


# ---------------- 数据集 (driver 侧) ----------------

class U1Dataset(Dataset):
    """并集数据集: 每条样本可变可用性; 曲线取自单波段 json;
    物理噪声 per-band (train.py 模型类复用); 波段随机丢弃在 __getitem__。"""

    def __init__(self, items, avail_mask, noise_config, use_slope=True,
                 use_curvature=True, train=True, drop_p=0.3):
        """items: list of dict(_id, band items); avail_mask (N,3) float。
        曲线预计算: 各 band 的 raw 特征 + 占位 (缺失波段填 0)。"""
        self.use_slope = use_slope
        self.use_curvature = use_curvature
        self.train = train
        self.drop_p = drop_p if train else 0.0

        base = BaseDataset(noise_config, use_slope, use_curvature)
        self.compute_features = base.compute_features
        self.extract_params = base.extract_params

        # ---- 曲线特征 (每 band 只算可用样本; 缺失填 0) ----
        bc = BAND_CONFIG_CONCAT
        N = len(items)
        self.pta_raw = np.zeros((N, 4, bc['PTA']['len']), dtype=np.float32)
        self.lisa_raw = np.zeros((N, 4, bc['LISA']['len']), dtype=np.float32)
        self.ligo_raw = np.zeros((N, 4, bc['LIGO']['len']), dtype=np.float32)
        for i, it in enumerate(items):
            if avail_mask[i, 0]:
                om = np.array(it['pta']['log10OmegaGW'], dtype=np.float32)
                fr = np.array(it['pta']['f'], dtype=np.float32)
                self.pta_raw[i] = base.compute_features(om, fr)
            if avail_mask[i, 1]:
                om = np.array(it['lisa']['log10OmegaGW'], dtype=np.float32)
                fr = np.array(it['lisa']['f'], dtype=np.float32)
                self.lisa_raw[i] = base.compute_features(om, fr)
            if avail_mask[i, 2]:
                om = np.array(it['ligo']['log10OmegaGW'], dtype=np.float32)
                fr = np.array(it['ligo']['f'], dtype=np.float32)
                self.ligo_raw[i] = base.compute_features(om, fr)
        self.num_channels = 4
        self.avail = avail_mask.astype(np.float32)

        # ---- 噪声模型 (复用 train.py 构建函数; 频格取样本实值) ----
        # R46 eval 修复: 单波段评估 (E1 只 PTA / E3 只 LIGO) 时该行可能无可用
        # 样本 → freq_grid 改从 items 的 json 条目直接取 (频格是波段固有属性,
        # 与可用性无关; 三 json 同源 canonical, 任取一行即代表该波段频格)
        def _band_freq(key, ln):
            for it in items:
                b = it.get(key)
                if b is not None:
                    return np.array(b['f'], dtype=np.float32)[:ln]
            # items 全缺该波段 (E1/E3 单波段评估): 从波段 json 首行取频格
            # (频格是波段固有属性; 噪声模型只需 grid, 与样本无关)
            import json as _json
            pj = {'pta': str(WS / 'dataset/single/PTA_fixed_14pts.json'),
                  'lisa': str(WS / 'dataset/single/LISA_fixed_500pts.json'),
                  'ligo': str(WS / 'dataset/single/LIGO_fixed_546pts.json')}[key]
            with open(pj) as fh:
                first = _json.load(fh)[0]
            return np.array(first['f'], dtype=np.float32)[:ln]
        pta_freq = _band_freq('pta', BAND_CONFIG_CONCAT['PTA']['len'])
        self.pta_noise = base._build_pta_noise_model(noise_config)
        if self.pta_noise is not None and hasattr(self.pta_noise, 'validate_freqs'):
            self.pta_noise.validate_freqs(pta_freq)
        self.lisa_freq_grid = _band_freq('lisa', BAND_CONFIG_CONCAT['LISA']['len'])
        self.lisa_noise = base._build_lisa_noise_model(noise_config, self.lisa_freq_grid)
        self.ligo_freq_grid = _band_freq('ligo', BAND_CONFIG_CONCAT['LIGO']['len'])
        self.ligo_noise = base._build_ligo_noise_model(noise_config, self.ligo_freq_grid)

        # ---- 标准化器: 波段级只在该波段可用样本上、按加噪参考拟合 ----
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
            # eval: scaler 全部由 ckpt 覆写 (load_scalers_into), 构造期跳过
            # fit — 单波段评估时缺失波段无可用样本, fit 空集必炸
            self.params = np.zeros_like(self.params_raw)
        self._cache = {}
        print(f'U1Dataset: {N} samples; avail mean = {avail_mask.mean(axis=0)}')

    def _noisy_ref(self, raw, model, band, max_n=4000, seed=12345):
        """加噪参考 (波段可用样本; 上限 4,000 条固定种子子集 — 全量 11 万条
        逐条 inject 过慢, 4k 子集 scaler 统计量已稳定; 差异声明在 config)。"""
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
        for band, raw, avail, model, scalers, ln in (
                ('pta', self.pta_raw, self.avail[:, 0], self.pta_noise,
                 self.pta_scalers, BAND_CONFIG_CONCAT['PTA']['len']),
                ('lisa', self.lisa_raw, self.avail[:, 1], self.lisa_noise,
                 self.lisa_scalers, BAND_CONFIG_CONCAT['LISA']['len']),
                ('ligo', self.ligo_raw, self.avail[:, 2], self.ligo_noise,
                 self.ligo_scalers, BAND_CONFIG_CONCAT['LIGO']['len'])):
            sub = raw[avail > 0]
            print(f'[{band.upper()} Noise] fitting scalers on '
                  f'{len(sub)} available samples (noise-injected ref)...')
            ref = self._noisy_ref(sub, model, band)
            for c in range(4):
                ch = ref[:, c, :].reshape(-1, 1)
                scalers[c].fit(ch)
        # 标准化占位曲线 (缺失波段全 0 不参与)

    def refresh_noisy_cache(self, noise_scale_override=1.0):
        """每 epoch: 向量化重生成三段含噪 (只算可用样本, 缺失填 0)."""
        import numpy as _np
        for band, raw, avail, model, scalers in (
                ('PTA', self.pta_raw, self.avail[:, 0], self.pta_noise,
                 self.pta_scalers),
                ('LISA', self.lisa_raw, self.avail[:, 1], self.lisa_noise,
                 self.lisa_scalers),
                ('LIGO', self.ligo_raw, self.avail[:, 2], self.ligo_noise,
                 self.ligo_scalers)):
            if model is None:
                self._cache[band] = None
                continue
            mask = avail > 0
            sub = raw[mask]
            if len(sub) == 0:
                # 单波段评估: 该波段全缺失 → cache 置零占位 (曲线值无影响,
                # 编码器输出已被缺失向量替换)
                self._cache[band] = np.zeros(raw.shape, dtype=np.float32)
                continue
            om = sub[:, 0, :].copy()
            fr = sub[0, 1, :]
            # 复用 BaseDataset 的批注入 (通过临时实例)
            omega_noisy = self._inject_batch(om, model, band,
                                             noise_scale_override)
            feats = self._features_batch(omega_noisy, fr)
            scaled = np.empty_like(feats)
            for c in range(4):
                scaled[:, c, :] = scalers[c].transform(
                    feats[:, c, :].reshape(-1, 1)).reshape(feats.shape[0], -1)
            full = np.zeros(raw.shape, dtype=np.float32)
            full[mask] = scaled.astype(np.float32)
            self._cache[band] = full

    def _inject_batch(self, om, model, band, scale):
        """向量化注入 (physical 数学同 train.py):
        PTA: Ω_dev = sample_omega_noise (零均值偏差); curriculum scale 线性缩放偏差
        LISA/LIGO: noise ~ N(0, sigma_omega·scale); log10|Ω+noise|"""
        if band == 'PTA':
            # ★ R46 T1 修正 (判词): train.py 同款, 不取 abs, 负和直接截 1e-300
            dev = model.sample_omega_noise(om.shape[0])   # (N,14) 零均值
            omega = 10.0 ** om.astype(np.float64)
            obs = omega + dev * scale
            return np.log10(np.maximum(obs, 1e-300)).astype(np.float32)
        sig = np.asarray(model.sigma_omega, dtype=np.float64) * scale
        omega = 10.0 ** om.astype(np.float64)
        noise = np.random.normal(0.0, 1.0, om.shape) * sig[None, :]
        obs = np.abs(omega + noise)
        return np.log10(np.maximum(obs, 1e-300)).astype(np.float32)

    def _features_batch(self, om_noisy, freq):
        out = np.empty((om_noisy.shape[0], self.num_channels, om_noisy.shape[1]),
                       dtype=np.float32)
        for i in range(om_noisy.shape[0]):
            out[i] = self.compute_features(om_noisy[i], freq)
        return out

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        avail = self.avail[idx].copy()  # (3,)
        if self.train and self.drop_p > 0:
            drop = np.random.random(3) < self.drop_p
            newa = avail * (1 - drop.astype(np.float32))
            if newa.sum() < 1:
                # 至少保一段: 随机恢复一个原本可用的波段
                cand = np.where(avail > 0)[0]
                keep = cand[np.random.randint(len(cand))]
                newa = avail.copy()
                newa[:] = 0
                newa[keep] = 1.0
            avail = newa
        pta = self._cache['PTA'][idx].copy() if self._cache.get('PTA') is not None \
            else np.zeros_like(self.pta_raw[idx])
        lisa = self._cache['LISA'][idx].copy() if self._cache.get('LISA') is not None \
            else np.zeros_like(self.lisa_raw[idx])
        ligo = self._cache['LIGO'][idx].copy() if self._cache.get('LIGO') is not None \
            else np.zeros_like(self.ligo_raw[idx])
        # 缺失波段曲线置 0 (编码器输出被缺失向量替换, 曲线值无影响但保持确定性)
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


def make_union_items(pta, lisa, ligo, id_list):
    """按 id 列表组装 U1 条目: {id, params, pta?, lisa?, ligo?}。"""
    pta_by = {str(x['_id']): x for x in pta}
    lisa_by = {str(x['_id']): x for x in lisa}
    ligo_by = {str(x['_id']): x for x in ligo}
    items, avails = [], []
    for oid in id_list:
        p = pta_by.get(oid)
        l = lisa_by.get(oid)
        g = ligo_by.get(oid)
        assert p is not None, f'{oid} 无 PTA (任务单: PTA 恒有)'
        items.append(dict(_id=oid, params=p, pta=p,
                          lisa=l if l else None,
                          ligo=g if g else None))
        avails.append([1.0, 1.0 if l else 0.0, 1.0 if g else 0.0])
    return items, np.array(avails, dtype=np.float32)


# ---------------- 训练 ----------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # J1 母本 config diff (新规: 贴条数 + diff)
    j1_sha = sha256_file(J1_CKPT)
    assert j1_sha.startswith(J1_SHA), j1_sha
    ck = torch.load(J1_CKPT, map_location='cpu', weights_only=False)
    jcfg = ck['config']
    # 与 J1 config 逐项 diff: 训练超参一致; 噪声三段一致 (路径豁免)
    assert jcfg['training']['batch_size'] == HP['batch_size']
    assert abs(jcfg['training']['learning_rate'] - HP['learning_rate']) < 1e-12
    assert abs(jcfg['training']['weight_decay'] - HP['weight_decay']) < 1e-12
    assert jcfg['training']['grad_clip'] == HP['grad_clip']
    assert jcfg['training']['scheduler_T_max'] == HP['scheduler_T_max']
    # epochs: U1 用 500 (任务单), J1 实跑 1500 — 声明差异
    # 噪声: 三段逐项 (路径豁免)
    for band in ('pta', 'lisa'):
        a = dict(jcfg['noise'][band])
        b = dict(NOISE[band])
        a.pop('noise_dir', None); b.pop('noise_dir', None)
        assert a == b, f'{band} noise diff: {a} vs {b}'
    a = dict(jcfg['noise']['ligo']); a.pop('noise_file')
    b = dict(NOISE['ligo']); b.pop('noise_file')
    assert a == b, 'ligo noise diff'
    log(f'J1 母本 ({J1_SHA}) config diff: 训练超参/噪声逐项一致 (路径豁免)。'
        f'声明差异全清单: (1) epochs 1500→500 (任务单); (2) context 64→67; '
        f'(3) 每波段可学习缺失向量; (4) 波段随机丢弃 p=0.3; '
        f'(5) 数据并集切分 train 114,012 / val 并集 52,115; '
        f'(6) 波段 scaler 4,000 条子集拟合; '
        f'(7) ★ 模型选择 = 2,000 条训练集 holdout (非验证并集); '
        f'(8) ★ PTA 加噪修正无 abs (train.py 同款, R45 判词)')

    set_seed(SEED)
    t0 = time.time()
    pta, lisa, ligo = load_jsons()
    split, counts = build_split(pta, lisa, ligo)
    spot = spotcheck(pta, lisa, ligo)
    (PREP / 'u1_prep_report.json').write_text(json.dumps(
        dict(counts=counts, spotcheck=spot, seed=SEED, utc=utc(),
             train_sample=list(counts.items())), indent=1) + '\n',
        encoding='utf-8')

    train_items, train_avail = make_union_items(pta, lisa, ligo, split['train_ids'])
    del pta, lisa, ligo
    gc.collect()

    noise_config = {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0,
                    'glitch_prob': 0.0,
                    'pta': dict(NOISE['pta']), 'lisa': dict(NOISE['lisa']),
                    'ligo': dict(NOISE['ligo'])}
    # R47 改动 2: 留出 = 按三种可用性各 1,000 条共 3,000 (不参训)
    rng = np.random.default_rng(20261008)
    grp_idx = {'only_pta': np.where((train_avail[:, 1] == 0) & (train_avail[:, 2] == 0))[0],
               'pta_lisa': np.where((train_avail[:, 1] > 0) & (train_avail[:, 2] == 0))[0],
               'all3': np.where(train_avail[:, 2] > 0)[0]}
    hold_idx = {}
    for g, ix in grp_idx.items():
        hold_idx[g] = rng.choice(ix, size=1000, replace=False)
    hold_all = np.concatenate(list(hold_idx.values()))
    assert len(np.unique(hold_all)) == 3000, '留出重叠'
    hold_mask = np.zeros(len(train_items), dtype=bool); hold_mask[hold_all] = True
    # 训练条目 = 原训练集 - 留出 (111,012)
    keep_items = [it for it, m in zip(train_items, ~hold_mask) if m]
    keep_avail = train_avail[~hold_mask]
    log(f'R47 留出: only_pta 1,000 + pta_lisa 1,000 + all3 1,000 = 3,000 '
        f'(不参训); 实际训练 {len(keep_items)}')

    ds = U1Dataset(keep_items, keep_avail, noise_config,
                   HP['use_slope'], HP['use_curvature'], train=True,
                   drop_p=DROP_P)

    run_dir = OUT / 'U1b_s1'
    run_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ds, batch_size=HP['batch_size'], shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)

    band_config = BAND_CONFIG_CONCAT
    band_config['LIGO']['start'] = 514  # 与 J1 一致 (联合切法; U1 曲线来自
    # 单波段 json, 该 start 只影响 band_config 内嵌记录, 输入已由 json 给定)
    model = U1Flow(band_config, num_inputs=HP['num_inputs'],
                   num_hidden=HP['num_hidden'], num_layers=HP['num_layers'],
                   context_dim=HP['context_dim'],
                   in_channels=ds.num_channels).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f'U1Flow params: {nparam:,}')

    optimizer = optim.Adam(model.parameters(), lr=HP['learning_rate'],
                           weight_decay=HP['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=HP['scheduler_T_max'])
    aux_weight, pw_cpu = _build_param_weights(
        {'param_emphasis': PARAM_EMPHASIS}, torch.device('cpu'))
    param_weights = pw_cpu.to(device) if pw_cpu is not None else None
    curriculum_cfg = dict(CURRICULUM)

    epochs = 2 if args.smoke else HP['epochs']
    # 三组留出 dataset: train=False (不丢弃) 但 scaler 从训练 ds 复制
    # (训练期无 ckpt 可覆写; R46 eval 路径的跳 fit 是给评估脚本用的)
    val_dss = {g: U1Dataset([train_items[i] for i in ix],
                            train_avail[ix], noise_config,
                            HP['use_slope'], HP['use_curvature'], train=False)
               for g, ix in hold_idx.items()}
    for v in val_dss.values():
        v.pta_scalers = [type(s)() for s in ds.pta_scalers]
        for c in range(4):
            v.pta_scalers[c].fit(ds.pta_raw[ds.avail[:, 0] > 0][:, c, :].reshape(-1, 1))
        v.lisa_scalers = [type(s)() for s in ds.lisa_scalers]
        for c in range(4):
            v.lisa_scalers[c].fit(ds.lisa_raw[ds.avail[:, 1] > 0][:, c, :].reshape(-1, 1))
        v.ligo_scalers = [type(s)() for s in ds.ligo_scalers]
        for c in range(4):
            v.ligo_scalers[c].fit(ds.ligo_raw[ds.avail[:, 2] > 0][:, c, :].reshape(-1, 1))
        v.param_scaler = type(ds.param_scaler)()
        v.param_scaler.fit(ds.params_raw)
        v.params = v.param_scaler.transform(v.params_raw)
    val_loaders = {g: DataLoader(v, batch_size=HP['batch_size'], shuffle=False,
                                 num_workers=2, pin_memory=True)
                   for g, v in val_dss.items()}
    # R47 改动 4: 三段都有样本每轮过采样 ×2 — 用 WeightedRandomSample 等价实现:
    # 每 epoch 采样权重 all3 行=2 其余=1 (总数≈原训练集 + all3 行数)
    all3_keep = np.where((keep_avail[:, 1] > 0) & (keep_avail[:, 2] > 0))[0]
    w = np.ones(len(ds)); w[all3_keep] = 2.0
    # 用自定义 sampler 每 epoch 重抽 (shuffle 等价, drop_last 兼容)
    from torch.utils.data import WeightedRandomSampler
    n_draw = int(w.sum())
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=n_draw, replacement=True)
    loader = DataLoader(ds, batch_size=HP['batch_size'], sampler=sampler,
                        drop_last=True, num_workers=4, pin_memory=True)
    log(f'R47 过采样: all3 训练行 {len(all3_keep)} 权重×2, 每 epoch 抽 '
        f'{n_draw} 条 (原 {len(ds)})')

    log(f'start U1b training: epochs={epochs} drop_p={DROP_P} seed={SEED} '
        f'train={len(ds)} (+all3 过采样) | 留出 3x1,000 | '
        f'4 ckpt (等权平均 + 三组各一)')

    def save_ckpt(path, name):
        torch.save(dict(
            model_state_dict=model.state_dict(),
            pta_scalers=ds.pta_scalers, lisa_scalers=ds.lisa_scalers,
            ligo_scalers=ds.ligo_scalers, param_scaler=ds.param_scaler,
            band_config=band_config, num_channels=ds.num_channels,
            use_slope=HP['use_slope'], use_curvature=HP['use_curvature'],
            config=dict(name=name, seed=SEED,
                        training={k: HP[k] for k in (
                            'epochs', 'batch_size', 'learning_rate',
                            'weight_decay', 'grad_clip', 'scheduler_T_max')},
                        model={k: HP[k] for k in (
                            'num_inputs', 'num_hidden', 'num_layers',
                            'context_dim')},
                        noise_curriculum=curriculum_cfg,
                        param_emphasis=PARAM_EMPHASIS,
                        noise=noise_config,
                        u1_meta=dict(
                            drop_p=DROP_P, context_dim_plus3=67,
                            missing_vectors='learnable per band',
                            dataset='union 166,127; train 111,012 (留出 3,000 不参训)',
                            mother=f'J1 {J1_SHA} config',
                            declared_diffs=[
                                '★ R47 改动 1: epochs 500→1500 (curriculum 0.2 比例不变)',
                                '★ R47 改动 2: 留出按三种可用性各 1,000 共 3,000; 选模型=三组 NLL 等权平均; 4 ckpt',
                                '★ R47 改动 3: drop 0.3→0.15',
                                '★ R47 改动 4: 三段都有训练样本过采样 x2 (WeightedRandomSampler; 留出不过采样)',
                                '(承 R46) context 64→67 / 缺失向量 / 并集切分 / scaler 4k 子集 / PTA 加噪无 abs',
                            ])),
        ), path)

    best_val = float('inf')
    best_epoch = -1
    grp_best = {g: float('inf') for g in val_loaders}
    grp_best_ep = {g: -1 for g in val_loaders}
    curve = []  # (epoch, tr, v_only_pta, v_pta_lisa, v_all3, v_mean)

    for epoch in range(epochs):
        noise_scale = _get_curriculum_scale(epoch, HP['epochs'], curriculum_cfg)
        ds.refresh_noisy_cache(noise_scale)
        for v in val_dss.values():
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
        v_mean = float(np.mean([vlosses[g] for g in val_loaders]))
        if not (np.isfinite(tr_loss) and np.isfinite(v_mean)
                and all(np.isfinite(v) for v in vlosses.values())):
            log(f'NaN/Inf at epoch {epoch} → STOP')
            sys.exit(4)
        curve.append((epoch, tr_loss, vlosses['only_pta'], vlosses['pta_lisa'],
                      vlosses['all3'], v_mean))
        if v_mean < best_val:
            best_val = v_mean
            best_epoch = epoch
            save_ckpt(run_dir / 'U1b_flow.pt', 'r47_t1_U1b_s1_eqmean')
        for g in val_loaders:
            if vlosses[g] < grp_best[g]:
                grp_best[g] = vlosses[g]
                grp_best_ep[g] = epoch
                save_ckpt(run_dir / f'U1b_flow_grp_{g}.pt',
                          f'r47_t1_U1b_s1_grp_{g}')
        if epoch % 10 == 0 or epoch < 20 or args.smoke:
            log(f'Epoch {epoch:4d} | Train: {tr_loss:.4f} | v_pta: '
                f'{vlosses["only_pta"]:.4f} | v_pta_lisa: {vlosses["pta_lisa"]:.4f} '
                f'| v_all3: {vlosses["all3"]:.4f} | v_mean: {v_mean:.4f} '
                f'(Best: {best_val:.4f}) | NS: {noise_scale:.3f}')

    elapsed = round(time.time() - t0, 1)
    per_ep = elapsed / epochs
    est_h = per_ep * HP['epochs'] / 3600
    log(f'DONE smoke={args.smoke} best_epoch={best_epoch} best_val={best_val:.6f} '
        f'elapsed={elapsed}s ({per_ep:.1f}s/ep; {HP["epochs"]}ep 估 {est_h:.1f}h)')
    with open(run_dir / 'val_curve.tsv', 'w', encoding='utf-8') as f:
        f.write('epoch\ttrain\tv_only_pta\tv_pta_lisa\tv_all3\tv_mean\n')
        for row in curve:
            if row[0] % 10 == 0 or abs(row[0] - best_epoch) <= 2:
                f.write('\t'.join(str(x) if not isinstance(x, float)
                                   else f'{x:.6f}' for x in row) + '\n')
    if args.smoke:
        (OUT / 'u1b_smoke_result.json').write_text(json.dumps(dict(
            epochs=2, best_val=best_val, elapsed_s=elapsed,
            per_ep_s=per_ep, est_hours=est_h, utc=utc()),
            indent=1) + '\n', encoding='utf-8')
        if est_h > 24:
            log(f'⚠ 估时 {est_h:.1f}h > 24h — 按任务单先报')
    else:
        import os as _os
        ckpts = {}
        for nm, fn, val, ep in (
                ('eqmean', 'U1b_flow.pt', best_val, best_epoch),
                ('grp_only_pta', 'U1b_flow_grp_only_pta.pt',
                 grp_best['only_pta'], grp_best_ep['only_pta']),
                ('grp_pta_lisa', 'U1b_flow_grp_pta_lisa.pt',
                 grp_best['pta_lisa'], grp_best_ep['pta_lisa']),
                ('grp_all3', 'U1b_flow_grp_all3.pt',
                 grp_best['all3'], grp_best_ep['all3'])):
            p = run_dir / fn
            ckpts[nm] = dict(path=str(p),
                             sha256=sha256_file(p) if p.is_file() else None,
                             best_val=val, best_epoch=int(ep))
        (run_dir / 'run_result.json').write_text(json.dumps(dict(
            tag='U1b_s1', seed=SEED, smoke=False, n_train=len(ds),
            n_holdout=3000,
            best_epoch=int(best_epoch), best_val=best_val,
            grp_best={g: dict(val=grp_best[g], epoch=int(grp_best_ep[g]))
                      for g in grp_best},
            epochs_run=epochs, elapsed_s=elapsed, ckpts=ckpts,
            train_py_sha256=sha256_file(WS / 'new/train.py'),
            driver_sha256=sha256_file(_os.path.abspath(__file__)),
            finished_utc=utc()), indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
