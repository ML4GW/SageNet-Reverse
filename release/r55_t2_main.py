#!/usr/bin/env python
# R55 T2: U1c 论文主表——验证集并集 52,115（pta_val∪lisa_val∪联合 val，mongo id 去重），
# 每条三探测器全给（avail=1,1,1），真实延伸不到的波段按 U1c 规则填纯噪声（ZERO_OM 占位+加噪）。
# 分档=合成 SNR=sqrt(SNR_PTA^2+SNR_LISA^2+SNR_LIGO^2)（r40 定义：干净曲线 sqrt sum (x/sigma)^2；
#   真实延伸不到的波段信号=0 -> 该段 SNR=0）；另按真实可用性拆子表。
# 指标=区间/先验（W_pop 同 r54_popW）、覆盖、R2，5 参数。
# 自检=联合 val 9,420 id 子集 vs r54_popW U1c E4 合成 SNR 分档行（d宽<=0.02，d覆盖<=0.03）。
# v2（R55 中报判词 2）：6 次噪声实现取均值+min–max 波动表；自检=r54 E4 单次值 ∈ [min−0.01, max+0.01]
# 用法: python r55_t2_main.py --tier now|fut
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
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
from r47_t1_u1b_train import U1Flow  # noqa: E402
from r51_u1c_train import (U1cDataset, ZERO_OM, PTA_JSON, LISA_JSON,  # noqa: E402
                           LIGO_JSON)

SNR_TABLE = WS / 'outputs/r40/t4/r40_snr_table.npz'
JOINT_SNR = WS / 'outputs/r41/t3/joint_snr_components.npz'
POPW = np.array([34.744, 5.008, 8.963, 8.745, 35.451])  # r54_popW W_pop
PARAMS = ['log10 r', 'n_t', 'log10 kappa10', 'log10 T_re', 'DN_re']
SEED_POST = 20265921
NOISE_SEEDS = (20265921, 11, 22, 33, 44, 55)  # R55 中报判词 2
N = 5000
OUT = WS / 'outputs/r55/t2_main'
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 't2_main.log'
# r54_popW.md「E4 U1c_s1」合成 SNR 分档行（新比值, 覆盖），自检参照（现在档）
REF_E4_NOW = {
    '<1': [(0.982, 0.893), (0.951, 0.936), (0.985, 0.876), (1.018, 0.879), (0.995, 0.889)],
    '1-10': [(0.982, 0.840), (0.952, 1.000), (0.984, 0.896), (1.018, 0.993), (0.996, 0.903)],
    '10-100': [(0.894, 0.937), (0.401, 0.958), (0.953, 0.838), (0.328, 0.923), (0.861, 0.873)],
    '>100': [(0.773, 0.923), (0.336, 0.875), (0.924, 0.841), (0.281, 0.894), (0.783, 0.812)],
    '>1 合计': [(0.904, 0.903), (0.448, 0.935), (0.963, 0.856), (0.331, 0.931), (0.925, 0.856)],
}
BINS = (('<1', lambda s: s <= 1), ('1-10', lambda s: (s > 1) & (s <= 10)),
        ('10-100', lambda s: (s > 10) & (s <= 100)), ('>100', lambda s: s > 100),
        ('>1 合计', lambda s: s > 1))


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


def r2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def metrics(z, cut):
    t, pm = z['true'][cut], z['posterior_mean'][cut]
    q05, q95 = z['q05'][cut], z['q95'][cut]
    out = []
    for i in range(5):
        w = np.median(q95[:, i] - q05[:, i]) / POPW[i]
        cov = np.mean((t[:, i] >= q05[:, i]) & (t[:, i] <= q95[:, i]))
        out.append((w, cov, r2(pm[:, i], t[:, i])))
    return out


def table(md, title, z, snr, mask=None):
    md += [f'### {title}', '',
           '| 档 | n | ' + ' | '.join(f'{p} 宽÷先验/覆盖/R²' for p in PARAMS) + ' |',
           '|---|---|' + '---|' * 5]
    for lab, f in BINS:
        cut = f(snr) if mask is None else (f(snr) & mask)
        n = int(cut.sum())
        if n == 0:
            md.append(f'| {lab} | 0 | ' + ' | '.join('—' for _ in PARAMS) + ' |')
            continue
        mt = metrics(z, cut)
        md.append(f'| {lab} | {n} | ' + ' | '.join(
            f'{w:.3f}/{c:.3f}/{r:+.3f}' for w, c, r in mt) + ' |')
    md.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['now', 'fut'], required=True)
    ap.add_argument('--gpu', type=int, default=1)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    device = torch.device('cuda')
    if a.tier == 'now':
        tag = 'U1c_s1'
        CK = WS / 'outputs/r51/t1_train/U1c_s1/U1c_flow.pt'
        snr_keys = ('snr_pta_ng15', 'snr_lisa_cbe', 'snr_ligo_o3')
    else:
        tag = 'U1c_fut_s1'
        CK = WS / 'outputs/r55/t1_train/U1c_fut_s1/U1c_flow.pt'
        snr_keys = ('snr_pta_ngmock', 'snr_lisa_cbe', 'snr_ligo_ce40')
    ck = torch.load(CK, map_location='cpu', weights_only=False)
    ck_sha = sha256_file(CK)
    noise_config = ck['config']['noise']  # 噪声口径直接取 ckpt（训练同款）
    log(f'T2 tier={a.tier} ckpt={CK} sha={ck_sha[:8]} pta={noise_config["pta"]["mode"]} '
        f'ligo={noise_config["ligo"]["noise_type"]}:'
        f'{Path(noise_config["ligo"]["noise_file"]).name}')
    model = U1Flow(ck['band_config'], num_inputs=9, num_hidden=128, num_layers=10,
                   context_dim=64, in_channels=ck['num_channels']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    # ---- 并集 52,115（顺序=pta_val 行序，再 lisa_val 新 id，再联合 val 新 id）----
    pta_all = json.load(open(PTA_JSON))
    lisa_all = json.load(open(LISA_JSON))
    ligo_all = json.load(open(LIGO_JSON))
    _, pv = train_test_split(np.arange(len(pta_all)), test_size=0.2, random_state=42)
    _, lv = train_test_split(np.arange(len(lisa_all)), test_size=0.2, random_state=42)
    _, gv = train_test_split(np.arange(len(ligo_all)), test_size=0.2, random_state=42)
    pta_by = {str(x['_id']): x for x in pta_all}
    lisa_by = {str(x['_id']): x for x in lisa_all}
    ligo_by = {str(x['_id']): x for x in ligo_all}
    ids, seen = [], set()
    for src, ix in ((pta_all, pv), (lisa_all, lv), (ligo_all, gv)):
        for i in ix:
            oid = str(src[i]['_id'])
            if oid not in seen:
                seen.add(oid)
                ids.append(oid)
    assert len(ids) == 52115, len(ids)
    ligo_val_ids = [str(ligo_all[i]['_id']) for i in gv]
    del pta_all, lisa_all, ligo_all
    real_av = np.array([[1.0, float(oid in lisa_by), float(oid in ligo_by)] for oid in ids],
                       dtype=np.float32)
    n_op = int(((real_av[:, 1] == 0) & (real_av[:, 2] == 0)).sum())
    n_pl = int(((real_av[:, 1] == 1) & (real_av[:, 2] == 0)).sum())
    n_a3 = int((real_av[:, 2] == 1).sum())
    n_bad = int(((real_av[:, 1] == 0) & (real_av[:, 2] == 1)).sum())
    log(f'并集 {len(ids)}: 只PTA {n_op} / PTA+LISA {n_pl} / 三段 {n_a3} / 有LIGO无LISA {n_bad}')

    # ---- 合成 SNR：r40 表只含各自验证集 → 用 r40 表内存的 σ 数组对干净曲线直接复算
    #      （定义同 r40：SNR=sqrt Σ(10^log10Ω/σ)²），再与 r40 表/联合 val 分量表重叠 id 逐条核对 ----
    z = np.load(SNR_TABLE, allow_pickle=True)
    sig_key = {'snr_pta_ng15': 'sigma_pta_ng15', 'snr_pta_ngmock': 'sigma_pta_ngmock',
               'snr_lisa_cbe': 'sigma_lisa_cbe', 'snr_ligo_o3': 'sigma_ligo_o3',
               'snr_ligo_ce40': 'sigma_ligo_ce40'}
    by = (pta_by, lisa_by, ligo_by)
    snr_b = np.zeros((len(ids), 3))
    for b, k in enumerate(snr_keys):
        sig = np.asarray(z[sig_key[k]], dtype=np.float64)
        for jj, oid in enumerate(ids):
            if real_av[jj, b] > 0:
                x = 10.0 ** np.asarray(by[b][oid]['log10OmegaGW'], dtype=np.float64)[:len(sig)]
                snr_b[jj, b] = np.sqrt(((x / sig) ** 2).sum())
    pos = {oid: jj for jj, oid in enumerate(ids)}
    chk = {}
    j = np.load(JOINT_SNR, allow_pickle=True)
    for b, k in enumerate(snr_keys):
        rel = []
        for src in (z, j):
            for oid, v in zip(src['mongo_ids'], src[k]):
                o = str(oid)
                if np.isfinite(v) and o in pos and v > 0:
                    rel.append(abs(snr_b[pos[o], b] - v) / v)
        chk[k] = (len(rel), float(max(rel)) if rel else None)
    log(f'SNR 复算 vs r40/联合 val 表（重叠 id 条数, 最大相对差）: {chk}')
    assert all(v[1] is not None and v[1] < 1e-5 for v in chk.values()), chk
    snr = np.sqrt((snr_b ** 2).sum(axis=1))

    # ---- 推理：R55 中报判词 2——6 次噪声实现（种子 20265921 + 11/22/33/44/55），每次后验采样种子同 SEED_POST ----
    def _zero(p):
        first = json.load(open(p))[0]
        return dict(f=first['f'], log10OmegaGW=[ZERO_OM] * len(first['f']))
    zl, zg = _zero(LISA_JSON), _zero(LIGO_JSON)
    items = [dict(_id=oid, params=pta_by[oid], pta=pta_by[oid],
                  lisa=lisa_by.get(oid, zl), ligo=ligo_by.get(oid, zg)) for oid in ids]
    np.random.seed(12345)  # PTA omega_mean MC 固定（R52 B 修法）
    ds = U1cDataset(items, noise_config, True, True, train=False)
    ds.pta_scalers = ck['pta_scalers']
    ds.lisa_scalers = ck['lisa_scalers']
    ds.ligo_scalers = ck['ligo_scalers']
    ds.param_scaler = ck['param_scaler']
    ds.params = ck['param_scaler'].transform(ds.params_raw)
    B = len(ds)
    av = torch.ones((64, 3), device=device)
    true6 = ds.params_raw[:, [0, 1, 2, 3, 4, 8]].astype(np.float64)
    seeds = NOISE_SEEDS
    zs, sha_by_seed = {}, {}
    for sd in seeds:
        np.random.seed(sd)
        ds.refresh_noisy_cache(1.0)
        np.random.seed(SEED_POST)
        torch.manual_seed(SEED_POST)
        torch.cuda.manual_seed_all(SEED_POST)
        q05, q50, q95, mean_, ranks = [], [], [], [], []
        t0 = time.time()
        for s in range(0, B, 64):
            e = min(s + 64, B)
            with torch.no_grad():
                samp = model.sample(torch.tensor(ds._cache['PTA'][s:e]).to(device),
                                    torch.tensor(ds._cache['LISA'][s:e]).to(device),
                                    torch.tensor(ds._cache['LIGO'][s:e]).to(device),
                                    av[:e - s], num_samples=N).cpu().numpy()
            b = e - s
            sp = ds.param_scaler.inverse_transform(samp.reshape(b * N, 9)).reshape(b, N, 9)
            act = sp[:, :, [0, 1, 2, 3, 4, 8]].astype(np.float64)
            mean_.append(act.mean(axis=1))
            qs = np.quantile(act, [0.05, 0.5, 0.95], axis=1)
            q05.append(qs[0])
            q50.append(qs[1])
            q95.append(qs[2])
            ranks.append(np.mean(act < true6[s:e][:, None, :], axis=1))
        zout = dict(true=true6, point=np.concatenate(q50), q05=np.concatenate(q05),
                    q16=np.full((B, 6), np.nan), q50=np.concatenate(q50),
                    q84=np.full((B, 6), np.nan), q95=np.concatenate(q95),
                    ranks=np.concatenate(ranks), posterior_mean=np.concatenate(mean_),
                    posterior_std=np.zeros((B, 6)), mongo_ids=np.array(ids, dtype=object),
                    real_avail=real_av, snr_components=snr_b, snr_combined=snr,
                    snr_keys=np.array(snr_keys), ckpt_sha256=ck_sha,
                    noise_seed=sd, sample_seed=SEED_POST)
        for k in ('q05', 'q95', 'posterior_mean'):
            assert np.isfinite(zout[k]).all(), f'NaN in {k} seed {sd}'
        fn = OUT / f'{tag}_main_{a.tier}_noise{sd}_Median5000_plus.npz'
        np.savez_compressed(fn, **zout)
        sha_by_seed[sd] = sha256_file(fn)[:12]
        log(f'noise seed {sd}: sampled {B}x{N} in {time.time() - t0:.0f}s; export {fn.name} '
            f'sha {sha_by_seed[sd]} {fn.stat().st_size / 1e6:.1f}MB')
        zs[sd] = zout

    # ---- 聚合：每格 6 次的均值 / 最小 / 最大 ----
    def agg(cut):
        M = np.array([[list(t) for t in metrics(zs[sd], cut)] for sd in seeds])  # (R,5,3)
        return M.mean(axis=0), M.min(axis=0), M.max(axis=0)

    def table_multi(md, title, mask=None):
        md += [f'### {title}', '', '均值（宽÷先验 / 覆盖 / R²）', '',
               '| 档 | n | ' + ' | '.join(f'{p}' for p in PARAMS) + ' |', '|---|---|' + '---|' * 5]
        rng_rows = []
        for lab, f in BINS:
            cut = f(snr) if mask is None else (f(snr) & mask)
            n = int(cut.sum())
            if n == 0:
                md.append(f'| {lab} | 0 | ' + ' | '.join('—' for _ in PARAMS) + ' |')
                rng_rows.append(f'| {lab} | 0 | ' + ' | '.join('—' for _ in PARAMS) + ' |')
                continue
            mn, lo, hi = agg(cut)
            md.append(f'| {lab} | {n} | ' + ' | '.join(
                f'{mn[i, 0]:.3f}/{mn[i, 1]:.3f}/{mn[i, 2]:+.3f}' for i in range(5)) + ' |')
            rng_rows.append(f'| {lab} | {n} | ' + ' | '.join(
                f'{lo[i, 0]:.3f}–{hi[i, 0]:.3f} / {lo[i, 1]:.3f}–{hi[i, 1]:.3f}' for i in range(5)) + ' |')
        md += ['', f'波动（{len(seeds)} 次实现的 最小–最大：宽÷先验 / 覆盖）', '',
               '| 档 | n | ' + ' | '.join(f'{p}' for p in PARAMS) + ' |', '|---|---|' + '---|' * 5]
        md += rng_rows + ['']

    tier_txt = '现在档 NG15+LISA cbe+O3' if a.tier == 'now' else '未来档 NGmock+LISA cbe+CE40'
    md = [f'# R55 T2 — {tag} 论文主表（{tier_txt}；{len(seeds)} 次噪声实现平均）', '',
          f'- 生成 {utc()}；ckpt {CK.name} sha {ck_sha[:8]}；后验采样 seed {SEED_POST} n={N}',
          f'- **噪声实现种子 {"/".join(str(s) for s in seeds)}**（R55 中报判词 2：每格报 {len(seeds)} 次均值，'
          '另附最小–最大波动表）；每次实现只换噪声种子，后验采样种子不变。',
          '- 样本=三验证集并集 52,115（pta_val∪lisa_val∪联合 val，mongo id 去重）；'
          '**每条三探测器全给**，真实延伸不到的波段按 U1c 规则填纯噪声（信号=0+探测器噪声），'
          '即“三探测器都在”的现实情形。',
          f'- 分档=合成 SNR √(SNR_PTA²+SNR_LISA²+SNR_LIGO²)（{"/".join(snr_keys)}；'
          'r40 定义=干净曲线，按 r40 表内 σ 复算，与 r40/联合 val 表重叠 id 相对差 0；'
          '延伸不到的波段信号=0 故分量为 0）。',
          f'- 指标=区间÷先验（分母 W_pop={"/".join(f"{w:g}" for w in POPW)}，同 r54_popW）/覆盖/R²。',
          f'- 真实可用性：只 PTA {n_op} / PTA+LISA {n_pl} / 三段都有 {n_a3}。',
          '- 各次导出 npz sha256 前 12 位：' + '；'.join(f'{sd}: {h}' for sd, h in sha_by_seed.items()), '',
          '## 主表（并集 52,115）', '']
    table_multi(md, '全体')
    md += ['## 子表（按真实可用性）', '']
    for nm, m in (('只有 PTA', (real_av[:, 1] == 0) & (real_av[:, 2] == 0)),
                  ('PTA+LISA', (real_av[:, 1] == 1) & (real_av[:, 2] == 0)),
                  ('三段都有', real_av[:, 2] == 1)):
        table_multi(md, f'{nm}（n={int(m.sum())}）', m)

    ok = True
    if a.tier == 'now':
        pos = {oid: jj for jj, oid in enumerate(ids)}
        sub = np.zeros(B, bool)
        sub[[pos[o] for o in ligo_val_ids]] = True
        assert sub.sum() == 9420
        md += [f'## 自检（R55 中报判词 2：联合 val 9,420 子集，r54_popW「U1c_s1 E4」单次值须落在 '
               f'{len(seeds)} 次实现 [最小−0.01, 最大+0.01] 内；宽÷先验与覆盖都查）', '',
               '| 档 | 参数 | r54 E4 宽 | 本表宽 [min, max] | r54 E4 覆 | 本表覆 [min, max] | 判 |',
               '|---|---|---|---|---|---|---|']
        for lab, f in BINS:
            cut = f(snr) & sub
            mn, lo, hi = agg(cut)
            for i, p in enumerate(PARAMS):
                rw, rc = REF_E4_NOW[lab][i]
                gw = (lo[i, 0] - 0.01) <= rw <= (hi[i, 0] + 0.01)
                gc = (lo[i, 1] - 0.01) <= rc <= (hi[i, 1] + 0.01)
                good = gw and gc
                ok = ok and good
                md.append(f'| {lab} (n={int(cut.sum())}) | {p} | {rw:.3f} | [{lo[i, 0]:.3f}, {hi[i, 0]:.3f}] | '
                          f'{rc:.3f} | [{lo[i, 1]:.3f}, {hi[i, 1]:.3f}] | {"PASS" if good else "**FAIL**"} |')
        md += ['', f'**自检总判：{"PASS" if ok else "FAIL → 按停跑条件停并报告"}**', '']
        log(f'自检 {"PASS" if ok else "FAIL"}')
    mdp = WS / f'outputs/r55/r55_main_{a.tier}.md'
    mdp.write_text('\n'.join(md) + '\n', encoding='utf-8')
    log(f'wrote {mdp}')
    if not ok:
        sys.exit(3)


if __name__ == '__main__':
    main()
