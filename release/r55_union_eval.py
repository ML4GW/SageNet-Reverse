#!/usr/bin/env python
# R55 T1 评估：U1c E1–E5（R51 T2 口径），分母 W_pop（r54_popW 全体有效先验宽）
#   --tier fut：U1c_fut_s1，对照 E1 mock_s1 / E2 L1 v3 / E3 L3_CE40_s1 / E4 Jfut_fix_s1；SNR ngmock/cbe/ce40/joint_fut
#   --tier now：U1c_s1（脚本验证 + 修正版 E5），对照 S_PTA / L1 v3 / L3 v3 / J1；SNR ng15/cbe/o3/joint_now
# ★ E5 子集按"全数据集成员"拆（R50 口径）——R51 原脚本按"验证集成员"拆有误（见 r51_e5_resplit_check.txt）
# 噪声实现：每个 E 构建 ds 前 np.random.seed(12345)（PTA omega_mean MC，R52 B 修法），加噪前 np.random.seed(SEED_POST)
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
POPW = np.array([34.744, 5.008, 8.963, 8.745, 35.451])
PARAMS = ['log10 r', 'n_t', 'log10 kappa10', 'log10 T_re', 'DN_re']
SEED_POST = 20265921
N = 5000
CFG = dict(
    now=dict(tag='U1c_s1', ck=WS / 'outputs/r51/t1_train/U1c_s1/U1c_flow.pt',
             runres=WS / 'outputs/r51/t1_train/U1c_s1/run_result.json',
             refs=dict(E1=('S_PTA', WS / 'outputs/r35_export/S_PTA_Median5000_plus.npz'),
                       E2=('L1 v3', WS / 'outputs/r35_export/L1_Median5000_plus.npz'),
                       E3=('L3 v3', WS / 'outputs/r35_export/L3_Median5000_plus.npz'),
                       E4=('J1', WS / 'outputs/r41/t2_export/J1_Median5000_plus.npz')),
             snr=dict(E1='snr_pta_ng15', E2='snr_lisa_cbe', E3='snr_ligo_o3', E4='joint_snr_now'),
             out='r55_union_now_check.md'),
    fut=dict(tag='U1c_fut_s1', ck=WS / 'outputs/r55/t1_train/U1c_fut_s1/U1c_flow.pt',
             runres=WS / 'outputs/r55/t1_train/U1c_fut_s1/run_result.json',
             refs=dict(E1=('mock_s1', WS / 'outputs/r37/a3_export/mock_s1_Median5000_plus.npz'),
                       E2=('L1 v3', WS / 'outputs/r35_export/L1_Median5000_plus.npz'),
                       E3=('L3_CE40_s1', WS / 'outputs/r40/t2_export/L3_CE40_s1_Median5000_plus.npz'),
                       E4=('Jfut_fix_s1', WS / 'outputs/r41/t1_export/Jfut_fix_s1_Median5000_plus.npz')),
             snr=dict(E1='snr_pta_ngmock', E2='snr_lisa_cbe', E3='snr_ligo_ce40', E4='joint_snr_fut'),
             out='r55_union_fut.md'))
BINS = (('<1', lambda s: s <= 1), ('1-10', lambda s: (s > 1) & (s <= 10)),
        ('10-100', lambda s: (s > 10) & (s <= 100)), ('>100', lambda s: s > 100),
        ('>1 合计', lambda s: s > 1))


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '+00:00'


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
    t, pm, q05, q95 = z['true'][cut], z['posterior_mean'][cut], z['q05'][cut], z['q95'][cut]
    out = []
    for i in range(5):
        out.append((np.median(q95[:, i] - q05[:, i]) / POPW[i],
                    np.mean((t[:, i] >= q05[:, i]) & (t[:, i] <= q95[:, i])),
                    r2(pm[:, i], t[:, i])))
    return out


def cells(mt):
    return ' | '.join(f'{w:.3f}/{c:.3f}/{r:+.3f}' for w, c, r in mt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['now', 'fut'], required=True)
    ap.add_argument('--gpu', type=int, default=1)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    device = torch.device('cuda')
    C = CFG[a.tier]
    OUT = WS / f'outputs/r55/eval_{a.tier}'
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = OUT / 'eval.log'

    def log(msg):
        line = f'[{utc()}] {msg}'
        print(line, flush=True)
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    t0 = time.time()
    ck = torch.load(C['ck'], map_location='cpu', weights_only=False)
    ck_sha = sha256_file(C['ck'])
    rr = json.loads(Path(C['runres']).read_text())
    noise_config = ck['config']['noise']
    model = U1Flow(ck['band_config'], num_inputs=9, num_hidden=128, num_layers=10,
                   context_dim=64, in_channels=ck['num_channels']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    log(f'eval tier={a.tier} {C["tag"]} sha {ck_sha[:8]} best_ep {rr["best_epoch"]} '
        f'pta={noise_config["pta"]["mode"]} ligo={Path(noise_config["ligo"]["noise_file"]).name}')

    P = json.load(open(PTA_JSON))
    L = json.load(open(LISA_JSON))
    G = json.load(open(LIGO_JSON))
    _, pv = train_test_split(np.arange(len(P)), test_size=0.2, random_state=42)
    _, lv = train_test_split(np.arange(len(L)), test_size=0.2, random_state=42)
    _, gv = train_test_split(np.arange(len(G)), test_size=0.2, random_state=42)
    pta_by = {str(x['_id']): x for x in P}
    lisa_by = {str(x['_id']): x for x in L}
    ligo_by = {str(x['_id']): x for x in G}
    e1_ids = [str(P[i]['_id']) for i in pv]
    e2_ids = [str(L[i]['_id']) for i in lv]
    e3_ids = [str(G[i]['_id']) for i in gv]
    del P, L, G

    def zero(p):
        first = json.load(open(p))[0]
        return dict(f=first['f'], log10OmegaGW=[ZERO_OM] * len(first['f']))
    zl, zg = zero(LISA_JSON), zero(LIGO_JSON)

    def build(ids):
        items = [dict(_id=o, params=pta_by[o], pta=pta_by[o],
                      lisa=lisa_by.get(o, zl), ligo=ligo_by.get(o, zg)) for o in ids]
        np.random.seed(12345)
        ds = U1cDataset(items, noise_config, True, True, train=False)
        ds.pta_scalers = ck['pta_scalers']
        ds.lisa_scalers = ck['lisa_scalers']
        ds.ligo_scalers = ck['ligo_scalers']
        ds.param_scaler = ck['param_scaler']
        ds.params = ck['param_scaler'].transform(ds.params_raw)
        np.random.seed(SEED_POST)
        ds.refresh_noisy_cache(1.0)
        return ds

    def sample(ds, av_rows, ids, tag):
        np.random.seed(SEED_POST)
        torch.manual_seed(SEED_POST)
        torch.cuda.manual_seed_all(SEED_POST)
        B = len(ds)
        av = torch.tensor(np.asarray(av_rows, dtype=np.float32), device=device)
        q05, q50, q95, mean_, ranks = [], [], [], [], []
        true6 = ds.params_raw[:, [0, 1, 2, 3, 4, 8]].astype(np.float64)
        for s in range(0, B, 64):
            e = min(s + 64, B)
            with torch.no_grad():
                samp = model.sample(torch.tensor(ds._cache['PTA'][s:e]).to(device),
                                    torch.tensor(ds._cache['LISA'][s:e]).to(device),
                                    torch.tensor(ds._cache['LIGO'][s:e]).to(device),
                                    av[s:e], num_samples=N).cpu().numpy()
            b = e - s
            sp = ds.param_scaler.inverse_transform(samp.reshape(b * N, 9)).reshape(b, N, 9)
            act = sp[:, :, [0, 1, 2, 3, 4, 8]].astype(np.float64)
            mean_.append(act.mean(axis=1))
            qs = np.quantile(act, [0.05, 0.5, 0.95], axis=1)
            q05.append(qs[0])
            q50.append(qs[1])
            q95.append(qs[2])
            ranks.append(np.mean(act < true6[s:e][:, None, :], axis=1))
        z = dict(true=true6, point=np.concatenate(q50), q05=np.concatenate(q05),
                 q16=np.full((B, 6), np.nan), q50=np.concatenate(q50),
                 q84=np.full((B, 6), np.nan), q95=np.concatenate(q95),
                 ranks=np.concatenate(ranks), posterior_mean=np.concatenate(mean_),
                 posterior_std=np.zeros((B, 6)), mongo_ids=np.array(ids, dtype=object),
                 avail_rows=np.asarray(av_rows, dtype=np.float32), ckpt_sha256=ck_sha)
        for k in ('q05', 'q95', 'posterior_mean'):
            assert np.isfinite(z[k]).all(), f'NaN {k} {tag}'
        fn = OUT / f'{C["tag"]}_{tag}_Median5000_plus.npz'
        np.savez_compressed(fn, **z)
        exports[tag] = sha256_file(fn)[:12]
        log(f'[{tag}] {B} rows sampled; export sha {exports[tag]}')
        return z

    exports = {}
    sz = np.load(SNR_TABLE, allow_pickle=True)
    jz = np.load(JOINT_SNR, allow_pickle=True)

    def snr_of(eid, ids):
        k = C['snr'][eid]
        if eid == 'E4':
            assert [str(x) for x in jz['mongo_ids']] == ids
            return jz[k].astype(np.float64)
        dsn = {'E1': 'pta', 'E2': 'lisa', 'E3': 'ligo'}[eid]
        m = sz['dataset'] == dsn
        assert [str(x) for x in sz['mongo_ids'][m]] == ids, f'{eid} snr 行序'
        return sz[k][m].astype(np.float64)

    tier_txt = '未来档 NGmock+LISA cbe+CE40' if a.tier == 'fut' else '现在档 NG15+LISA cbe+O3（脚本验证+修正版 E5）'
    md = [f'# R55 — {C["tag"]} 评估 E1–E5（{tier_txt}）', '',
          f'- 生成 {utc()}；{C["tag"]} 等权 best_epoch={rr["best_epoch"]} best_val={rr["best_val"]:.6f}；'
          f'ckpt sha {ck_sha[:8]}；后验 seed {SEED_POST} n={N}；噪声实现 seed {SEED_POST}（单次，同 R51 口径）',
          f'- 指标=区间÷先验（分母 W_pop={"/".join(f"{w:g}" for w in POPW)}，r54_popW 全体有效先验宽，'
          'U1c 与对照共用）/ 覆盖 / R²。',
          '- E1–E3 单波段输入=只有该探测器在；E4 三段都给；E5 泄露复查按真实可用性拆分，'
          '**子集按全数据集成员拆（R50 口径）**；(ii) 真实输入=真实存在的波段全给。', '']
    zs, snrs, ids_by = {}, {}, dict(E1=e1_ids, E2=e2_ids, E3=e3_ids, E4=e3_ids)
    titles = dict(E1='E1 — pta_val 33,226（只给 PTA）', E2='E2 — lisa_val 15,927（只给 LISA）',
                  E3='E3 — LIGO val 9,420（只给 LIGO）', E4='E4 — 联合 val 9,420（三段都给）')
    for eid, band in (('E1', 0), ('E2', 1), ('E3', 2), ('E4', None)):
        log(f'{eid} start')
        ids = ids_by[eid]
        ds = build(ids)
        av = np.ones((len(ids), 3), np.float32) if band is None else np.zeros((len(ids), 3), np.float32)
        if band is not None:
            av[:, band] = 1
        z = sample(ds, av, ids, eid)
        del ds
        snr = snr_of(eid, ids)
        ref_name, ref_path = C['refs'][eid]
        zr = np.load(ref_path, allow_pickle=True)
        assert [str(x) for x in zr['mongo_ids']] == ids, f'{eid} 对照行序'
        zs[eid], snrs[eid] = (z, zr), snr
        md += [f'## {titles[eid]}；对照 {ref_name}', '',
               '| 模型 | 档 | n | ' + ' | '.join(f'{p} 宽÷先验/覆/R²' for p in PARAMS) + ' |',
               '|---|---|---|' + '---|' * 5]
        for nm, zz in ((C['tag'], z), (f'{ref_name}（对照）', zr)):
            for lab, f in BINS:
                cut = f(snr)
                md.append(f'| {nm} | {lab} | {int(cut.sum())} | {cells(metrics(zz, cut))} |')
        md.append('')

    # ---- E5 ----
    log('E5 start')
    lisa_all, ligo_all = set(lisa_by), set(ligo_by)
    e5 = {}
    for eid, ids in (('E1', e1_ids), ('E2', e2_ids)):
        av = np.array([[1.0, float(o in lisa_all), float(o in ligo_all)] for o in ids], np.float32)
        ds = build(ids)
        e5[eid] = sample(ds, av, ids, f'E5_{eid}_ii')
        del ds
    md += ['## E5 — 泄露复查（全数据集成员拆分；(i)=单波段输入取自 E1/E2，(ii)=真实输入）', '']
    accept = []
    for eid, groups in (('E1', ('只有PTA', 'PTA+LISA', '三段都有')), ('E2', ('PTA+LISA', '三段都有'))):
        ids = ids_by[eid]
        l = np.array([o in lisa_all for o in ids])
        g = np.array([o in ligo_all for o in ids])
        masks = {'只有PTA': ~l & ~g, 'PTA+LISA': l & ~g, '三段都有': g}
        lt1 = snrs[eid] <= 1
        md += [f'### E5/{eid}（SNR<1 档覆盖率）', '',
               '| 输入 | 子集 (全集 n / <1 档 n) | ' + ' | '.join(f'{p}' for p in PARAMS) + ' |',
               '|---|---|' + '---|' * 5]
        for sub in groups:
            cm = masks[sub] & lt1
            for lab, zz in (('(i) 单波段', zs[eid][0]), ('(ii) 真实输入', e5[eid])):
                cv = [c for _, c, _ in metrics(zz, cm)]
                md.append(f'| {lab} | {sub} ({int(masks[sub].sum())} / {int(cm.sum())}) | '
                          + ' | '.join(f'{c:.3f}' for c in cv) + ' |')
                if eid == 'E1' and lab.startswith('(i)'):
                    accept.append((sub, min(cv)))
        md.append('')
    md += ['### E5 验收（E1 SNR<1 档三子集每参数覆盖 ≥0.85）', '', '| 子集 | 最差覆盖 | 判 |', '|---|---|---|']
    md += [f'| {s} | {w:.3f} | {"达到" if w >= 0.85 else "未达到"} |' for s, w in accept] + ['']

    # ---- 验收（R49 口径：对照 R²≥0.1 的格子，U1 宽 ≤ 对照+0.05 且覆盖 ≥0.85）----
    md += ['## 验收判据（SNR 10–100 / >100，只对对照 R²≥0.1 的格子；U1c 宽÷先验 ≤ 对照+0.05 且覆盖 ≥0.85）', '',
           '| 实验 | 档 | 参数 | U1c 宽/覆 | 对照 宽/覆 | 判 |', '|---|---|---|---|---|---|']
    hits = []
    for eid in ('E1', 'E2', 'E3', 'E4'):
        z, zr = zs[eid]
        for lab, f in BINS[2:4]:
            cut = f(snrs[eid])
            if not cut.any():
                continue
            mu, mr = metrics(z, cut), metrics(zr, cut)
            for i, p in enumerate(PARAMS):
                if mr[i][2] >= 0.1:
                    hit = mu[i][0] <= mr[i][0] + 0.05 and mu[i][1] >= 0.85
                    hits.append(hit)
                    md.append(f'| {eid} | {lab} | {p} | {mu[i][0]:.3f}/{mu[i][1]:.3f} | '
                              f'{mr[i][0]:.3f}/{mr[i][1]:.3f} | {"达到" if hit else "未达到"} |')
    md += ['', f'合计达到 {sum(hits)}/{len(hits)}。', '', '## 导出件', '']
    md += [f'- {C["tag"]}_{k}_Median5000_plus.npz — sha256 {v}' for k, v in exports.items()] + ['']
    (WS / 'outputs/r55' / C['out']).write_text('\n'.join(md) + '\n', encoding='utf-8')
    log(f'DONE {C["out"]} elapsed {time.time() - t0:.0f}s; 验收 {sum(hits)}/{len(hits)}')


if __name__ == '__main__':
    main()
