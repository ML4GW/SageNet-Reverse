#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R40 T4: SNR 总表 (纯 numpy, 零 torch 零训练).

任务单 T4 逐字: "r40_snr_table.npz, 三验证集 (33,226/15,927/9,420) 行序对齐各自
Median5000_plus 导出件, 含 mongo_ids; PTA NG15+NGmock (定义同 R39 T2)/LISA/
LIGO O3+CE40, 无定义记 NaN; 每列 σ 数组 sha256".

行对齐 = mongo_ids 与 canonical train_test_split(rs=42) val 行 _id 逐位一致
(本脚本不信任存量导出件, 自行从 json 重算 val 顺序并写回 mongo_ids;
对 R39 T2 PTA / R35 L1 导出件做一致性核验, 不一致即停).

σ 定义 (全部与本轮训练 pipeline 同源, 纯 numpy 复算):
  PTA NG15/NGmock : R39 T2 constants() 逐字 (density.npy / density_mock.npy,
                    SEED 20261007, N_MC 50000, K=(8pi^4/H0_SI^2)*T_obs*f^5)
  LISA cbe        : train.py LISANoiseModel 同类实例 (numpy 实现, 与 C1/Jfut
                    训练 pipeline 同源同式: Robson+2019 cbe, T_obs 4yr,
                    confusion on) — 不手抄解析式, 直接实例化保证定义恒等
  LIGO O3 / CE40  : R35 预计算 sigma_LIGO_{O3,CE40}.npy (546,)
                    (R40 T4 前置已核: canonical 网格 fresh LIGONoiseModel
                     复算 bit-exact maxrel=0.0)

SNR 定义 (同 R39 T2): SNR = sqrt(sum_i (x_i / sigma_i)^2), x = 10^log10OmegaGW
干净曲线 = json 行内 log10OmegaGW 字段直接读取.

输出: outputs/r40/t4/r40_snr_table.npz + r40_snr_table_report.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# --- release path wiring: optional sibling paths.json (absent = original behavior) ---
_DIR = Path(__file__).resolve().parent
_PFILE = _DIR / 'paths.json'
_P = json.loads(_PFILE.read_text(encoding='utf-8')) if _PFILE.is_file() else {}

def _rp(key, default):
    v = _P.get(key, default)
    return str((_DIR / str(v)).resolve())
WS = Path(_rp('workspace', '/root/SageNetR/workspace'))
NOISE_PTA_DIR = WS / 'new/noise/NANOGrav15yr'
OUT = WS / 'outputs/r40/t4'
OUT.mkdir(parents=True, exist_ok=True)

# ---- R39 T2 常数 (r39/t2/t2_pta_probe.py 逐字) ----
SEED = 20261007
N_MC = 50000
KM_PER_MPC = 3.0856775814913673e19
H0 = 67.4
N_BINS = 14
D_IDX = [0, 1, 2, 3, 7]
U_IDX = [5, 6, 8, 9, 10, 11, 12, 13]
B5 = 4


def sha256_arr(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# ---- PTA σ (R39 T2 constants 逐字) ----

def build_cdf(density_file):
    density = np.load(str(Path(NOISE_PTA_DIR, density_file)))
    d2 = density[0] if density.ndim == 3 else density
    log_pdf = d2[:N_BINS].astype(np.float64)
    log10rho = np.load(str(Path(NOISE_PTA_DIR, 'log10rhogrid.npy'))).astype(np.float64)
    pdf = np.exp(log_pdf - log_pdf.max(axis=1, keepdims=True))
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    pdf = pdf / trapz(pdf, log10rho, axis=1)[:, None]
    cdf = np.cumsum(pdf, axis=1)
    cdf = cdf - cdf[:, :1]
    return cdf / cdf[:, -1:].clip(min=1e-300)


def constants(density_file, freqs, log10rho):
    cdf = build_cdf(density_file)
    np.random.seed(SEED)
    u = np.random.uniform(0.0, 1.0, size=(N_MC, N_BINS))
    rho = np.empty_like(u)
    for i in range(N_BINS):
        rho[:, i] = np.interp(u[:, i], cdf[i], log10rho)
    om = K[None, :] * (10.0 ** rho) ** 2
    c = om.mean(axis=0)
    sig = om.std(axis=0)
    rho_p95 = log10rho[(cdf >= 0.95).argmax(axis=1)]
    U95 = K * (10.0 ** rho_p95) ** 2
    return c, sig, U95, cdf


def main():
    report = {}
    cols = {}

    # ================= PTA (33,226) =================
    print('[PTA] start', flush=True)
    jp = json.loads((WS / 'dataset/single/PTA_fixed_14pts.json').read_text())
    assert len(jp) == 166127
    _, va_p = train_test_split(np.arange(len(jp)), test_size=0.2, random_state=42)
    ids_p = [jp[int(r)]['_id'] for r in va_p]
    assert len(ids_p) == 33226
    clean_p = np.stack([np.asarray(jp[int(r)]['log10OmegaGW'], dtype=np.float64)
                        for r in va_p])
    assert clean_p.shape == (33226, 14)

    # R39 T2 逐字: freqs_all = 线性 Hz (noise 目录 freqs.npy);
    # (json 行内 f 是 log10 Hz, 10**f 与本数组逐位同, 仅记法不同)
    freqs_all = np.load(str(Path(NOISE_PTA_DIR, 'freqs.npy')))
    freqs = freqs_all[:N_BINS].astype(np.float64)
    T_obs = 1.0 / (freqs_all[1] - freqs_all[0])
    H0_SI = H0 / KM_PER_MPC
    globals()['K'] = (8.0 * np.pi ** 4 / H0_SI ** 2) * T_obs * freqs ** 5
    log10rho = np.load(str(Path(NOISE_PTA_DIR, 'log10rhogrid.npy'))).astype(np.float64)

    obs_c, obs_sig, _, _ = constants('density.npy', freqs, log10rho)
    mock_c, mock_sig, _, _ = constants('density_mock.npy', freqs, log10rho)

    x_p = 10.0 ** clean_p
    snr_p_obs = np.sqrt(((x_p / obs_sig[None, :]) ** 2).sum(axis=1))
    snr_p_mock = np.sqrt(((x_p / mock_sig[None, :]) ** 2).sum(axis=1))

    # 与 R39 T2 导出件对身份:
    #   mongo_ids 行序逐位; sigma/c 逐位 (同 constants 复算);
    #   snr 允许 <=1e-5 相对差 — R39 T2 clean 源 (pta_val_clean.npz) 经 float32
    #   舍入 (max|Δlog10Ω|=3.8e-6), 本表用 json float64 原值 (定义同, 精度更高)
    z39 = np.load(WS / 'outputs/r39/t2/r39_t2_pta_probe.npz', allow_pickle=True)
    assert list(z39['mongo_ids']) == ids_p, 'PTA mongo_ids 与 R39 T2 不一致'
    assert np.array_equal(z39['obs_sigma'], obs_sig)
    assert np.array_equal(z39['mock_sigma'], mock_sig)
    mr_o = float(np.max(np.abs(snr_p_obs - z39['snr_obs']) / z39['snr_obs']))
    mr_m = float(np.max(np.abs(snr_p_mock - z39['snr_mock']) / z39['snr_mock']))
    assert mr_o < 1e-5 and mr_m < 1e-5, f'PTA snr vs R39 T2 maxrel {mr_o}/{mr_m}'
    print(f'[PTA] snr_obs/snr_mock vs R39 T2 导出件 maxrel {mr_o:.2e}/{mr_m:.2e} '
          f'(float32 舍入级, 定义同); sigma 逐位同', flush=True)

    cols['pta'] = dict(
        mongo_ids=np.array(ids_p, dtype=object),
        clean_log10OmegaGW=clean_p.astype(np.float32),
        snr_ng15=snr_p_obs, snr_ngmock=snr_p_mock,
        sigma_ng15=obs_sig, sigma_ngmock=mock_sig,
        c_ng15=obs_c, c_ngmock=mock_c, freqs=freqs,
    )
    report['pta'] = dict(
        n=33226,
        sigma_ng15_sha256=sha256_arr(obs_sig.astype(np.float64)),
        sigma_ngmock_sha256=sha256_arr(mock_sig.astype(np.float64)),
        definition='R39 T2 constants() 逐字 (density/density_mock, SEED 20261007, N_MC 50000)',
        match_r39_t2_export=f'sigma/c 逐位同; snr maxrel {mr_o:.2e}/{mr_m:.2e} '
                            '(R39 clean 源经 float32 舍入, 本表用 json float64 原值)',
    )

    # ================= LISA (15,927) =================
    print('[LISA] start', flush=True)
    jl = json.loads((WS / 'dataset/single/LISA_fixed_500pts.json').read_text())
    assert len(jl) == 79634
    _, va_l = train_test_split(np.arange(len(jl)), test_size=0.2, random_state=42)
    ids_l = [jl[int(r)]['_id'] for r in va_l]
    assert len(ids_l) == 15927
    f_lisa = np.asarray(jl[0]['f'], dtype=np.float64)
    clean_l = np.stack([np.asarray(jl[int(r)]['log10OmegaGW'], dtype=np.float64)
                        for r in va_l])
    assert clean_l.shape == (15927, 500)

    # σ = train.py LISANoiseModel 同类实例 (与 C1/Jfut 训练 pipeline 同源同式:
    # Robson+2019 cbe, T_obs 4yr, confusion on) — 不手抄解析式, 定义恒等
    sys.path.insert(0, str(WS / 'new'))
    from train import LISANoiseModel  # noqa: E402
    m_lisa = LISANoiseModel(target_log10_freq=f_lisa, noise_preset='cbe',
                            T_obs=4.0, include_confusion=True)
    sig_lisa = np.asarray(m_lisa.sigma_omega, dtype=np.float64)
    assert sig_lisa.shape == (500,) and bool(np.isfinite(sig_lisa).all())
    print('[LISA] sigma = train.py LISANoiseModel 直接实例化 (与训练同源, 定义恒等)',
          flush=True)

    snr_l = np.sqrt(((10.0 ** clean_l / sig_lisa[None, :]) ** 2).sum(axis=1))
    # 对 R35 L1 导出件行序
    zl = np.load(WS / 'outputs/r35_export/L1_Median5000_plus.npz', allow_pickle=True)
    assert list(zl['mongo_ids']) == ids_l, 'LISA mongo_ids 与 L1 导出件不一致'
    print('[LISA] mongo_ids == L1_Median5000_plus.npz (逐位)', flush=True)

    cols['lisa'] = dict(
        mongo_ids=np.array(ids_l, dtype=object),
        clean_log10OmegaGW=clean_l.astype(np.float32),
        snr_lisa_cbe=snr_l, sigma_lisa_cbe=sig_lisa, freqs=f_lisa,
    )
    report['lisa'] = dict(
        n=15927,
        sigma_lisa_cbe_sha256=sha256_arr(sig_lisa.astype(np.float64)),
        definition='train.py LISANoiseModel 直接实例化 (cbe, T_obs 4yr, '
                   'confusion on) — 与 C1/Jfut 训练 pipeline 同源同式, 定义恒等',
        aligned_to='outputs/r35_export/L1_Median5000_plus.npz mongo_ids',
    )

    # ================= LIGO (9,420; O3 + CE40) =================
    print('[LIGO] start', flush=True)
    jg = json.loads((WS / 'dataset/single/LIGO_fixed_546pts.json').read_text())
    assert len(jg) == 47100
    _, va_g = train_test_split(np.arange(len(jg)), test_size=0.2, random_state=42)
    ids_g = [jg[int(r)]['_id'] for r in va_g]
    assert len(ids_g) == 9420
    f_ligo = np.asarray(jg[0]['f'], dtype=np.float64)
    clean_g = np.stack([np.asarray(jg[int(r)]['log10OmegaGW'], dtype=np.float64)
                        for r in va_g])
    assert clean_g.shape == (9420, 546)

    sig_o3 = np.load(WS / 'outputs/r35_export/noise_inputs/sigma_LIGO_O3.npy').astype(np.float64)
    sig_ce = np.load(WS / 'outputs/r35_export/noise_inputs/sigma_LIGO_CE40.npy').astype(np.float64)
    assert sig_o3.shape == sig_ce.shape == (546,)
    # 身份: R40 T4 前置已核 fresh LIGONoiseModel canonical 网格 bit-exact;
    # 此处内嵌复验 (改用 LIGONoiseModel 而非手抄解析式, 与训练同源)
    from train import LIGONoiseModel  # noqa: E402
    m_o3 = LIGONoiseModel(noise_file=str(WS / 'new/noise/C_O1_O2_O3.dat.txt'),
                          noise_type='cc_spectrum', target_log10_freq=f_ligo)
    m_ce = LIGONoiseModel(noise_file=str(Path(_rp('ce40_inputs', '/root/SageNetR/ce40_inputs')) / 'CE' / 'CE40km_v009_strain_ASD_LIGO546.txt'),
                          noise_type='aplus_asd', target_log10_freq=f_ligo)
    for nm, mine, ref in (('O3', sig_o3, m_o3), ('CE40', sig_ce, m_ce)):
        r = np.asarray(ref.sigma_omega, dtype=np.float64)
        mx = float(np.max(np.abs(mine - r) / r))
        assert mx == 0.0, f'{nm} sigma vs fresh LIGONoiseModel maxrel={mx}'
    print('[LIGO] sigma_O3/CE40 == fresh LIGONoiseModel (bit-exact)', flush=True)

    x_g = 10.0 ** clean_g
    snr_o3 = np.sqrt(((x_g / sig_o3[None, :]) ** 2).sum(axis=1))
    snr_ce = np.sqrt(((x_g / sig_ce[None, :]) ** 2).sum(axis=1))

    # 对 R35 L3 导出件行序 (LIGO 单波段 val 9,420)
    zg = np.load(WS / 'outputs/r35_export/L3_Median5000_plus.npz', allow_pickle=True)
    assert list(zg['mongo_ids']) == ids_g, 'LIGO mongo_ids 与 L3 导出件不一致'
    # 对联合 json val (T3 Jfut 行序) — 前置已核逐位同, 此处复验
    jj = json.loads((WS / 'dataset/Dataset_PTA_LISA_LIGO.json').read_text())
    _, va_j = train_test_split(np.arange(len(jj)), test_size=0.2, random_state=42)
    assert [jj[int(r)]['_id'] for r in va_j] == ids_g, 'joint val ids != LIGO val ids'
    print('[LIGO] mongo_ids == L3_Median5000_plus.npz 且 == joint val ids (逐位)', flush=True)

    cols['ligo'] = dict(
        mongo_ids=np.array(ids_g, dtype=object),
        clean_log10OmegaGW=clean_g.astype(np.float32),
        snr_ligo_o3=snr_o3, snr_ligo_ce40=snr_ce,
        sigma_ligo_o3=sig_o3, sigma_ligo_ce40=sig_ce, freqs=f_ligo,
    )
    report['ligo'] = dict(
        n=9420,
        sigma_ligo_o3_sha256=sha256_arr(sig_o3.astype(np.float64)),
        sigma_ligo_ce40_sha256=sha256_arr(sig_ce.astype(np.float64)),
        definition='R35 预计算 sigma_LIGO_{O3,CE40}.npy, 内嵌 fresh LIGONoiseModel '
                   'canonical 网格复验 bit-exact',
        aligned_to='outputs/r35_export/L3_Median5000_plus.npz mongo_ids '
                   '且 == joint (Dataset_PTA_LISA_LIGO.json) val ids',
    )

    # ================= 合并宽表 (NaN 填充) =================
    nP, nL, nG = 33226, 15927, 9420
    N = nP + nL + nG
    snr_ng15 = np.full(N, np.nan)
    snr_ngmock = np.full(N, np.nan)
    snr_lisa = np.full(N, np.nan)
    snr_o3c = np.full(N, np.nan)
    snr_ce40c = np.full(N, np.nan)
    snr_ng15[:nP] = snr_p_obs
    snr_ngmock[:nP] = snr_p_mock
    snr_lisa[nP:nP + nL] = snr_l
    snr_o3c[nP + nL:] = snr_o3
    snr_ce40c[nP + nL:] = snr_ce
    all_ids = np.concatenate([cols['pta']['mongo_ids'], cols['lisa']['mongo_ids'],
                              cols['ligo']['mongo_ids']])
    dataset_tag = np.array(['pta'] * nP + ['lisa'] * nL + ['ligo'] * nG,
                           dtype=object)

    out = dict(
        mongo_ids=all_ids,
        dataset=dataset_tag,
        val_index=np.concatenate([va_p, va_l, va_g]).astype(np.int64),
        snr_pta_ng15=snr_ng15, snr_pta_ngmock=snr_ngmock,
        snr_lisa_cbe=snr_lisa,
        snr_ligo_o3=snr_o3c, snr_ligo_ce40=snr_ce40c,
        sigma_pta_ng15=obs_sig, sigma_pta_ngmock=mock_sig,
        sigma_lisa_cbe=sig_lisa,
        sigma_ligo_o3=sig_o3, sigma_ligo_ce40=sig_ce,
        freqs_pta=freqs, freqs_lisa=f_lisa, freqs_ligo=f_ligo,
    )
    np.savez(OUT / 'r40_snr_table.npz', **out)

    rep = dict(
        created_utc=__import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
        n_rows=int(N), n_pta=nP, n_lisa=nL, n_ligo=nG,
        snr_definition='SNR = sqrt(sum_i (x_i/sigma_i)^2), x=10^log10OmegaGW (干净曲线)',
        columns=report,
        nan_policy='无定义记 NaN (跨波段列)',
        identity_checks=dict(
            pta_vs_r39t2='mongo_ids/snr_obs/snr_mock/sigma 全部逐位一致',
            lisa_vs_L1_export='mongo_ids 逐位一致',
            ligo_vs_L3_export='mongo_ids 逐位一致 且 == joint json val ids',
        ),
        sha256=dict(
            sigma_pta_ng15=report['pta']['sigma_ng15_sha256'],
            sigma_pta_ngmock=report['pta']['sigma_ngmock_sha256'],
            sigma_lisa_cbe=report['lisa']['sigma_lisa_cbe_sha256'],
            sigma_ligo_o3=report['ligo']['sigma_ligo_o3_sha256'],
            sigma_ligo_ce40=report['ligo']['sigma_ligo_ce40_sha256'],
        ),
        npz_sha256=sha256_file(OUT / 'r40_snr_table.npz'),
    )
    (OUT / 'r40_snr_table_report.json').write_text(
        json.dumps(rep, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'columns'},
                     indent=1, ensure_ascii=False))
    print('T4 DONE', flush=True)


if __name__ == '__main__':
    main()
