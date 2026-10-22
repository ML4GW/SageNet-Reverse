"""
数据预处理脚本 v2: 先判断覆盖范围 → 在完整曲线上插值 → 按频段截取

核心改进 (相对 v1):
  - 消除外推: 不再先裁切再插值, 而是在原始完整曲线上直接对目标频率点插值
  - 分级生成: 根据每条曲线的 f_max 决定能生成哪些频段的样本
    Level 3: f_max >= LIGO_max → PTA + LISA + LIGO + Joint(三频段)
    Level 2: f_max >= LISA_max → PTA + LISA + Joint(PTA+LISA)
    Level 1: f_max >= PTA_max → 仅 PTA
    Level 0: f_max < PTA_max  → 丢弃
  - 一次遍历: 只扫描一遍数据库, 同时生成所有数据集
  - 纯插值, 无外推: interp1d 使用 bounds_error=True, 任何越界立即报错

使用:
  python preprocess_v2.py
"""

import pymongo
import json
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import defaultdict

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["solve_plus"]
collection = db["main"]


# ============================================================
# 频率网格定义 (与 v1 完全一致)
# ============================================================

# ---- PTA: NANOGrav 15yr 真实实验频率点 ----
PTA_FREQ_HZ = np.array([
    1.97682646e-09, 3.95365292e-09, 5.93047937e-09, 7.90730583e-09,
    9.88413229e-09, 1.18609587e-08, 1.38377852e-08, 1.58146117e-08,
    1.77914381e-08, 1.97682646e-08, 2.17450910e-08, 2.37219175e-08,
    2.56987439e-08, 2.76755704e-08
])
PTA_FREQ_LOG = np.log10(PTA_FREQ_HZ)

# ---- LIGO: 从 O3 CC 谱池化下采样 546 点 ----
LIGO_FREQ_HZ = pd.read_csv('freqs_downsampled.csv')['f'].values
LIGO_FREQ_LOG = np.log10(LIGO_FREQ_HZ)

# ---- LISA: log10 均匀分布 500 点 ----
LISA_NPTS = 500
LISA_FREQ_LOG = np.linspace(-3, -1, LISA_NPTS)

# ---- 频段配置 (按频率从低到高排列) ----
BAND_CONFIGS = {
    'PTA': {
        'min': PTA_FREQ_LOG[0],
        'max': PTA_FREQ_LOG[-1],
        'points': len(PTA_FREQ_LOG),
        'grid': PTA_FREQ_LOG,
        'downsample': 'interp',
    },
    'LISA': {
        'min': LISA_FREQ_LOG[0],
        'max': LISA_FREQ_LOG[-1],
        'points': LISA_NPTS,
        'grid': LISA_FREQ_LOG,
        'downsample': 'pool',
    },
    'LIGO': {
        'min': LIGO_FREQ_LOG[0],
        'max': LIGO_FREQ_LOG[-1],
        'points': len(LIGO_FREQ_LOG),
        'grid': LIGO_FREQ_LOG,
        'downsample': 'pool',
    },
}

# 频段按频率升序, 用于分级判断
BANDS_ORDERED = ['PTA', 'LISA', 'LIGO']

# 导出字段
EXPORT_FIELDS = {
    '_id': 1, 'f': 1, 'log10OmegaGW': 1,
    'r': 1, 'n_t': 1, 'kappa10': 1, 'T_re': 1, 'DN_re': 1,
    'Omega_bh2': 1, 'Omega_ch2': 1, 'H0': 1, 'A_s': 1
}

PARAM_KEYS = ['r', 'n_t', 'kappa10', 'T_re', 'DN_re',
              'Omega_bh2', 'Omega_ch2', 'H0', 'A_s']


# ============================================================
# 辅助函数
# ============================================================

def clean_and_sort(doc):
    """清洗原始文档: 排序、自动检测频率单位, 返回 (f_log, omega) 或 (None, None)"""
    f_arr = np.array(doc['f'], dtype=np.float64)
    omega_arr = np.array(doc['log10OmegaGW'], dtype=np.float64)

    if len(f_arr) < 2:
        return None, None

    # 确保升序
    if f_arr[0] > f_arr[-1]:
        f_arr = f_arr[::-1]
        omega_arr = omega_arr[::-1]

    # 去重 (interp1d 要求严格单调)
    _, unique_idx = np.unique(f_arr, return_index=True)
    if len(unique_idx) < len(f_arr):
        f_arr = f_arr[unique_idx]
        omega_arr = omega_arr[unique_idx]

    # 自动检测: 如果值域在 (0, 1) 附近, 可能是线性 Hz 而非 log10(Hz)
    if f_arr[0] > 0 and f_arr[-1] < 1.0:
        f_arr = np.log10(f_arr)

    return f_arr, omega_arr


def determine_level(f_min, f_max):
    """根据曲线的频率覆盖范围确定可生成的最高级别。

    要求:
      - f_min 必须 <= PTA_min (最低频段的起点), 否则丢弃
      - f_max 决定能覆盖到哪个频段

    Returns:
        level (int): 0=丢弃, 1=仅PTA, 2=PTA+LISA, 3=PTA+LISA+LIGO
        covered_bands (list[str]): 该样本可以生成的频段列表
    """
    pta_min = BAND_CONFIGS['PTA']['min']
    pta_max = BAND_CONFIGS['PTA']['max']
    lisa_max = BAND_CONFIGS['LISA']['max']
    ligo_max = BAND_CONFIGS['LIGO']['max']

    # f_min 必须覆盖到 PTA 的起点
    if f_min > pta_min:
        return 0, []

    # 根据 f_max 分级
    if f_max >= ligo_max:
        return 3, ['PTA', 'LISA', 'LIGO']
    elif f_max >= lisa_max:
        return 2, ['PTA', 'LISA']
    elif f_max >= pta_max:
        return 1, ['PTA']
    else:
        return 0, []


def interpolate_on_full_curve(f_full, omega_full, target_grid, method='interp'):
    """在完整原始曲线上对目标频率点进行插值 (不做任何外推)。

    Args:
        f_full:      原始完整 log10(f) 数组 (已排序, 已去重)
        omega_full:  原始完整 log10(Ω) 数组
        target_grid: 目标频率网格 (log10 Hz)
        method:      'interp' = 三次样条, 'pool' = bin-average 池化

    Returns:
        omega_on_grid: 目标网格上的 log10(Ω) 值
    """
    if method == 'pool':
        return pool_average_on_full(f_full, omega_full, target_grid)
    else:
        # 三次样条插值, bounds_error=True 防止任何外推
        interp_func = interp1d(f_full, omega_full, kind='cubic',
                               bounds_error=True)
        return interp_func(target_grid)


def pool_average_on_full(f_full, omega_full, target_grid):
    """在完整曲线上做 bin-average 池化。

    与 v1 的区别: 输入是完整曲线而非裁切后的片段, 所以:
      1. 不需要外推
      2. 空 bin 用完整曲线的插值填充 (仍然是内插, 不是外推)
    """
    n = len(target_grid)

    # 统计目标频段内的原始点数
    in_band = (f_full >= target_grid[0]) & (f_full <= target_grid[-1])
    n_raw_in_band = in_band.sum()

    # 插值函数 (在完整曲线上, 不会外推)
    interp_func = interp1d(f_full, omega_full, kind='cubic', bounds_error=True)

    # 如果频段内原始点太稀疏, 直接用插值
    if n_raw_in_band < n * 1.5:
        return interp_func(target_grid)

    # 逐 bin 池化
    omega_pooled = np.zeros(n)
    edges = np.zeros(n + 1)
    edges[0] = target_grid[0] - 0.5 * (target_grid[1] - target_grid[0])
    edges[-1] = target_grid[-1] + 0.5 * (target_grid[-1] - target_grid[-2])
    for i in range(1, n):
        edges[i] = 0.5 * (target_grid[i - 1] + target_grid[i])

    for i in range(n):
        mask = (f_full >= edges[i]) & (f_full < edges[i + 1])
        if mask.sum() > 0:
            omega_pooled[i] = np.mean(omega_full[mask])
        else:
            omega_pooled[i] = interp_func(target_grid[i])

    return omega_pooled


def build_single_band_doc(doc_id, omega_on_grid, band_name, raw_doc):
    """构建单频段输出文档"""
    cfg = BAND_CONFIGS[band_name]
    return {
        '_id': doc_id,
        'f': cfg['grid'].tolist(),
        'log10OmegaGW': omega_on_grid.tolist(),
        **{k: raw_doc.get(k) for k in PARAM_KEYS}
    }


def build_joint_doc(doc_id, band_results, raw_doc):
    """构建联合频段输出文档

    Args:
        band_results: dict  {band_name: omega_on_grid}
    """
    bands = list(band_results.keys())
    concatenated_f = []
    concatenated_omega = []
    band_indices = {}
    current_idx = 0

    for b_name in BANDS_ORDERED:
        if b_name not in band_results:
            continue
        cfg = BAND_CONFIGS[b_name]
        omega = band_results[b_name]
        concatenated_f.extend(cfg['grid'].tolist())
        concatenated_omega.extend(omega.tolist())
        band_indices[b_name] = [current_idx, current_idx + cfg['points']]
        current_idx += cfg['points']

    return {
        '_id': doc_id,
        'f': concatenated_f,
        'log10OmegaGW': concatenated_omega,
        'band_structure': band_indices,
        **{k: raw_doc.get(k) for k in PARAM_KEYS}
    }


# ============================================================
# 主处理流程
# ============================================================

def process_all():
    """一次遍历数据库, 根据每条曲线的覆盖范围分级生成所有数据集。"""

    # 输出容器
    single_band_data = defaultdict(list)   # {'PTA': [...], 'LISA': [...], 'LIGO': [...]}
    joint_data = defaultdict(list)          # {'PTA_LISA': [...], 'PTA_LISA_LIGO': [...]}

    # 统计
    stats = {
        'total_scanned': 0,
        'level_0': 0,   # 丢弃
        'level_1': 0,   # 仅 PTA
        'level_2': 0,   # PTA + LISA
        'level_3': 0,   # PTA + LISA + LIGO
        'errors': 0,
    }

    print("\n" + "=" * 60)
    print("开始处理: 一次遍历, 分级生成")
    print("=" * 60)
    print(f"频段配置:")
    for name, cfg in BAND_CONFIGS.items():
        print(f"  {name}: {cfg['points']} pts, "
              f"[{cfg['min']:.4f}, {cfg['max']:.4f}] log10(Hz), "
              f"策略: {cfg['downsample']}")

    cursor = collection.find({"f": {"$exists": True}}, EXPORT_FIELDS).batch_size(1000)

    for doc in cursor:
        stats['total_scanned'] += 1
        doc_id = str(doc['_id'])

        # ---- Step 1: 清洗 & 排序 ----
        f_full, omega_full = clean_and_sort(doc)
        if f_full is None:
            stats['level_0'] += 1
            continue

        # ---- Step 2: 判断覆盖级别 ----
        f_min, f_max = f_full[0], f_full[-1]
        level, covered_bands = determine_level(f_min, f_max)

        if level == 0:
            stats['level_0'] += 1
            continue

        stats[f'level_{level}'] += 1

        # ---- Step 3: 在完整曲线上对各频段插值 ----
        try:
            band_results = {}  # {band_name: omega_on_grid}

            for b_name in covered_bands:
                cfg = BAND_CONFIGS[b_name]
                omega_on_grid = interpolate_on_full_curve(
                    f_full, omega_full, cfg['grid'], method=cfg['downsample'])
                band_results[b_name] = omega_on_grid

                # 生成单频段样本
                single_doc = build_single_band_doc(doc_id, omega_on_grid, b_name, doc)
                single_band_data[b_name].append(single_doc)

            # ---- Step 4: 生成联合样本 ----
            if level >= 3:
                # PTA + LISA + LIGO 三频段联合
                joint_doc = build_joint_doc(doc_id, band_results, doc)
                joint_data['PTA_LISA_LIGO'].append(joint_doc)

                # 同时生成 PTA + LISA 联合 (三频段样本也可以做双频段)
                joint_doc_2 = build_joint_doc(
                    doc_id, {k: band_results[k] for k in ['PTA', 'LISA']}, doc)
                joint_data['PTA_LISA'].append(joint_doc_2)

            elif level == 2:
                # 仅 PTA + LISA 联合
                joint_doc = build_joint_doc(doc_id, band_results, doc)
                joint_data['PTA_LISA'].append(joint_doc)

        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 10:
                print(f"  [Error] ID {doc_id}: {e}")

        # 进度
        if stats['total_scanned'] % 5000 == 0:
            n_pta = len(single_band_data.get('PTA', []))
            n_lisa = len(single_band_data.get('LISA', []))
            n_ligo = len(single_band_data.get('LIGO', []))
            print(f"  已扫描 {stats['total_scanned']}  |  "
                  f"PTA:{n_pta}  LISA:{n_lisa}  LIGO:{n_ligo}")

    # ---- 打印统计 ----
    print(f"\n{'='*60}")
    print("处理完成 — 统计")
    print(f"{'='*60}")
    print(f"  总扫描: {stats['total_scanned']}")
    print(f"  Level 3 (PTA+LISA+LIGO): {stats['level_3']}")
    print(f"  Level 2 (PTA+LISA):      {stats['level_2']}")
    print(f"  Level 1 (仅PTA):         {stats['level_1']}")
    print(f"  Level 0 (丢弃):          {stats['level_0']}")
    print(f"  错误:                     {stats['errors']}")

    # ---- 保存 ----
    print(f"\n{'='*60}")
    print("保存数据集")
    print(f"{'='*60}")

    # 单频段
    for b_name in BANDS_ORDERED:
        data = single_band_data.get(b_name, [])
        if len(data) == 0:
            print(f"  {b_name}: 空, 跳过")
            continue
        n_pts = BAND_CONFIGS[b_name]['points']
        fname = f"{b_name}_fixed_{n_pts}pts.json"
        with open(fname, 'w') as fp:
            json.dump(data, fp)
        print(f"  {b_name}: {len(data)} 样本 → {fname}")

    # 联合
    for jname, data in joint_data.items():
        if len(data) == 0:
            print(f"  Joint_{jname}: 空, 跳过")
            continue
        fname = f"Dataset_{jname}.json"
        with open(fname, 'w') as fp:
            json.dump(data, fp)
        print(f"  Joint_{jname}: {len(data)} 样本 → {fname}")

    # ---- 质量检查: 绘图 ----
    print(f"\n{'='*60}")
    print("质量检查")
    print(f"{'='*60}")

    # 检查每个单频段的值域
    for b_name in BANDS_ORDERED:
        data = single_band_data.get(b_name, [])
        if len(data) == 0:
            continue
        all_omega = np.concatenate([np.array(d['log10OmegaGW']) for d in data])
        print(f"  {b_name} log10(Ω) 统计:")
        print(f"    范围: [{all_omega.min():.4f}, {all_omega.max():.4f}]")
        print(f"    均值: {all_omega.mean():.4f}")
        print(f"    中位数: {np.median(all_omega):.4f}")
        print(f"    std:  {all_omega.std():.4f}")
        n_abnormal = np.sum((all_omega < -64) | (all_omega > -1))
        pct = 100.0 * n_abnormal / len(all_omega)
        print(f"    超出 [-64, -1] 的比例: {pct:.2f}%")

    # 绘制样本图
    for b_name in BANDS_ORDERED:
        data = single_band_data.get(b_name, [])
        if len(data) == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        n_plot = min(200, len(data))
        for i in range(n_plot):
            ax.plot(data[i]['f'], data[i]['log10OmegaGW'],
                    alpha=0.3, linewidth=0.5)
        ax.set_title(f"{b_name} ({len(data)} samples, showing {n_plot})")
        ax.set_xlabel('log10(f/Hz)')
        ax.set_ylabel('log10(ΩGW)')
        ax.set_ylim(-70, 5)
        plt.tight_layout()
        plt.savefig(f"{b_name}_check.png", dpi=150)
        print(f"  检查图: {b_name}_check.png")
        plt.close()

    # 联合样本检查图
    for jname, data in joint_data.items():
        if len(data) == 0:
            continue
        sample = data[0]
        bands_in = list(sample['band_structure'].keys())
        fig, axes = plt.subplots(1, len(bands_in),
                                 figsize=(5 * len(bands_in), 4))
        if len(bands_in) == 1:
            axes = [axes]
        for i, b_name in enumerate(bands_in):
            idx_s, idx_e = sample['band_structure'][b_name]
            axes[i].plot(sample['f'][idx_s:idx_e],
                         sample['log10OmegaGW'][idx_s:idx_e],
                         '.', markersize=2)
            axes[i].set_title(f"{b_name} ({idx_e - idx_s} pts)")
            axes[i].set_xlabel('log10(f/Hz)')
            axes[i].set_ylabel('log10(ΩGW)')
        plt.suptitle(f"Joint {jname} — Sample 0")
        plt.tight_layout()
        plt.savefig(f"Dataset_{jname}_sample.png", dpi=150)
        print(f"  检查图: Dataset_{jname}_sample.png")
        plt.close()

    print("\n全部完成!")


if __name__ == "__main__":
    process_all()