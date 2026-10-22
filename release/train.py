"""
统一的 Normalizing Flow 训练代码
支持通过 JSON 配置文件或命令行参数配置

使用示例:
  # 使用配置文件 (推荐)
  python train.py --config config_slope.json
  python train.py --config config_conti.json
  python train.py --config config_single.json

  # 命令行参数 (配置文件优先级更高)
  python train.py --dataset-type concat --epochs 2000
"""

import json
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.interpolate import interp1d
import os
import argparse


# ==========================================
# 工具: 同时写到 stdout 和文件 (用于 package training log)
# ==========================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


# ==========================================
# 0-A. LIGO 物理噪声模型
# ==========================================

class LIGONoiseModel:
    """LIGO 频段的物理噪声模型

    支持两种数据源:
      - 'cc_spectrum': O3 交叉关联谱 (C_O1_O2_O3.dat)
        格式: freq[Hz]  C(f)  sigma(f), 其中 sigma 已经是 ΩGW 单位
      - 'aplus_asd':  A+ 设计灵敏度 (AplusDesign.txt)
        格式: freq[Hz]  ASD[1/sqrt(Hz)], 需转换为 ΩGW 等效

    噪声注入模式:
      - 'physical': 在线性 ΩGW 空间添加 N(0, σ_Ω(f))，再转回 log10
                    物理上最准确；信号远低于噪声底时观测值 ≈ |noise|
      - 'logscale': 在 log10(ΩGW) 空间添加频率加权噪声
                    保留信号结构的同时引入真实的频率依赖性

    Args:
        noise_file:   噪声数据文件路径
        noise_type:   'cc_spectrum' 或 'aplus_asd'
        target_log10_freq: LIGO 目标频率网格, log10(f/Hz), shape (L,)
        noise_scale:  整体噪声缩放因子 (1.0 = 原始幅度)
        injection_mode: 'physical' 或 'logscale'
        logscale_base: logscale 模式下 log10 空间的基础噪声标准差
    """

    def __init__(self, noise_file, noise_type='cc_spectrum',
                 target_log10_freq=None, noise_scale=1.0,
                 injection_mode='physical', logscale_base=1.0):
        self.noise_scale = noise_scale
        self.injection_mode = injection_mode
        self.logscale_base = logscale_base

        if target_log10_freq is None:
            raise ValueError("target_log10_freq is required")

        self.target_log10_freq = np.asarray(target_log10_freq, dtype=np.float64)

        # 加载并插值到目标网格
        if noise_type == 'cc_spectrum':
            self.sigma_omega = self._load_cc_spectrum(noise_file)
        elif noise_type == 'aplus_asd':
            self.sigma_omega = self._load_aplus_asd(noise_file)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # 缩放
        self.sigma_omega = self.sigma_omega * self.noise_scale

        # logscale 模式: log 压缩的频率权重
        # 使用 log10(σ) 的归一化, 避免线性比值跨越数个量级
        log_sigma = np.log10(np.maximum(self.sigma_omega, 1e-30))
        log_min = log_sigma.min()
        log_max = log_sigma.max()
        if log_max > log_min:
            # 映射到 [1, max_ratio], 对数均匀
            max_ratio = 10.0  # 最嘈杂频率的权重是最安静频率的 10 倍
            self.freq_weight = 1.0 + (max_ratio - 1.0) * (log_sigma - log_min) / (log_max - log_min)
        else:
            self.freq_weight = np.ones_like(self.sigma_omega)
        self.freq_weight = self.freq_weight.astype(np.float32)

        print(f"[LIGO Noise] type={noise_type}, scale={noise_scale}, "
              f"mode={injection_mode}")
        print(f"  σ_Ω range: [{self.sigma_omega.min():.3e}, "
              f"{self.sigma_omega.max():.3e}]")
        if injection_mode == 'logscale':
            print(f"  logscale_base={logscale_base}, "
                  f"freq_weight range: [{self.freq_weight.min():.3f}, "
                  f"{self.freq_weight.max():.3f}]")

    # ------------------------------------------------------------------
    def _load_cc_spectrum(self, filepath):
        """从 CC 谱加载 σ_Ω(f)，bin-average 到目标网格。

        处理:
        1. inf / 负值 → 在 log 空间插值填补
        2. Bin-average: 对应目标网格每个 bin 内的原始点取 RMS 均值,
           并除以 √N (N 个独立 bin 平均后 σ 降低)
        """
        data = np.loadtxt(filepath, skiprows=1)
        freq_hz = data[:, 0]
        sigma_raw = data[:, 2]

        # 清理 inf / nan / 非正值
        valid = np.isfinite(sigma_raw) & (sigma_raw > 0)
        log_sigma_interp = np.interp(
            np.log10(freq_hz),
            np.log10(freq_hz[valid]),
            np.log10(sigma_raw[valid])
        )
        sigma_clean = 10.0 ** log_sigma_interp

        # Bin-average 到目标网格
        target_hz = 10.0 ** self.target_log10_freq
        n = len(target_hz)
        sigma_binned = np.zeros(n, dtype=np.float64)

        for i in range(n):
            # 几何中点作为 bin 边界 (log 空间等距)
            if i == 0:
                f_lo = freq_hz.min()
            else:
                f_lo = np.sqrt(target_hz[i] * target_hz[i - 1])
            if i == n - 1:
                f_hi = freq_hz.max()
            else:
                f_hi = np.sqrt(target_hz[i] * target_hz[i + 1])

            mask = (freq_hz >= f_lo) & (freq_hz < f_hi)
            n_bins = mask.sum()

            if n_bins > 0:
                # RMS 均值 / √N (bin-average 降噪)
                sigma_binned[i] = np.sqrt(np.mean(sigma_clean[mask] ** 2)) / np.sqrt(n_bins)
            else:
                sigma_binned[i] = 10.0 ** np.interp(
                    self.target_log10_freq[i],
                    np.log10(freq_hz[valid]),
                    np.log10(sigma_raw[valid])
                )

        return sigma_binned.astype(np.float32)

    # ------------------------------------------------------------------
    def _load_aplus_asd(self, filepath):
        """从 A+ ASD 计算等效 σ_Ω(f) 并插值到目标网格。

        使用单探测器近似: σ_Ω(f) ∝ f³ Sₙ(f)
        其中 Sₙ(f) = ASD²(f), 系数 = 2π²/(3H₀²)

        注: 这是灵敏度代理, 真正的 CC 搜索 σ 还依赖 γ(f), T, Δf 等。
        """
        data = np.loadtxt(filepath)
        freq_hz = data[:, 0]
        asd = data[:, 1]

        H0_si = 67.4e3 / 3.0857e22  # H₀ 单位: Hz
        sigma_proxy = (2.0 * np.pi ** 2 / (3.0 * H0_si ** 2)) * freq_hz ** 3 * asd ** 2

        # log 空间插值到目标网格
        target_hz = 10.0 ** self.target_log10_freq
        interp_fn = interp1d(np.log10(freq_hz), np.log10(sigma_proxy),
                             kind='linear', fill_value='extrapolate')
        sigma_on_grid = 10.0 ** interp_fn(self.target_log10_freq)

        return sigma_on_grid.astype(np.float32)

    # ------------------------------------------------------------------
    def inject(self, log10_omega, log10_freq=None):
        """对单条 LIGO 频谱注入噪声。

        Args:
            log10_omega: 干净的 log10(ΩGW), shape (L,)
            log10_freq:  log10(f/Hz), shape (L,), physical 模式下可选
        Returns:
            含噪的 log10(ΩGW), shape (L,)
        """
        if self.injection_mode == 'physical':
            return self._inject_physical(log10_omega)
        else:
            return self._inject_logscale(log10_omega)

    def _inject_physical(self, log10_omega):
        """在线性 ΩGW 空间添加高斯噪声 → log10。

        Ω_obs(f) = Ω_true(f) + N(0, σ_Ω(f))
        若 Ω_obs < 0 取绝对值 (cross-correlation 估计量可为负)
        """
        omega = np.float64(10.0) ** log10_omega.astype(np.float64)
        noise = np.random.normal(0.0, self.sigma_omega.astype(np.float64))
        omega_obs = omega + noise

        omega_obs = np.abs(omega_obs)
        omega_obs = np.maximum(omega_obs, 1e-300)
        return np.log10(omega_obs).astype(np.float32)

    def _inject_logscale(self, log10_omega):
        """在 log10 空间添加频率加权噪声。

        δ(log10 Ω)(f) = N(0, 1) × freq_weight(f) × logscale_base
        freq_weight 归一化为中位数=1, logscale_base 控制幅度。
        """
        noise = np.random.normal(0.0, 1.0, log10_omega.shape).astype(np.float32)
        return log10_omega + noise * self.freq_weight * self.logscale_base


# ==========================================
# 0-B. PTA 物理噪声模型
# ==========================================

class PTANoiseModel:
    """PTA 频段的物理噪声模型 (NANOGrav 15yr free spectrum)

    数据来源: Zenodo 10344086  NANOGrav15yr/
      - freqs.npy            30 个频率 bin (Hz), Δf = 1/T_obs ≈ 1.977e-9 Hz
      - log10rhogrid.npy     log10(ρ) 网格 (10000 个等间距点, [-15.5, -1.0])
      - density_mock.npy     shape (1, 30, 10000), 存的是 log(p(log10ρ_i | data))
                             由 NGmock free spectrum 给出 (Li & Shapiro 2025, Sec.3)
                             —— mock 版本, 对应 design sensitivity
      - density.npy          shape (1, 30, 10000), 同上, 但来自真实
                             NANOGrav 15yr observation —— observed 版本

    物理路径 (mode='density_mock' 或 'density_obs', 推荐):
      1. 对每个 bin i ∈ [0, 14), 从后验 p_i(log10ρ) 采样 → ρ_i (秒)
      2. 通过 Li & Shapiro 2025 Eq.(3) 转 Ω_GW:
             Ω_i(f_i) = (8π⁴/H₀²) · T_obs · f_i^5 · ρ_i²
      3. ★ 减去每 bin 的 Ω 均值 (跨后验), 得到纯噪声偏差:
             Ω_noise(f_i) = Ω_i_sampled - <Ω_i>_posterior
         直觉: data_NG - μ_NG = Noise_NG. 后验本身代表"信号+噪声"的总和,
         我们要的是噪声分量, 必须减掉均值.
      4. 在线性 Ω 空间注入: Ω_obs = Ω_signal + Ω_noise, 再回 log10

    Args:
        noise_dir:        包含 freqs.npy / log10rhogrid.npy / density{_mock}.npy 的目录
        mode:             'density_mock' → density_mock.npy (推荐, mock 版)
                          'density_obs'  → density.npy (NANOGrav 15yr 真实观测)
                          'white' / 'posterior' (legacy)
        noise_scale:      整体缩放因子, 作用在 Ω_noise 偏差上
                          (curriculum learning: 0=无噪声, 1=全噪声)
        n_bins_use:       使用前几个频率 bin (默认 14, 与 PTA 数据集对齐)
        per_sample_noise: True  → 每个训练样本独立采样 ρ (default)
                          False → 一次 refresh 整个 batch 共享同一组 ρ
        H0_fiducial:      用于 ρ→Ω 转换的 Hubble 常数 (km/s/Mpc), 默认 67.4 (Planck18)
        T_obs:            观测时长 (秒). 默认 None → 从 freqs.npy 反推 1/Δf
    """

    # mode → density 文件名映射
    _DENSITY_FILE_MAP = {
        'density_mock': 'density_mock.npy',
        'density_obs':  'density.npy',
    }

    # ρ→Ω 转换里要用 H0 的 SI 单位 (s^-1)
    _KM_PER_MPC = 3.0856775814913673e19  # 1 Mpc in km
    # numpy 1.x: trapz; numpy 2.x: trapezoid
    _trapz = staticmethod(getattr(np, 'trapezoid', getattr(np, 'trapz', None)))

    def __init__(self, noise_dir,
                 mode='density_mock',
                 noise_scale=1.0,
                 n_bins_use=14,
                 per_sample_noise=True,
                 H0_fiducial=67.4,
                 T_obs=None):

        self.mode = mode
        self.noise_scale = float(noise_scale)
        self.per_sample_noise = bool(per_sample_noise)
        self.H0_fiducial = float(H0_fiducial)
        # 标记: density_mock 和 density_obs 走同一物理路径
        self.is_density_mode = mode in self._DENSITY_FILE_MAP

        freqs_all = np.load(os.path.join(noise_dir, 'freqs.npy'))         # (30,)
        log10rho  = np.load(os.path.join(noise_dir, 'log10rhogrid.npy'))  # (10000,)

        # 根据 mode 选择 density 文件
        if self.is_density_mode:
            density_filename = self._DENSITY_FILE_MAP[mode]
        else:
            # legacy 模式: 默认仍读 density_mock.npy 用于计算 sigma_per_bin
            density_filename = 'density_mock.npy'
        density_path = os.path.join(noise_dir, density_filename)
        if not os.path.isfile(density_path):
            raise FileNotFoundError(
                f"PTA density file not found: {density_path}\n"
                f"  mode='{mode}' expects file '{density_filename}' under {noise_dir}")
        density = np.load(density_path)                                    # (1,30,10000)

        # 兼容多种存储 layout
        if density.ndim == 3:
            density2d = density[0]                # (30, 10000)
        elif density.ndim == 2:
            density2d = density                   # (30, 10000) 或 (14, 10000)
        else:
            raise ValueError(f"density_mock.npy 维度异常: {density.shape}")

        self.n_bins = int(n_bins_use)
        if self.n_bins > density2d.shape[0]:
            raise ValueError(f"n_bins_use={self.n_bins} > density rows={density2d.shape[0]}")

        self.freqs    = freqs_all[:self.n_bins].astype(np.float64)        # (14,) Hz
        self.log10rho = log10rho.astype(np.float64)                       # (10000,)
        self.log_pdf  = density2d[:self.n_bins].astype(np.float64)        # (14, 10000)

        # T_obs: 默认从 freqs.npy 的等间距反推 (= 1/Δf)
        if T_obs is None:
            df = float(freqs_all[1] - freqs_all[0])
            self.T_obs = 1.0 / df
        else:
            self.T_obs = float(T_obs)

        # ρ→Ω 转换系数: K_i = (8π⁴/H₀²) T_obs f_i^5
        # H0 转 SI: H0_SI [s^-1] = H0 [km/s/Mpc] / (Mpc in km)
        H0_SI = self.H0_fiducial / self._KM_PER_MPC
        self.K_omega = (8.0 * np.pi**4 / H0_SI**2) * self.T_obs * self.freqs**5  # (14,)

        # ---- 预计算每个 bin 的 inverse-CDF 表 (10000 点 → 直接 np.interp 采样) ----
        # 在 log space 减最大值再 exp, 数值稳定
        pdf = np.exp(self.log_pdf - self.log_pdf.max(axis=1, keepdims=True))  # (14, G)
        # 用 trapezoid 归一化
        norm = self._trapz(pdf, self.log10rho, axis=1)                          # (14,)
        pdf  = pdf / norm[:, None]
        # 离散 CDF
        cdf = np.cumsum(pdf, axis=1)
        cdf -= cdf[:, :1]
        cdf /= cdf[:, -1:].clip(min=1e-300)
        self._cdf = cdf                                                       # (14, G)

        # ---- ★ 预计算每 bin 的 Ω 后验均值 (用于减均值得纯噪声偏差) ----
        # 物理: Noise_NG = data_NG - μ_NG. 后验代表"信号+噪声"的总和,
        # 减掉跨样本均值后得到的才是可加的纯噪声分量.
        # 用 Monte Carlo 在 noise_scale=1 下计算, 后续 sample 时再乘 noise_scale².
        N_mc = 50000
        u_mc = np.random.uniform(0.0, 1.0, size=(N_mc, self.n_bins))
        log10_rho_mc = np.empty_like(u_mc)
        for i in range(self.n_bins):
            log10_rho_mc[:, i] = np.interp(u_mc[:, i], self._cdf[i], self.log10rho)
        omega_mc = self.K_omega[None, :] * (10.0 ** log10_rho_mc) ** 2
        self.omega_mean = omega_mc.mean(axis=0).astype(np.float64)            # (14,)

        # 同时存一个 sanity 用的 σ_log10rho (后验宽度), 仅用于 legacy 模式
        mean = self._trapz(self.log10rho[None, :] * pdf, self.log10rho, axis=1)  # (14,)
        var  = self._trapz((self.log10rho[None, :] - mean[:, None])**2 * pdf,
                        self.log10rho, axis=1)
        self.sigma_per_bin = np.sqrt(np.maximum(var, 1e-30)).astype(np.float32) * self.noise_scale
        self.sigma_white = float(np.median(self.sigma_per_bin))

        # 报告
        rho_lo = self.log10rho[(cdf >= 0.025).argmax(axis=1)]
        rho_hi = self.log10rho[(cdf >= 0.975).argmax(axis=1)]
        print(f"[PTA Noise] mode={mode}, per_sample={self.per_sample_noise}, "
              f"n_bins={self.n_bins}, T_obs={self.T_obs:.3e} s ({self.T_obs/3.15576e7:.2f} yr)")
        print(f"[PTA Noise]   freqs: [{self.freqs[0]:.3e}, {self.freqs[-1]:.3e}] Hz")
        print(f"[PTA Noise]   log10ρ 95% CR: bin0=[{rho_lo[0]:.2f},{rho_hi[0]:.2f}]  "
              f"bin{self.n_bins-1}=[{rho_lo[-1]:.2f},{rho_hi[-1]:.2f}]")
        if self.is_density_mode:
            # 给一个等效 Ω_noise 量级 (用每 bin ρ 的中位数)
            rho_med = self.log10rho[(cdf >= 0.5).argmax(axis=1)]
            omega_med = self.K_omega * (10.0**rho_med)**2
            print(f"[PTA Noise]   density file: {density_filename}")
            print(f"[PTA Noise]   Ω (posterior median) range: "
                  f"[{omega_med.min():.3e}, {omega_med.max():.3e}]")
            print(f"[PTA Noise]   Ω (posterior mean) range:   "
                  f"[{self.omega_mean.min():.3e}, {self.omega_mean.max():.3e}]")
            print(f"[PTA Noise]   ★ noise = sampled - mean (zero-mean deviation)")
        elif mode in ('white', 'posterior'):
            print(f"[PTA Noise]   [legacy] σ_log10Ω: white={self.sigma_white:.4f}, "
                  f"per-bin range=[{self.sigma_per_bin.min():.4f}, "
                  f"{self.sigma_per_bin.max():.4f}]")

    # ------------------------------------------------------------------
    # 核心: 从后验中采样 log10ρ → Ω_noise (秒²)
    # ------------------------------------------------------------------
    def sample_omega_noise(self, n_samples):
        """采样得到 N 组纯噪声偏差 Ω_noise(f_i), shape (N, n_bins).

        ★ 关键: 返回的是 (Ω_sampled - <Ω>_posterior), 即零均值的纯噪声分量,
        而不是后验值本身. 这样才能正确叠加到任意理论信号上:
            Ω_obs = Ω_signal + Ω_noise   (Ω_noise 可正可负)

        noise_scale 作用在偏差上 (curriculum learning 友好):
            noise_scale=0 → 零噪声; noise_scale=1 → 全噪声.

        如果 per_sample_noise=False, 只生成一组然后 broadcast 到 N。
        """
        n_draw = n_samples if self.per_sample_noise else 1
        u = np.random.uniform(0.0, 1.0, size=(n_draw, self.n_bins))
        log10_rho = np.empty_like(u)
        for i in range(self.n_bins):
            log10_rho[:, i] = np.interp(u[:, i], self._cdf[i], self.log10rho)
        # noise_scale=1 下的 Ω_sampled
        omega_sampled = self.K_omega[None, :] * (10.0 ** log10_rho) ** 2  # (n_draw,14)
        # ★ 减均值得纯噪声偏差, 再用 noise_scale² 缩放
        omega_dev = (omega_sampled - self.omega_mean[None, :]) * (self.noise_scale ** 2)
        if not self.per_sample_noise:
            omega_dev = np.broadcast_to(omega_dev, (n_samples, self.n_bins)).copy()
        return omega_dev.astype(np.float64)

    def inject(self, log10_omega):
        """单样本注入 (fallback, 兼容 __getitem__ 慢路径).

        Args:
            log10_omega: shape (14,)  log10(Ω_signal)
        Returns:
            shape (14,) log10(Ω_obs) = log10(Ω_signal + Ω_noise)
        """
        if self.is_density_mode:
            omega_lin = 10.0 ** log10_omega.astype(np.float64)
            omega_noise = self.sample_omega_noise(1)[0]                      # (14,)
            omega_obs = np.maximum(omega_lin + omega_noise, 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        # ---- legacy modes (log-space additive Gaussian) ----
        if self.mode == 'white':
            noise = np.random.normal(0.0, self.sigma_white, log10_omega.shape)
        else:  # 'posterior'
            noise = np.random.normal(0.0, self.sigma_per_bin[:len(log10_omega)])
        return (log10_omega + noise).astype(np.float32)

    def validate_freqs(self, freq_log10_dataset, atol=0.02):
        """检查数据集前 14 个频率与 freqs.npy 是否对齐 (log10 空间)."""
        f_data = 10.0 ** np.asarray(freq_log10_dataset[:self.n_bins], dtype=np.float64)
        if not np.allclose(np.log10(f_data), np.log10(self.freqs), atol=atol):
            print(f"[PTA Noise] WARNING: dataset freqs vs noise freqs mismatch!")
            print(f"  data:  {f_data}")
            print(f"  noise: {self.freqs}")
        else:
            print(f"[PTA Noise]   freq alignment OK (max log10 diff "
                  f"{np.abs(np.log10(f_data)-np.log10(self.freqs)).max():.4f})")


# ==========================================
# 0-C. LISA 物理噪声模型
# ==========================================

class LISANoiseModel:
    """LISA 频段的物理噪声模型

    支持两种数据源:
      - 'analytical': 基于 Robson+ 2019 (arXiv:1803.01944) 的解析 PSD
                      参数来自 LISA SciRD (OMS=15 pm/√Hz, acc=3 fm/s²/√Hz)
      - 'datafile':   外部灵敏度曲线文件 (freq[Hz], ASD[1/√Hz])

    噪声注入模式与 LIGONoiseModel 相同 (physical / logscale)。

    额外参数:
        T_obs:       观测时长 [yr], 用于银河前景混淆噪声 (默认 4 yr)
        include_confusion: 是否包含银河前景混淆噪声 (默认 True)

    Args:
        target_log10_freq: LISA 目标频率网格 log10(f/Hz), shape (L,)
        noise_source:      'analytical' 或文件路径
        noise_scale:       整体缩放因子
        injection_mode:    'physical' 或 'logscale'
        logscale_base:     logscale 模式下的基础 σ
        T_obs:             观测年数 (影响混淆噪声)
        include_confusion: 是否含银河混淆噪声
    """

    # Robson+2019 Table 1: 银河混淆噪声拟合参数
    _CONFUSION_PARAMS = {
        0.5: {'alpha': 0.133, 'beta': 243., 'kappa': 482., 'gamma': 917., 'f_k': 2.58e-3},
        1.0: {'alpha': 0.171, 'beta': 292., 'kappa': 1020., 'gamma': 1680., 'f_k': 2.15e-3},
        2.0: {'alpha': 0.165, 'beta': 299., 'kappa': 611., 'gamma': 1340., 'f_k': 1.73e-3},
        4.0: {'alpha': 0.138, 'beta': -221., 'kappa': 521., 'gamma': 1680., 'f_k': 1.13e-3},
    }

    # 噪声预设: 直接对应 LISA Definition Study Report Figure 7.1 中的曲线
    # 通过调整底层 OMS / acc 噪声参数, 而不是事后乘缩放因子.
    #
    # 'scird':  LISA-LCST-SGS-TN-001 (Babak+2021) 官方 SciRD 公式参数
    #           OMS = 15 pm/√Hz, acc = 3 fm/s²/√Hz
    #           ≈ Red Book Figure 7.1 中的 Allocation (绿带)
    #           = 任务需求 / requirement curve
    #
    # 'cbe':    Current Best Estimate
    #           OMS ≈ 7.9 pm/√Hz, acc ≈ 2.4 fm/s²/√Hz
    #           ≈ Red Book Figure 7.1 中的 CBE (蓝线)
    #           = 实际预期性能, 显著优于 requirement
    #           参数取自 LDC AnalyticNoise (Sangria, LDC2a)
    #
    # 'robson2019': 别名, 与 'scird' 一致 (向后兼容)
    _NOISE_PRESETS = {
        'scird':      {'oms': 15.0e-12, 'acc': 3.0e-15},
        'cbe':        {'oms':  7.9e-12, 'acc': 2.4e-15},
        'robson2019': {'oms': 15.0e-12, 'acc': 3.0e-15},
    }

    # 旧 calibration 名 → 新 noise_preset 名 (向后兼容)
    _LEGACY_CAL_MAP = {
        'robson2019':             'scird',
        'redbook2023_allocation': 'scird',
        'redbook2023_cbe':        'cbe',
    }

    def __init__(self, target_log10_freq, noise_source='analytical',
                 noise_scale=1.0, injection_mode='physical',
                 logscale_base=1.0, T_obs=4.0, include_confusion=True,
                 noise_preset='scird', calibration=None):
        self.noise_scale = noise_scale
        self.injection_mode = injection_mode
        self.logscale_base = logscale_base
        self.target_log10_freq = np.asarray(target_log10_freq, dtype=np.float64)

        # 向后兼容: 如果用户传了 calibration, 自动映射到 noise_preset
        if calibration is not None:
            mapped = self._LEGACY_CAL_MAP.get(calibration)
            if mapped is None:
                raise ValueError(
                    f"Unknown legacy calibration '{calibration}'. "
                    f"Valid: {list(self._LEGACY_CAL_MAP.keys())}, "
                    f"or use new 'noise_preset': {list(self._NOISE_PRESETS.keys())}")
            print(f"[LISA Noise] [DEPRECATED] calibration='{calibration}' "
                  f"→ noise_preset='{mapped}'")
            noise_preset = mapped

        if noise_preset not in self._NOISE_PRESETS:
            raise ValueError(
                f"Unknown LISA noise_preset '{noise_preset}'. "
                f"Valid: {list(self._NOISE_PRESETS.keys())}")
        self.noise_preset = noise_preset
        preset = self._NOISE_PRESETS[noise_preset]
        self._oms_noise = preset['oms']
        self._acc_noise = preset['acc']

        target_hz = 10.0 ** self.target_log10_freq

        if noise_source == 'analytical':
            sn = self._analytical_psd(target_hz, T_obs, include_confusion,
                                       oms_noise=self._oms_noise,
                                       acc_noise=self._acc_noise)
        else:
            sn = self._load_datafile(noise_source, target_hz)

        # 转换为 σ_Ω(f)
        H0_si = 67.4e3 / 3.0857e22
        self.sigma_omega = ((2 * np.pi ** 2 / (3 * H0_si ** 2))
                            * target_hz ** 3 * sn).astype(np.float32)
        self.sigma_omega *= self.noise_scale

        # logscale 权重 (log 压缩)
        log_sigma = np.log10(np.maximum(self.sigma_omega, 1e-30))
        log_min, log_max = log_sigma.min(), log_sigma.max()
        if log_max > log_min:
            max_ratio = 10.0
            self.freq_weight = (1.0 + (max_ratio - 1.0)
                                * (log_sigma - log_min) / (log_max - log_min))
        else:
            self.freq_weight = np.ones_like(self.sigma_omega)
        self.freq_weight = self.freq_weight.astype(np.float32)

        print(f"[LISA Noise] source={noise_source}, preset={noise_preset} "
              f"(OMS={self._oms_noise*1e12:.2f} pm/√Hz, "
              f"acc={self._acc_noise*1e15:.2f} fm/s²/√Hz), "
              f"scale={noise_scale}, mode={injection_mode}, "
              f"T_obs={T_obs}yr, confusion={include_confusion}")
        print(f"  σ_Ω range: [{self.sigma_omega.min():.3e}, "
              f"{self.sigma_omega.max():.3e}]")

    @staticmethod
    def _analytical_psd(f, T_obs=4.0, include_confusion=True,
                        oms_noise=15.0e-12, acc_noise=3.0e-15):
        """Robson+ 2019 Eq.(10)+(12)+(14) — LISA 有效应变 PSD [1/Hz]

        噪声参数 (oms_noise, acc_noise) 决定 SciRD 还是 CBE:
            SciRD/Allocation: oms=15 pm/√Hz,  acc=3 fm/s²/√Hz
            CBE:              oms=7.9 pm/√Hz, acc=2.4 fm/s²/√Hz
        """
        L = 2.5e9           # 臂长 [m]
        f_star = 19.09e-3   # c/(2πL) [Hz]

        P_oms = oms_noise ** 2 * (1 + (2e-3 / f) ** 4)
        P_acc = acc_noise ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)

        Sn = (10.0 / (3 * L ** 2)) * (
            P_oms + 2 * (1 + np.cos(f / f_star) ** 2) * P_acc / (2 * np.pi * f) ** 4
        ) * (1 + 0.6 * (f / f_star) ** 2)

        if include_confusion and T_obs > 0:
            Sc = LISANoiseModel._confusion_noise(f, T_obs)
            Sn = Sn + Sc

        return Sn

    @staticmethod
    def _confusion_noise(f, T_obs):
        """Robson+ 2019 Eq.(14) — 银河前景混淆噪声"""
        # 选取最近的参数集
        years = sorted(LISANoiseModel._CONFUSION_PARAMS.keys())
        yr = min(years, key=lambda y: abs(y - T_obs))
        p = LISANoiseModel._CONFUSION_PARAMS[yr]

        A = 9e-45
        Sc = (A * f ** (-7.0 / 3.0)
              * np.exp(-f ** p['alpha'] + p['beta'] * f * np.sin(p['kappa'] * f))
              * (1 + np.tanh(p['gamma'] * (p['f_k'] - f))))
        return Sc

    @staticmethod
    def _load_datafile(filepath, target_hz):
        """从外部文件加载 ASD 并转为 PSD, 插值到目标网格"""
        data = np.loadtxt(filepath)
        freq_hz, asd = data[:, 0], data[:, 1]
        psd = asd ** 2
        fn = interp1d(np.log10(freq_hz), np.log10(psd),
                      kind='linear', fill_value='extrapolate')
        return 10.0 ** fn(np.log10(target_hz))

    def inject(self, log10_omega, log10_freq=None):
        """注入噪声, 接口与 LIGONoiseModel 一致。"""
        if self.injection_mode == 'physical':
            omega = np.float64(10.0) ** log10_omega.astype(np.float64)
            noise = np.random.normal(0.0, self.sigma_omega.astype(np.float64))
            omega_obs = np.abs(omega + noise)
            omega_obs = np.maximum(omega_obs, 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        else:
            noise = np.random.normal(0.0, 1.0, log10_omega.shape).astype(np.float32)
            return log10_omega + noise * self.freq_weight * self.logscale_base

# ==========================================
# 0. 频段配置
# ==========================================

# Concat 数据集配置 (非均匀拼接)
BAND_CONFIG_CONCAT = {
    'PTA':  {'start': 0,   'len': 14,  'freq_range': (-8.70, -7.56)},
    'LISA': {'start': 14,  'len': 500, 'freq_range': (-3.00, -1.00)},
    'LIGO': {'start': 270, 'len': 546, 'freq_range': (1.30, 3.24)},
}

# Conti 数据集配置 (连续插值)
BAND_FREQ_RANGES = {
    'PTA':  (-8.70, -7.56),
    'LISA': (-3.00, -1.00),
    'LIGO': (1.30, 3.24),
}
FREQ_INTERVAL = 0.01465073

def calc_band_points(freq_range, interval):
    return int(round((freq_range[1] - freq_range[0]) / interval)) + 1

BAND_CONFIG_CONTI = {
    'PTA':  {'len': calc_band_points(BAND_FREQ_RANGES['PTA'], FREQ_INTERVAL),  'freq_range': BAND_FREQ_RANGES['PTA']},
    'LISA': {'len': calc_band_points(BAND_FREQ_RANGES['LISA'], FREQ_INTERVAL), 'freq_range': BAND_FREQ_RANGES['LISA']},
    'LIGO': {'len': calc_band_points(BAND_FREQ_RANGES['LIGO'], FREQ_INTERVAL), 'freq_range': BAND_FREQ_RANGES['LIGO']},
}
BAND_CONFIG_CONTI['PTA']['start'] = 0
BAND_CONFIG_CONTI['LISA']['start'] = int(round((BAND_FREQ_RANGES['LISA'][0] - BAND_FREQ_RANGES['PTA'][0]) / FREQ_INTERVAL))
BAND_CONFIG_CONTI['LIGO']['start'] = int(round((BAND_FREQ_RANGES['LIGO'][0] - BAND_FREQ_RANGES['PTA'][0]) / FREQ_INTERVAL))

# 单频段数据集路径
SINGLE_BAND_PATHS = {
    'PTA':  'PTA_fixed_14pts.json',
    'LISA': 'LISA_fixed_500pts.json',
    'LIGO': 'LIGO_fixed_546pts.json',
}

# 9 个参数配置
PARAM_CONFIG = {
    'names': ['r', 'n_t', 'kappa10', 'T_re', 'DN_re', 'Omega_bh2', 'Omega_ch2', 'H0', 'A_s'],
    'log10_idx': [0, 2, 3],
    'log_idx': [8],
    'raw_idx': [1, 4, 5, 6, 7],
}


# ==========================================
# 1. 数据集类
# ==========================================

class BaseDataset(Dataset):
    """基类数据集，包含公共方法"""

    def __init__(self, noise_config=None, use_slope=True, use_curvature=True):
        self.noise_config = noise_config if noise_config else {}
        self.use_complex = self.noise_config.get('USE_COMPLEX_NOISE', False)
        self.noise_level = self.noise_config.get('noise_level', 0.0)
        self.glitch_prob = self.noise_config.get('glitch_prob', 0.0)
        self.use_slope = use_slope
        self.use_curvature = use_curvature

    @staticmethod
    def _build_ligo_noise_model(noise_config, target_log10_freq):
        """如果配置了 LIGO 物理噪声, 构建 LIGONoiseModel。"""
        ligo_cfg = noise_config.get('ligo', {}) if noise_config else {}
        noise_file = ligo_cfg.get('noise_file', None)
        if noise_file is None or not os.path.isfile(noise_file):
            return None
        return LIGONoiseModel(
            noise_file=noise_file,
            noise_type=ligo_cfg.get('noise_type', 'cc_spectrum'),
            target_log10_freq=target_log10_freq,
            noise_scale=ligo_cfg.get('noise_scale', 1.0),
            injection_mode=ligo_cfg.get('injection_mode', 'physical'),
            logscale_base=ligo_cfg.get('logscale_base', 1.0),
        )

    @staticmethod
    def _build_pta_noise_model(noise_config):
        """如果配置了 PTA 物理噪声, 构建 PTANoiseModel。"""
        pta_cfg = noise_config.get('pta', {}) if noise_config else {}
        noise_dir = pta_cfg.get('noise_dir', None)
        if noise_dir is None or not os.path.isdir(noise_dir):
            return None
        return PTANoiseModel(
            noise_dir=noise_dir,
            mode=pta_cfg.get('mode', 'density_mock'),
            noise_scale=pta_cfg.get('noise_scale', 1.0),
            n_bins_use=pta_cfg.get('n_bins_use', 14),
            per_sample_noise=pta_cfg.get('per_sample_noise', True),
            H0_fiducial=pta_cfg.get('H0_fiducial', 67.4),
            T_obs=pta_cfg.get('T_obs', None),
        )

    @staticmethod
    def _build_lisa_noise_model(noise_config, target_log10_freq):
        """如果配置了 LISA 物理噪声, 构建 LISANoiseModel。"""
        lisa_cfg = noise_config.get('lisa', {}) if noise_config else {}
        if not lisa_cfg.get('enabled', False):
            return None
        # 优先使用新参数 noise_preset, 回退到 legacy calibration
        noise_preset = lisa_cfg.get('noise_preset', None)
        legacy_cal = lisa_cfg.get('calibration', None)
        if noise_preset is None and legacy_cal is None:
            noise_preset = 'scird'  # 默认
        return LISANoiseModel(
            target_log10_freq=target_log10_freq,
            noise_source=lisa_cfg.get('noise_source', 'analytical'),
            noise_scale=lisa_cfg.get('noise_scale', 1.0),
            injection_mode=lisa_cfg.get('injection_mode', 'physical'),
            logscale_base=lisa_cfg.get('logscale_base', 1.0),
            T_obs=lisa_cfg.get('T_obs', 4.0),
            include_confusion=lisa_cfg.get('include_confusion', True),
            noise_preset=noise_preset if noise_preset is not None else 'scird',
            calibration=legacy_cal,
        )

    @staticmethod
    def _standardize_single(features, scalers):
        """用预拟合的 scaler 标准化单条样本的多通道特征。

        Args:
            features: shape (C, L) - 原始特征
            scalers:  list of StandardScaler, 每个通道一个
        Returns:
            standardized features, shape (C, L)
        """
        result = np.empty_like(features)
        for c in range(features.shape[0]):
            # scaler 是在 reshape(-1, 1) 上 fit 的, mean_ 和 scale_ 都是标量数组
            result[c] = (features[c] - scalers[c].mean_[0]) / scalers[c].scale_[0]
        return result

    def compute_features(self, omega_vals, freq_vals):
        """计算特征通道"""
        features = [omega_vals, freq_vals]
        if self.use_slope:
            slope = np.gradient(omega_vals, freq_vals)
            features.append(slope.astype(np.float32))
        if self.use_curvature:
            if self.use_slope:
                curv = np.gradient(features[-1], freq_vals)
            else:
                curv = np.gradient(np.gradient(omega_vals, freq_vals), freq_vals)
            features.append(curv.astype(np.float32))
        return np.stack(features, axis=0)

    def extract_params(self, item):
        """提取并变换 9 个参数"""
        return [
            np.log10(item['r']),
            item['n_t'],
            np.log10(item['kappa10']),
            np.log10(item['T_re']),
            item['DN_re'],
            item['Omega_bh2'],
            item['Omega_ch2'],
            item['H0'],
            np.log(1e10 * item['A_s'])
        ]

    def add_spectral_glitch(self, log_curve_normed):
        """在频域添加 Glitch 干扰"""
        seq_len = log_curve_normed.shape[0]
        linear_energy = np.exp(log_curve_normed)
        num_glitches = np.random.randint(1, 3)
        glitch_energy = np.zeros_like(linear_energy)

        for _ in range(num_glitches):
            center_idx = np.random.randint(0, seq_len)
            sigma = np.random.uniform(2.0, 15.0)
            amplitude = np.random.uniform(0.5, 3.0) * np.max(linear_energy)
            x = np.arange(seq_len)
            bump = amplitude * np.exp(-0.5 * ((x - center_idx) / sigma)**2)
            glitch_energy += bump

        total_energy = linear_energy + glitch_energy
        return np.log(total_energy + 1e-10)

    def apply_noise(self, curves):
        """应用噪声到曲线的 omega 通道"""
        if self.use_complex:
            if self.noise_level > 0:
                for curve in curves:
                    curve[0, :] += np.random.normal(0, self.noise_level, curve[0].shape)
            if self.glitch_prob > 0 and np.random.rand() < self.glitch_prob:
                for curve in curves:
                    curve[0, :] = self.add_spectral_glitch(curve[0, :])
        else:
            if self.noise_level > 0:
                for curve in curves:
                    curve[0, :] += np.random.normal(0, self.noise_level, curve[0].shape)
        return curves


class MultiBandDataset(BaseDataset):
    """三频段联合数据集 (concat 或 conti)"""

    def __init__(self, data, band_config, noise_config=None, use_slope=True, use_curvature=True):
        super().__init__(noise_config, use_slope, use_curvature)
        self.band_config = band_config

        pta_curves, lisa_curves, ligo_curves = [], [], []

        for item in data:
            omega = np.array(item['log10OmegaGW'], dtype=np.float32)
            freq = np.array(item['f'], dtype=np.float32)

            pta_omega = omega[band_config['PTA']['start']:band_config['PTA']['start']+band_config['PTA']['len']]
            pta_freq = freq[band_config['PTA']['start']:band_config['PTA']['start']+band_config['PTA']['len']]

            lisa_omega = omega[band_config['LISA']['start']:band_config['LISA']['start']+band_config['LISA']['len']]
            lisa_freq = freq[band_config['LISA']['start']:band_config['LISA']['start']+band_config['LISA']['len']]

            ligo_omega = omega[band_config['LIGO']['start']:band_config['LIGO']['start']+band_config['LIGO']['len']]
            ligo_freq = freq[band_config['LIGO']['start']:band_config['LIGO']['start']+band_config['LIGO']['len']]

            pta_curves.append(self.compute_features(pta_omega, pta_freq))
            lisa_curves.append(self.compute_features(lisa_omega, lisa_freq))
            ligo_curves.append(self.compute_features(ligo_omega, ligo_freq))

        self.pta_curves_raw = np.stack(pta_curves, axis=0)
        self.lisa_curves_raw = np.stack(lisa_curves, axis=0)
        self.ligo_curves_raw = np.stack(ligo_curves, axis=0)

        self.num_channels = self.pta_curves_raw.shape[1]
        print(f"Using {self.num_channels} channels: omega, freq" +
              (", slope" if use_slope else "") +
              (", curvature" if use_curvature else ""))

        # 参数提取
        params = [self.extract_params(item) for item in data]
        self.params_raw = np.array(params, dtype=np.float32)

        # ---- 三频段物理噪声模型 ----
        self.pta_noise_model = self._build_pta_noise_model(noise_config)
        if self.pta_noise_model is not None and hasattr(self.pta_noise_model, 'validate_freqs'):
            self.pta_noise_model.validate_freqs(self.pta_curves_raw[0, 1, :])

        self.lisa_freq_grid = self.lisa_curves_raw[0, 1, :]
        self.lisa_noise_model = self._build_lisa_noise_model(
            noise_config, self.lisa_freq_grid)

        self.ligo_freq_grid = self.ligo_curves_raw[0, 1, :]
        self.ligo_noise_model = self._build_ligo_noise_model(
            noise_config, self.ligo_freq_grid)

        # ---- 标准化 ----
        self.pta_scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.lisa_scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.ligo_scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.param_scaler = StandardScaler()

        # PTA scaler
        if self.pta_noise_model is not None:
            print("[PTA Noise] Fitting scalers on noise-injected reference data...")
            pta_ref = self._generate_noisy_reference('pta')
            self.pta_curves = self._scale_curves(pta_ref, self.pta_scalers, band_config['PTA']['len'])
        else:
            self.pta_curves = self._scale_curves(self.pta_curves_raw, self.pta_scalers, band_config['PTA']['len'])

        # LISA scaler
        if self.lisa_noise_model is not None and \
                self.lisa_noise_model.injection_mode == 'physical':
            print("[LISA Noise] Fitting scalers on noise-injected reference data...")
            lisa_ref = self._generate_noisy_reference('lisa')
            self.lisa_curves = self._scale_curves(lisa_ref, self.lisa_scalers, band_config['LISA']['len'])
        else:
            self.lisa_curves = self._scale_curves(self.lisa_curves_raw, self.lisa_scalers, band_config['LISA']['len'])

        # LIGO scaler
        if self.ligo_noise_model is not None and \
                self.ligo_noise_model.injection_mode == 'physical':
            print("[LIGO Noise] Fitting scalers on noise-injected reference data...")
            ligo_ref = self._generate_noisy_reference('ligo')
            self.ligo_curves = self._scale_curves(ligo_ref, self.ligo_scalers, band_config['LIGO']['len'])
        else:
            self.ligo_curves = self._scale_curves(self.ligo_curves_raw, self.ligo_scalers, band_config['LIGO']['len'])

        self.params = self.param_scaler.fit_transform(self.params_raw)

        print(f"Dataset loaded: {len(self.params)} samples")
        print(f"  PTA shape: {self.pta_curves.shape}")
        print(f"  LISA shape: {self.lisa_curves.shape}")
        print(f"  LIGO shape: {self.ligo_curves.shape}")

    def _generate_noisy_reference(self, band):
        """为 physical 模式生成含噪特征, 用于拟合 scaler。"""
        if band == 'pta':
            raw, model = self.pta_curves_raw, self.pta_noise_model
        elif band == 'lisa':
            raw, model = self.lisa_curves_raw, self.lisa_noise_model
        else:
            raw, model = self.ligo_curves_raw, self.ligo_noise_model

        noisy_curves = []
        for i in range(len(raw)):
            omega_raw = raw[i, 0, :].copy()
            freq_raw = raw[i, 1, :]
            omega_noisy = model.inject(omega_raw, freq_raw)
            features = self.compute_features(omega_noisy, freq_raw)
            noisy_curves.append(features)
        return np.stack(noisy_curves, axis=0)

    # ------------------------------------------------------------------
    # 向量化批量噪声缓存 (优化 B)
    # ------------------------------------------------------------------
    def _compute_features_batch(self, omega_batch, freq):
        """向量化计算整个 batch 的特征通道。"""
        freq_batch = np.broadcast_to(freq[None, :], omega_batch.shape).copy()
        features = [omega_batch, freq_batch]
        if self.use_slope:
            slope = np.gradient(omega_batch, freq, axis=1).astype(np.float32)
            features.append(slope)
        if self.use_curvature:
            if self.use_slope:
                curv = np.gradient(features[-1], freq, axis=1).astype(np.float32)
            else:
                curv = np.gradient(
                    np.gradient(omega_batch, freq, axis=1),
                    freq, axis=1
                ).astype(np.float32)
            features.append(curv)
        return np.stack(features, axis=1)

    def _standardize_batch(self, features_batch, scalers):
        """向量化标准化"""
        result = np.empty_like(features_batch)
        for c in range(features_batch.shape[1]):
            result[:, c, :] = ((features_batch[:, c, :]
                                - scalers[c].mean_[0]) / scalers[c].scale_[0])
        return result

    def _inject_noise_batch(self, omega_batch, noise_model, band_name, noise_scale_override=1.0):
        """向量化噪声注入"""
        if noise_model is None:
            return None

        if band_name == 'PTA':
            if noise_model.is_density_mode:
                # 物理路径: 线性 Ω 空间相加 (Li & Shapiro 2025 Eq.3, 减均值后的偏差)
                omega_lin = (np.float64(10.0) ** omega_batch.astype(np.float64))
                omega_noise = noise_model.sample_omega_noise(omega_batch.shape[0])  # (N,14)
                omega_obs = np.maximum(omega_lin + omega_noise * noise_scale_override,
                                       1e-300)
                return np.log10(omega_obs).astype(np.float32)
            # ---- legacy log-space additive ----
            if noise_model.mode == 'white':
                noise = np.random.normal(0.0, noise_model.sigma_white, omega_batch.shape)
            else:
                noise = np.random.normal(0.0,
                    noise_model.sigma_per_bin[:omega_batch.shape[1]], omega_batch.shape)
            return (omega_batch + noise * noise_scale_override).astype(np.float32)

        if getattr(noise_model, 'injection_mode', 'physical') == 'physical':
            omega_lin = np.float64(10.0) ** omega_batch.astype(np.float64)
            sigma = noise_model.sigma_omega.astype(np.float64)
            noise = np.random.normal(0.0, 1.0, omega_batch.shape) * sigma[None, :] * noise_scale_override
            omega_obs = np.maximum(np.abs(omega_lin + noise), 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        else:
            noise = np.random.normal(0.0, 1.0, omega_batch.shape).astype(np.float32)
            return omega_batch + noise * noise_model.freq_weight[None, :] * noise_model.logscale_base * noise_scale_override

    def refresh_noisy_cache(self, noise_scale_override=1.0):
        """每个 epoch 开始时调用: 向量化重新生成三个频段的含噪数据。"""
        self._cache = {}

        for band_name, raw, model, scalers, freq_grid in [
            ('PTA',  self.pta_curves_raw,  self.pta_noise_model,  self.pta_scalers,  None),
            ('LISA', self.lisa_curves_raw,  self.lisa_noise_model, self.lisa_scalers,  self.lisa_freq_grid),
            ('LIGO', self.ligo_curves_raw,  self.ligo_noise_model, self.ligo_scalers,  self.ligo_freq_grid),
        ]:
            if model is None:
                self._cache[band_name] = None
                continue
            omega_raw = raw[:, 0, :].copy()
            freq = raw[0, 1, :]
            omega_noisy = self._inject_noise_batch(omega_raw, model, band_name, noise_scale_override)
            features = self._compute_features_batch(omega_noisy, freq)
            self._cache[band_name] = self._standardize_batch(features, scalers)

    def _scale_curves(self, curves_raw, scalers, band_len):
        scaled = []
        for c in range(self.num_channels):
            channel_data = curves_raw[:, c, :].reshape(-1, 1)
            scalers[c].fit(channel_data)
            scaled.append(scalers[c].transform(channel_data).reshape(-1, band_len))
        return np.stack(scaled, axis=1)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx].copy()

        # ---- PTA ----
        if self.pta_noise_model is not None:
            if hasattr(self, '_cache') and self._cache.get('PTA') is not None:
                pta = self._cache['PTA'][idx]
            else:
                omega_raw = self.pta_curves_raw[idx, 0, :].copy()
                freq_raw = self.pta_curves_raw[idx, 1, :]
                omega_noisy = self.pta_noise_model.inject(omega_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                pta = self._standardize_single(features, self.pta_scalers)
        else:
            pta = self.pta_curves[idx].copy()

        # ---- LISA ----
        if self.lisa_noise_model is not None:
            if hasattr(self, '_cache') and self._cache.get('LISA') is not None:
                lisa = self._cache['LISA'][idx]
            else:
                omega_raw = self.lisa_curves_raw[idx, 0, :].copy()
                freq_raw = self.lisa_curves_raw[idx, 1, :]
                omega_noisy = self.lisa_noise_model.inject(omega_raw, freq_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                lisa = self._standardize_single(features, self.lisa_scalers)
        else:
            lisa = self.lisa_curves[idx].copy()

        # ---- LIGO ----
        if self.ligo_noise_model is not None:
            if hasattr(self, '_cache') and self._cache.get('LIGO') is not None:
                ligo = self._cache['LIGO'][idx]
            else:
                omega_raw = self.ligo_curves_raw[idx, 0, :].copy()
                freq_raw = self.ligo_curves_raw[idx, 1, :]
                omega_noisy = self.ligo_noise_model.inject(omega_raw, freq_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                ligo = self._standardize_single(features, self.ligo_scalers)
        else:
            ligo = self.ligo_curves[idx].copy()

        # 对没有物理噪声模型的频段, 沿用旧式高斯噪声
        if self.noise_level > 0:
            if self.pta_noise_model is None:
                pta[0, :] += np.random.normal(0, self.noise_level, pta[0].shape)
            if self.lisa_noise_model is None:
                lisa[0, :] += np.random.normal(0, self.noise_level, lisa[0].shape)
            if self.ligo_noise_model is None:
                ligo[0, :] += np.random.normal(0, self.noise_level, ligo[0].shape)

        return (
            torch.tensor(pta, dtype=torch.float32),
            torch.tensor(lisa, dtype=torch.float32),
            torch.tensor(ligo, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32)
        )


class SingleBandDataset(BaseDataset):
    """单频段数据集"""

    def __init__(self, data, band_name, band_config, noise_config=None, use_slope=True, use_curvature=True):
        super().__init__(noise_config, use_slope, use_curvature)
        self.band_name = band_name
        self.band_len = band_config['len']

        curves = []
        for item in data:
            omega = np.array(item['log10OmegaGW'], dtype=np.float32)
            freq = np.array(item['f'], dtype=np.float32)
            curves.append(self.compute_features(omega, freq))

        self.curves_raw = np.stack(curves, axis=0)
        self.num_channels = self.curves_raw.shape[1]

        print(f"[{band_name}] Using {self.num_channels} channels: omega, freq" +
              (", slope" if use_slope else "") +
              (", curvature" if use_curvature else ""))

        params = [self.extract_params(item) for item in data]
        self.params_raw = np.array(params, dtype=np.float32)

        # ---- 构建该频段的物理噪声模型 ----
        self.noise_model = None
        freq_grid = self.curves_raw[0, 1, :]  # log10(f/Hz)
        if band_name == 'PTA':
            self.noise_model = self._build_pta_noise_model(noise_config)
            if self.noise_model is not None and hasattr(self.noise_model, 'validate_freqs'):
                self.noise_model.validate_freqs(freq_grid)
        elif band_name == 'LISA':
            self.noise_model = self._build_lisa_noise_model(noise_config, freq_grid)
        elif band_name == 'LIGO':
            self.noise_model = self._build_ligo_noise_model(noise_config, freq_grid)

        # ---- 标准化 ----
        self.scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.param_scaler = StandardScaler()

        needs_noisy_scaler = (
            self.noise_model is not None and
            (band_name == 'PTA' or getattr(self.noise_model, 'injection_mode', '') == 'physical')
        )
        if needs_noisy_scaler:
            print(f"[{band_name} Noise] Fitting scalers on noise-injected reference...")
            noisy_ref = self._generate_noisy_reference()
            self.curves = self._scale_curves_single(noisy_ref)
        else:
            self.curves = self._scale_curves_single(self.curves_raw)

        self.params = self.param_scaler.fit_transform(self.params_raw)

        print(f"[{band_name}] Dataset loaded: {len(self.params)} samples, "
              f"shape: {self.curves.shape}")

    def _scale_curves_single(self, curves_data):
        scaled = []
        for c in range(self.num_channels):
            channel_data = curves_data[:, c, :].reshape(-1, 1)
            self.scalers[c].fit(channel_data)
            scaled.append(self.scalers[c].transform(channel_data).reshape(-1, self.band_len))
        return np.stack(scaled, axis=1)

    def _generate_noisy_reference(self):
        noisy = []
        for i in range(len(self.curves_raw)):
            omega_raw = self.curves_raw[i, 0, :].copy()
            freq_raw = self.curves_raw[i, 1, :]
            if self.band_name == 'PTA':
                omega_noisy = self.noise_model.inject(omega_raw)
            else:
                omega_noisy = self.noise_model.inject(omega_raw, freq_raw)
            features = self.compute_features(omega_noisy, freq_raw)
            noisy.append(features)
        return np.stack(noisy, axis=0)

    # ------------------------------------------------------------------
    # 向量化批量噪声缓存 (优化 B)
    # ------------------------------------------------------------------
    def _compute_features_batch(self, omega_batch, freq):
        """向量化计算整个 batch 的特征通道。

        Args:
            omega_batch: shape (N, L)
            freq:        shape (L,)  所有样本共享
        Returns:
            shape (N, C, L)
        """
        N = omega_batch.shape[0]
        freq_batch = np.broadcast_to(freq[None, :], omega_batch.shape).copy()
        features = [omega_batch, freq_batch]
        if self.use_slope:
            slope = np.gradient(omega_batch, freq, axis=1).astype(np.float32)
            features.append(slope)
        if self.use_curvature:
            if self.use_slope:
                curv = np.gradient(features[-1], freq, axis=1).astype(np.float32)
            else:
                curv = np.gradient(
                    np.gradient(omega_batch, freq, axis=1),
                    freq, axis=1
                ).astype(np.float32)
            features.append(curv)
        return np.stack(features, axis=1)

    def _standardize_batch(self, features_batch, scalers):
        """向量化标准化, features_batch shape (N, C, L)"""
        result = np.empty_like(features_batch)
        for c in range(features_batch.shape[1]):
            result[:, c, :] = ((features_batch[:, c, :]
                                - scalers[c].mean_[0]) / scalers[c].scale_[0])
        return result

    def _inject_noise_batch(self, omega_batch, noise_scale_override=1.0):
        """向量化噪声注入, omega_batch shape (N, L) 为 log10(Ω)

        Args:
            noise_scale_override: 噪声缩放因子, 用于 curriculum learning.
                0.0 = 无噪声, 1.0 = 全噪声.
        """
        model = self.noise_model
        if model is None:
            return omega_batch.copy()

        if self.band_name == 'PTA':
            if model.is_density_mode:
                omega_lin = (np.float64(10.0) ** omega_batch.astype(np.float64))
                omega_noise = model.sample_omega_noise(omega_batch.shape[0])  # (N,14)
                omega_obs = np.maximum(omega_lin + omega_noise * noise_scale_override,
                                       1e-300)
                return np.log10(omega_obs).astype(np.float32)
            # ---- legacy log-space additive ----
            if model.mode == 'white':
                noise = np.random.normal(0.0, model.sigma_white, omega_batch.shape)
            else:
                noise = np.random.normal(0.0, model.sigma_per_bin[:omega_batch.shape[1]],
                                         omega_batch.shape)
            return (omega_batch + noise * noise_scale_override).astype(np.float32)

        if getattr(model, 'injection_mode', 'physical') == 'physical':
            # Physical: 线性空间加噪 → log
            omega_lin = (np.float64(10.0)
                         ** omega_batch.astype(np.float64))
            sigma = model.sigma_omega.astype(np.float64)
            noise = np.random.normal(0.0, 1.0, omega_batch.shape) * sigma[None, :] * noise_scale_override
            omega_obs = np.maximum(np.abs(omega_lin + noise), 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        else:
            # Logscale
            noise = np.random.normal(0.0, 1.0, omega_batch.shape).astype(np.float32)
            return omega_batch + noise * model.freq_weight[None, :] * model.logscale_base * noise_scale_override

    def refresh_noisy_cache(self, noise_scale_override=1.0):
        """每个 epoch 开始时调用: 向量化重新生成整批含噪数据。

        Args:
            noise_scale_override: curriculum 噪声缩放, 0→1 渐进.
        """
        if self.noise_model is None:
            return  # 无物理噪声模型, 不需要缓存

        omega_raw = self.curves_raw[:, 0, :].copy()       # (N, L)
        freq = self.curves_raw[0, 1, :]                    # (L,)

        # 1) 向量化噪声注入
        omega_noisy = self._inject_noise_batch(omega_raw, noise_scale_override)  # (N, L)

        # 2) 向量化特征计算
        features = self._compute_features_batch(omega_noisy, freq)  # (N, C, L)

        # 3) 向量化标准化
        self._noisy_cache = self._standardize_batch(features, self.scalers)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx].copy()

        if self.noise_model is not None:
            if hasattr(self, '_noisy_cache') and self._noisy_cache is not None:
                # 从缓存读取 (快速路径)
                curve = self._noisy_cache[idx]
            else:
                # 回退: 逐样本计算 (首次 epoch 或未调用 refresh)
                omega_raw = self.curves_raw[idx, 0, :].copy()
                freq_raw = self.curves_raw[idx, 1, :]
                if self.band_name == 'PTA':
                    omega_noisy = self.noise_model.inject(omega_raw)
                else:
                    omega_noisy = self.noise_model.inject(omega_raw, freq_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                curve = self._standardize_single(features, self.scalers)
        else:
            curve = self.curves[idx].copy()
            curve = self.apply_noise([curve])[0]

        return (
            torch.tensor(curve, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32)
        )


# ==========================================
# 2. 模型组件
# ==========================================

class ResBlock1D(nn.Module):
    """1D 残差块"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class BandEncoder(nn.Module):
    """单个频段的编码器"""
    def __init__(self, in_channels=4, hidden_dim=64, output_dim=64, band_type='lisa', seq_len=256):
        super().__init__()
        self.band_type = band_type

        if band_type == 'pta':
            if seq_len <= 20:
                # 点数很少，用 MLP
                self.encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_channels * seq_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Tanh()
                )
            else:
                # 点数中等，用轻量 CNN
                self.encoder = nn.Sequential(
                    nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    ResBlock1D(32),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, output_dim),
                    nn.Tanh()
                )
        elif band_type == 'lisa':
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                ResBlock1D(32),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32, output_dim),
                nn.Tanh()
            )
        else:  # ligo
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                ResBlock1D(64),
                ResBlock1D(64),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, output_dim),
                nn.Tanh()
            )

    def forward(self, x):
        return self.encoder(x)


class MultiBandContextEncoder(nn.Module):
    """三频段融合编码器"""
    def __init__(self, band_config, context_dim=64, in_channels=4):
        super().__init__()
        per_band_dim = context_dim // 3
        remainder = context_dim - per_band_dim * 3

        self.pta_encoder = BandEncoder(in_channels, hidden_dim=32, output_dim=per_band_dim,
                                       band_type='pta', seq_len=band_config['PTA']['len'])
        self.lisa_encoder = BandEncoder(in_channels, hidden_dim=64, output_dim=per_band_dim,
                                        band_type='lisa', seq_len=band_config['LISA']['len'])
        self.ligo_encoder = BandEncoder(in_channels, hidden_dim=64, output_dim=per_band_dim + remainder,
                                        band_type='ligo', seq_len=band_config['LIGO']['len'])

    def forward(self, pta, lisa, ligo):
        h_pta = self.pta_encoder(pta)
        h_lisa = self.lisa_encoder(lisa)
        h_ligo = self.ligo_encoder(ligo)
        return torch.cat([h_pta, h_lisa, h_ligo], dim=1)


class SingleBandEncoder(nn.Module):
    """单频段编码器"""
    def __init__(self, band_name, in_channels=4, hidden_dim=64, output_dim=64, seq_len=256):
        super().__init__()
        band_type = 'pta' if band_name == 'PTA' else ('lisa' if band_name == 'LISA' else 'ligo')
        self.encoder = BandEncoder(in_channels, hidden_dim, output_dim, band_type, seq_len)

    def forward(self, x):
        return self.encoder(x)


class ConditionalCouplingLayer(nn.Module):
    """条件耦合层"""
    def __init__(self, num_inputs, num_hidden, mask, context_dim):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        input_dim = num_inputs + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_inputs * 2)
        )

    def forward(self, inputs, context, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask
        net_input = torch.cat([masked_inputs, context], dim=1)
        out = self.net(net_input)
        s, t = out.chunk(2, dim=1)
        s = torch.tanh(s) * (1 - mask)
        t = t * (1 - mask)
        if mode == 'direct':
            y = inputs * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)
            return y, log_det
        else:
            x = (inputs - t) * torch.exp(-s)
            return x, None


class MultiBandFlow(nn.Module):
    """三频段联合条件 Normalizing Flow"""
    def __init__(self, band_config, num_inputs=9, num_hidden=128, num_layers=10,
                 context_dim=64, in_channels=4):
        super().__init__()
        self.num_inputs = num_inputs
        self.context_encoder = MultiBandContextEncoder(band_config, context_dim, in_channels)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mask = torch.zeros(num_inputs)
            if i % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
            self.layers.append(ConditionalCouplingLayer(num_inputs, num_hidden, mask, context_dim))

        # 辅助参数预测头: 从 context 直接回归参数, 强化编码器对关键参数的编码
        self.param_predictor = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, num_inputs)
        )

    def _get_context(self, pta, lisa, ligo):
        return self.context_encoder(pta, lisa, ligo)

    def log_prob(self, inputs, pta, lisa, ligo):
        context = self._get_context(pta, lisa, ligo)
        log_det_sum = 0
        z = inputs
        for layer in self.layers:
            z, log_det = layer(z, context, mode='direct')
            log_det_sum += log_det
        log_p_z = -0.5 * (np.log(2 * np.pi) + z.pow(2)).sum(dim=1)
        return log_p_z + log_det_sum

    def predict_params(self, pta, lisa, ligo):
        """辅助参数预测 (用于 T_re 优化等)"""
        context = self._get_context(pta, lisa, ligo)
        return self.param_predictor(context)

    def sample(self, pta, lisa, ligo, num_samples=1):
        context = self._get_context(pta, lisa, ligo)
        B = context.shape[0]

        context_rep = context.unsqueeze(1).repeat(1, num_samples, 1).reshape(B * num_samples, -1)
        z = torch.randn(B * num_samples, 9, device=pta.device)

        for layer in reversed(self.layers):
            z, _ = layer(z, context_rep, mode='inverse')

        return z.reshape(B, num_samples, -1)


class SingleBandFlow(nn.Module):
    """单频段条件 Normalizing Flow"""
    def __init__(self, band_name, seq_len, num_inputs=9, num_hidden=128, num_layers=10,
                 context_dim=64, in_channels=4):
        super().__init__()
        self.band_name = band_name
        self.num_inputs = num_inputs
        self.context_encoder = SingleBandEncoder(band_name, in_channels, 64, context_dim, seq_len)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mask = torch.zeros(num_inputs)
            if i % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
            self.layers.append(ConditionalCouplingLayer(num_inputs, num_hidden, mask, context_dim))

        # 辅助参数预测头
        self.param_predictor = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, num_inputs)
        )

    def _get_context(self, curve):
        return self.context_encoder(curve)

    def log_prob(self, inputs, curve):
        context = self._get_context(curve)
        log_det_sum = 0
        z = inputs
        for layer in self.layers:
            z, log_det = layer(z, context, mode='direct')
            log_det_sum += log_det
        log_p_z = -0.5 * (np.log(2 * np.pi) + z.pow(2)).sum(dim=1)
        return log_p_z + log_det_sum

    def predict_params(self, curve):
        """辅助参数预测 (用于 T_re 优化等)"""
        context = self._get_context(curve)
        return self.param_predictor(context)

    def sample(self, curve, num_samples=1):
        context = self._get_context(curve)
        B = context.shape[0]

        context_rep = context.unsqueeze(1).repeat(1, num_samples, 1).reshape(B * num_samples, -1)
        z = torch.randn(B * num_samples, 9, device=curve.device)

        for layer in reversed(self.layers):
            z, _ = layer(z, context_rep, mode='inverse')

        return z.reshape(B, num_samples, -1)


# ==========================================
# 3. 训练函数
# ==========================================

def _build_param_weights(config, device):
    """从配置中构建参数权重向量和辅助损失系数。

    配置示例 (在 JSON 的 "training" 或顶层):
        "param_emphasis": {
            "enabled": true,
            "aux_weight": 0.1,
            "weights": {
                "T_re": 3.0,
                "DN_re": 0.3
            }
        }

    Returns:
        aux_weight (float):  辅助损失的全局系数 λ
        param_weights (Tensor): shape (9,), 每个参数的 MSE 权重
        如果 disabled 返回 (0.0, None)
    """
    pe = config.get('param_emphasis', {})
    if not pe.get('enabled', False):
        return 0.0, None

    aux_weight = pe.get('aux_weight', 0.1)
    weight_map = pe.get('weights', {})

    names = PARAM_CONFIG['names']
    w = torch.ones(len(names), device=device)
    for name, val in weight_map.items():
        if name in names:
            w[names.index(name)] = val
        else:
            print(f"[WARNING] param_emphasis: unknown parameter '{name}', ignored")

    print(f"[Param Emphasis] aux_weight={aux_weight}, "
          f"weights={dict(zip(names, w.tolist()))}")
    return aux_weight, w


def _get_curriculum_scale(epoch, total_epochs, curriculum_cfg):
    """计算当前 epoch 的噪声缩放因子 (curriculum noise schedule)。

    配置示例:
        "noise_curriculum": {
            "enabled": true,
            "warmup_fraction": 0.2,
            "start_scale": 0.05,
            "end_scale": 1.0
        }

    在 warmup_fraction * total_epochs 内, 噪声从 start_scale 线性增长到 end_scale,
    之后保持 end_scale。

    Returns:
        float: 当前 epoch 的噪声缩放因子
    """
    if not curriculum_cfg.get('enabled', False):
        return 1.0

    warmup_frac = curriculum_cfg.get('warmup_fraction', 0.2)
    start_scale = curriculum_cfg.get('start_scale', 0.05)
    end_scale = curriculum_cfg.get('end_scale', 1.0)

    warmup_epochs = int(total_epochs * warmup_frac)
    if warmup_epochs <= 0:
        return end_scale

    if epoch >= warmup_epochs:
        return end_scale

    # 线性插值
    t = epoch / warmup_epochs
    return start_scale + (end_scale - start_scale) * t

def train_multiband(config):
    """训练三频段联合模型"""
    # 解析配置
    dataset_type = config['dataset']['type']
    json_path = config['dataset']['path']
    use_slope = config['features']['use_slope']
    use_curvature = config['features']['use_curvature']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    grad_clip = config['training']['grad_clip']
    scheduler_T_max = config['training']['scheduler_T_max']

    noise_config = {
        'USE_COMPLEX_NOISE': config['noise']['use_complex'],
        'noise_level': config['noise']['level'],
        'glitch_prob': config['noise']['glitch_prob'],
    }

    # ---- 物理噪声配置 ----
    for band_key in ('pta', 'lisa', 'ligo'):
        band_cfg = config['noise'].get(band_key, {})
        if band_cfg:
            noise_config[band_key] = band_cfg
            print(f"{band_key.upper()} physical noise enabled: {band_cfg}")

    save_dir = config['output']['save_dir']
    save_name = config['output']['save_name']
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    # 选择频段配置
    band_config = BAND_CONFIG_CONCAT if dataset_type == 'concat' else BAND_CONFIG_CONTI

    # 加载数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"{'='*60}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Loaded {len(data)} samples from {json_path}")
    print(f"Band config: PTA={band_config['PTA']['len']}, "
          f"LISA={band_config['LISA']['len']}, LIGO={band_config['LIGO']['len']}")

    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    train_dataset = MultiBandDataset(train_data, band_config, noise_config, use_slope, use_curvature)

    # 验证集: 保留 LIGO 物理噪声 (评估含噪性能), 但关闭旧式高斯噪声
    val_noise_config = {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0}
    for band_key in ('pta', 'lisa', 'ligo'):
        if band_key in noise_config:
            val_noise_config[band_key] = noise_config[band_key]
    val_dataset = MultiBandDataset(val_data, band_config, val_noise_config,
                                   use_slope, use_curvature)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    in_channels = train_dataset.num_channels
    model = MultiBandFlow(band_config,
                          num_inputs=config['model']['num_inputs'],
                          num_hidden=config['model']['num_hidden'],
                          num_layers=config['model']['num_layers'],
                          context_dim=config['model']['context_dim'],
                          in_channels=in_channels).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_T_max)

    print(f"Noise Mode: {'COMPLEX (Glitch)' if noise_config['USE_COMPLEX_NOISE'] else 'SIMPLE (Gaussian)'}")
    print(f"Features: slope={use_slope}, curvature={use_curvature}")

    # ---- 参数强调 (T_re 优化等) ----
    aux_weight, param_weights = _build_param_weights(config, device)

    # ---- Curriculum noise schedule ----
    curriculum_cfg = config.get('noise_curriculum', {})
    if curriculum_cfg.get('enabled', False):
        print(f"[Curriculum] warmup={curriculum_cfg.get('warmup_fraction', 0.2):.0%} of {epochs} epochs, "
              f"scale: {curriculum_cfg.get('start_scale', 0.05)} → {curriculum_cfg.get('end_scale', 1.0)}")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ---- 每个 epoch 刷新噪声缓存 (向量化, 毫秒级) ----
        noise_scale = _get_curriculum_scale(epoch, epochs, curriculum_cfg)
        train_dataset.refresh_noisy_cache(noise_scale)
        val_dataset.refresh_noisy_cache()  # 验证集始终用全噪声

        model.train()
        train_loss = 0.0

        for pta, lisa, ligo, params in train_loader:
            pta, lisa, ligo, params = pta.to(device), lisa.to(device), ligo.to(device), params.to(device)

            loss_flow = -torch.mean(model.log_prob(params, pta, lisa, ligo))

            # 辅助参数预测损失
            if aux_weight > 0 and param_weights is not None:
                pred = model.predict_params(pta, lisa, ligo)
                loss_aux = torch.mean(param_weights * (pred - params) ** 2)
                loss = loss_flow + aux_weight * loss_aux
            else:
                loss = loss_flow

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pta, lisa, ligo, params in val_loader:
                pta, lisa, ligo, params = pta.to(device), lisa.to(device), ligo.to(device), params.to(device)
                val_loss += -torch.mean(model.log_prob(params, pta, lisa, ligo)).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'pta_scalers': train_dataset.pta_scalers,
                'lisa_scalers': train_dataset.lisa_scalers,
                'ligo_scalers': train_dataset.ligo_scalers,
                'param_scaler': train_dataset.param_scaler,
                'band_config': band_config,
                'param_config': PARAM_CONFIG,
                'num_channels': in_channels,
                'use_slope': use_slope,
                'use_curvature': use_curvature,
                'dataset_type': dataset_type,
                'config': config,
            }, save_path)

        if epoch % 10 == 0 or epoch < 20:
            ns_str = f" | NS: {noise_scale:.3f}" if curriculum_cfg.get('enabled', False) else ""
            print(f"Epoch {epoch:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} (Best: {best_val_loss:.4f}){ns_str}")

    print(f"\nTraining complete! Best model saved to: {save_path}")
    return best_val_loss


def train_single_band(config):
    """训练单频段模型"""
    bands = config['dataset']['bands']
    data_dir = config['dataset']['data_dir']
    band_files = config['dataset']['band_files']

    use_slope = config['features']['use_slope']
    use_curvature = config['features']['use_curvature']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    grad_clip = config['training']['grad_clip']
    scheduler_T_max = config['training']['scheduler_T_max']

    noise_config = {
        'USE_COMPLEX_NOISE': config['noise']['use_complex'],
        'noise_level': config['noise']['level'],
        'glitch_prob': config['noise']['glitch_prob'],
    }

    # ---- 物理噪声配置 ----
    for band_key in ('pta', 'lisa', 'ligo'):
        band_cfg = config['noise'].get(band_key, {})
        if band_cfg:
            noise_config[band_key] = band_cfg

    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # ---- 参数强调 (T_re 优化等) — 在 band 循环外构建一次 ----
    # device 还不确定, 先在 CPU 上构建, 进 band 循环后 .to(device)
    _aux_weight, _param_weights_cpu = _build_param_weights(config, torch.device('cpu'))

    results = {}

    for band_name in bands:
        json_path = os.path.join(data_dir, band_files[band_name])
        band_config = {'len': BAND_CONFIG_CONCAT[band_name]['len'],
                       'freq_range': BAND_CONFIG_CONCAT[band_name]['freq_range']}
        # 单频段时尊重配置中的 save_name, 多频段时自动命名
        if len(bands) == 1 and config['output'].get('save_name'):
            save_path = os.path.join(save_dir, config['output']['save_name'])
        else:
            save_path = os.path.join(save_dir, f"{band_name.lower()}_flow.pt")

        # 加载数据
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"Experiment: {config['name']} - {band_name}")
        print(f"{'='*60}")
        print(f"Loaded {len(data)} samples from {json_path}")

        train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        # 训练集带噪声, 验证集保留物理噪声但关闭旧式高斯噪声
        train_dataset = SingleBandDataset(train_data, band_name, band_config, noise_config, use_slope, use_curvature)

        val_noise_config = {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0}
        band_key = band_name.lower()
        if band_key in noise_config:
            val_noise_config[band_key] = noise_config[band_key]
        val_dataset = SingleBandDataset(val_data, band_name, band_config, val_noise_config,
                                        use_slope, use_curvature)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {device}")

        in_channels = train_dataset.num_channels
        model = SingleBandFlow(band_name, band_config['len'],
                               num_inputs=config['model']['num_inputs'],
                               num_hidden=config['model']['num_hidden'],
                               num_layers=config['model']['num_layers'],
                               context_dim=config['model']['context_dim'],
                               in_channels=in_channels).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_T_max)

        print(f"Noise Mode: {'COMPLEX (Glitch)' if noise_config['USE_COMPLEX_NOISE'] else 'SIMPLE (Gaussian)'}")
        print(f"Features: slope={use_slope}, curvature={use_curvature}")

        # 参数强调权重移到当前 device
        aux_weight = _aux_weight
        param_weights = _param_weights_cpu.to(device) if _param_weights_cpu is not None else None

        # ---- Curriculum noise schedule ----
        curriculum_cfg = config.get('noise_curriculum', {})
        if curriculum_cfg.get('enabled', False):
            print(f"[Curriculum] warmup={curriculum_cfg.get('warmup_fraction', 0.2):.0%} of {epochs} epochs, "
                  f"scale: {curriculum_cfg.get('start_scale', 0.05)} → {curriculum_cfg.get('end_scale', 1.0)}")

        best_val_loss = float('inf')
        patience_counter = 0
        patience = config['training'].get('early_stopping', {}).get('patience', 100)
        min_epochs = config['training'].get('early_stopping', {}).get('min_epochs', 500)

        for epoch in range(epochs):
            # ---- 每个 epoch 刷新噪声缓存 (向量化, 毫秒级) ----
            noise_scale = _get_curriculum_scale(epoch, epochs, curriculum_cfg)
            train_dataset.refresh_noisy_cache(noise_scale)
            val_dataset.refresh_noisy_cache()  # 验证集始终用全噪声

            model.train()
            train_loss = 0.0

            for curve, params in train_loader:
                curve, params = curve.to(device), params.to(device)

                loss_flow = -torch.mean(model.log_prob(params, curve))

                # 辅助参数预测损失
                if aux_weight > 0 and param_weights is not None:
                    pred = model.predict_params(curve)
                    loss_aux = torch.mean(param_weights * (pred - params) ** 2)
                    loss = loss_flow + aux_weight * loss_aux
                else:
                    loss = loss_flow

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for curve, params in val_loader:
                    curve, params = curve.to(device), params.to(device)
                    val_loss += -torch.mean(model.log_prob(params, curve)).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scalers': train_dataset.scalers,
                    'param_scaler': train_dataset.param_scaler,
                    'band_config': band_config,
                    'band_name': band_name,
                    'num_channels': in_channels,
                    'use_slope': use_slope,
                    'use_curvature': use_curvature,
                    'config': config,
                }, save_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch < 20:
                ns_str = f" | NS: {noise_scale:.3f}" if curriculum_cfg.get('enabled', False) else ""
                print(f"Epoch {epoch:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} (Best: {best_val_loss:.4f}){ns_str}")

            if patience_counter >= patience and epoch > min_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\n{band_name} training complete! Best model saved to: {save_path}")
        results[band_name] = best_val_loss

    print("\n" + "="*60)
    print("All single-band training complete!")
    print("="*60)
    for band_name, loss in results.items():
        print(f"  {band_name}: Best Val Loss = {loss:.4f}")

    return results


# ==========================================
# 3.5 实验包: 1 joint + 3 single 一键运行
# ==========================================

def _build_single_subconfig(pkg_cfg, package_dir, band_name):
    """从 package config 派生出单个 single-band 子任务的 config。"""
    sub = copy.deepcopy(pkg_cfg)
    sub.pop('tasks', None)
    sub.pop('output', None)
    sub['name'] = f"{pkg_cfg.get('name', 'package')}_{band_name}"
    sub['dataset'] = {
        'type': 'single',
        'data_dir': pkg_cfg['dataset']['single_dir'],
        'bands': [band_name],
        'band_files': pkg_cfg['dataset']['single_files'],
    }
    sub['output'] = {
        'save_dir': os.path.join(package_dir, 'single'),
        'save_name': f'{band_name}_flow.pt',
    }
    return sub


def _build_joint_subconfig(pkg_cfg, package_dir):
    """从 package config 派生出 joint 子任务的 config。"""
    sub = copy.deepcopy(pkg_cfg)
    sub.pop('tasks', None)
    sub.pop('output', None)
    sub['name'] = f"{pkg_cfg.get('name', 'package')}_joint"
    sub['dataset'] = {
        'type': pkg_cfg['dataset'].get('joint_type', 'concat'),
        'path': pkg_cfg['dataset']['joint_path'],
    }
    sub['output'] = {
        'save_dir': os.path.join(package_dir, 'joint'),
        'save_name': 'flow_joint.pt',
    }
    return sub


def train_experiment_package(config):
    """运行一个完整的实验包: 1 joint + 3 single, 共享所有训练超参数。

    必需的 config 字段:
        dataset.joint_path:    joint 数据集 JSON 路径
        dataset.single_dir:    single-band 数据集所在目录
        dataset.single_files:  {'PTA': '...', 'LISA': '...', 'LIGO': '...'}
        output.package_dir:    实验包根目录
        其他字段 (features, training, model, noise, ...) 与普通 config 一致.

    可选:
        tasks:  list, 默认 ['LIGO', 'LISA', 'PTA', 'joint']
                顺序: LIGO 数据最少先跑, 早暴露配置错误; joint 最后.
        dataset.joint_type: 'concat' (默认) 或 'conti'
    """
    package_dir = config['output']['package_dir']
    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(os.path.join(package_dir, 'single'), exist_ok=True)
    os.makedirs(os.path.join(package_dir, 'joint'), exist_ok=True)

    # 保存 config 快照, 防止后续追溯时找不到当时的设置
    with open(os.path.join(package_dir, 'package_config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # tee stdout → 同时写终端和 training_log.txt
    log_path = os.path.join(package_dir, 'training_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    try:
        tasks = config.get('tasks', ['LIGO', 'LISA', 'PTA', 'joint'])
        print("\n" + "#" * 70)
        print(f"# EXPERIMENT PACKAGE: {config.get('name', 'unnamed')}")
        print(f"# Description: {config.get('description', 'N/A')}")
        print(f"# Output dir:  {package_dir}")
        print(f"# Tasks:       {tasks}")
        print("#" * 70)

        for task in tasks:
            print("\n" + "#" * 70)
            print(f"# [PACKAGE TASK]  {task}")
            print("#" * 70)

            if task == 'joint':
                sub_cfg = _build_joint_subconfig(config, package_dir)
                train_multiband(sub_cfg)
            elif task in ('PTA', 'LISA', 'LIGO'):
                sub_cfg = _build_single_subconfig(config, package_dir, task)
                train_single_band(sub_cfg)
            else:
                print(f"[WARNING] Unknown task '{task}', skipping")

        print("\n" + "#" * 70)
        print(f"# PACKAGE COMPLETE: {package_dir}")
        print("#" * 70)
        print("Output files:")
        for task in tasks:
            if task == 'joint':
                print(f"  joint:  {os.path.join(package_dir, 'joint', 'flow_joint.pt')}")
            elif task in ('PTA', 'LISA', 'LIGO'):
                print(f"  {task:<6}: {os.path.join(package_dir, 'single', f'{task}_flow.pt')}")
        print(f"  log:    {log_path}")
    finally:
        sys.stdout = original_stdout
        log_file.close()


# ==========================================
# 4. 主函数
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Unified Normalizing Flow Training')

    # 配置文件 (优先级最高)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides other arguments)')

    # 命令行参数 (作为备选)
    parser.add_argument('--dataset-type', type=str, default='concat',
                        choices=['concat', 'conti', 'single'],
                        help='Dataset type')
    parser.add_argument('--band', type=str, default='all',
                        choices=['PTA', 'LISA', 'LIGO', 'all'],
                        help='Which band to train (single mode)')
    _wdir = os.path.dirname(os.path.abspath(__file__))
    _wpfile = os.path.join(_wdir, 'paths.json')
    _wp = json.load(open(_wpfile, encoding='utf-8')) if os.path.isfile(_wpfile) else {}
    def _wrp(key, default):
        v = str(_wp.get(key, default))
        return v if os.path.isabs(v) else os.path.abspath(os.path.join(_wdir, v))
    parser.add_argument('--data-dir', type=str,
                        default=_wrp('data_dir', '/root/autodl-tmp/SageNetR/.workspace/dataset'),
                        help='Data directory')
    parser.add_argument('--json-path', type=str, default=None,
                        help='Direct path to JSON file')
    parser.add_argument('--save-dir', type=str,
                        default=_wrp('save_dir', '/root/autodl-tmp/SageNetR/.workspace/new/models'),
                        help='Directory to save models')
    parser.add_argument('--save-name', type=str, default='flow.pt',
                        help='Save filename')
    parser.add_argument('--no-slope', action='store_true',
                        help='Disable slope feature')
    parser.add_argument('--no-curvature', action='store_true',
                        help='Disable curvature feature')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--ligo-noise', type=str, default=None,
                        help='Path to LIGO noise file (C_O1_O2_O3.dat or AplusDesign.txt)')
    parser.add_argument('--ligo-noise-type', type=str, default='cc_spectrum',
                        choices=['cc_spectrum', 'aplus_asd'],
                        help='LIGO noise data type')
    parser.add_argument('--ligo-noise-mode', type=str, default='physical',
                        choices=['physical', 'logscale'],
                        help='LIGO noise injection mode')
    parser.add_argument('--ligo-noise-scale', type=float, default=1.0,
                        help='LIGO noise scale factor')

    args = parser.parse_args()

    # 如果提供了配置文件，使用配置文件
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")

        # 优先级: package mode > single > joint
        if 'package_dir' in config.get('output', {}):
            train_experiment_package(config)
        elif config['dataset']['type'] == 'single':
            train_single_band(config)
        else:
            train_multiband(config)
        return

    # 否则使用命令行参数构建配置
    config = {
        'name': 'cli_config',
        'description': 'Configuration from command line arguments',
        'dataset': {
            'type': args.dataset_type,
            'path': args.json_path or os.path.join(args.data_dir, 'joint/Dataset_PTA_LISA_LIGO.json'),
        },
        'features': {
            'use_slope': not args.no_slope,
            'use_curvature': not args.no_curvature,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': 256,
            'learning_rate': 0.0002,
            'weight_decay': 1e-5,
            'grad_clip': 5.0,
            'scheduler_T_max': 300,
        },
        'model': {
            'num_inputs': 9,
            'num_hidden': 128,
            'num_layers': 10,
            'context_dim': 64,
        },
        'noise': {
            'use_complex': False,
            'level': 1.0,
            'glitch_prob': 0.5,
        },
        'output': {
            'save_dir': args.save_dir,
            'save_name': args.save_name,
        },
    }

    # ---- 物理噪声配置 (CLI 模式) ----
    # LIGO: 需要外部数据文件
    if args.ligo_noise:
        config['noise']['ligo'] = {
            'noise_file': args.ligo_noise,
            'noise_type': args.ligo_noise_type,
            'noise_scale': args.ligo_noise_scale,
            'injection_mode': args.ligo_noise_mode,
            'logscale_base': 1.0,
        }
    # LISA: 内置解析公式, 默认启用 (CLI 下自动开启)
    # 若不想启用可通过 JSON 配置 "enabled": false
    config['noise']['lisa'] = {
        'enabled': True,
        'noise_source': 'analytical',
        'noise_scale': 1.0,
        'injection_mode': 'physical',
        'T_obs': 4.0,
        'include_confusion': True,
    }
    # PTA: 需要 NANOGrav .npy 文件, CLI 下暂不启用
    # 使用 JSON 配置: "pta": {"noise_dir": "/path/to/NANOGrav15yr", ...}

    if args.dataset_type == 'single':
        config['dataset']['bands'] = ['PTA', 'LISA', 'LIGO'] if args.band == 'all' else [args.band]
        config['dataset']['data_dir'] = os.path.join(args.data_dir, 'single')
        config['dataset']['band_files'] = SINGLE_BAND_PATHS
        config['output']['save_dir'] = os.path.join(args.save_dir, 'single')
        train_single_band(config)
    else:
        if args.dataset_type == 'conti':
            config['dataset']['path'] = args.json_path or os.path.join(args.data_dir, 'joint/Dataset_PTA_LISA_LIGO_conti.json')
        train_multiband(config)


if __name__ == "__main__":
    main()