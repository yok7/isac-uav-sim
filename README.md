# ISAC-UAV Transition Project

This repository is a starter workspace for the transition project:
"Integrated Sensing and Communication (ISAC) waveform design and multi-antenna DOA estimation for low-altitude UAV scenarios".

## Directory Layout

- `waveform/`: ISAC waveform generation (5G NR OFDM + simplified)
- `doa/`: DOA estimation algorithms (classical + robust + joint)
- `channel/`: channel modeling (Sionna RT ray-tracing + statistical multipath)
- `experiments/`: experiment scripts and configurations
- `results/`: generated figures, logs, and experiment outputs
- `report/`: technical report draft materials
- `docs/`: reading notes, plans, and weekly report templates
- `scripts/`: setup and utility scripts

## Module Structure

```
isac_uav/
├── waveform/                    # 波形生成模块
│   ├── __init__.py
│   ├── isac_waveform.py        # Sionna 5G NR PUSCH OFDM 波形生成
│   │   ├── SionnaWaveformGenerator   # 完整OFDM收发机
│   │   ├── IsacWaveformConfig        # 波形配置
│   │   ├── estimate_channel_ls()      # LS信道估计
│   │   └── equalize_ofdm_symbols()    # OFDM均衡
│   └── isac_ofdm.py            # 简化版OFDM（保留原有用法）
│
├── channel/                    # 信道模型模块
│   ├── __init__.py
│   ├── sionna_rt_channel.py    # Sionna RT射线追踪信道
│   │   ├── SionnaRTChannel           # 射线追踪信道
│   │   ├── SionnaRTConfig            # 配置（LoS/NLoS/多径/城市峡谷）
│   │   ├── ChannelModelType          # 信道类型枚举
│   │   └── generate_cir_dataset()    # CIR数据集生成
│   ├── multipath_channel.py    # 几何/统计多径信道
│   │   ├── MultipathChannel          # 多径信道（支持Rician/Rayleigh）
│   │   ├── MultipathConfig           # 配置（延迟/角度扩展）
│   │   ├── GeometricChannel          # 几何信道模型
│   │   └── StatisticalChannel       # 统计信道模型
│   └── ul
│
├── doa/                        # DOA估计模块
│   ├── __init__.py
│   ├── classical.py            # 经典DOA（保留原有用法）
│   ├── robust_doa.py           # 多径鲁棒DOA
│   │   ├── SpatialSmoothing                 # 空间平滑
│   │   ├── RootMUSIC                        # 根MUSIC
│   │   ├── BeamspaceMUSIC                   # 波束空间MUSIC
│   │   ├── multipath_robust_music()         # 多径鲁棒MUSIC
│   │   ├── esprit_with_spatial_smoothing() # 平滑ESPRIT
│   │   └── estimate_num_sources_aic/md     # 信源数估计
│   ├── joint_estimator.py      # 联合估计算法
│   │   ├── JointRangeAngleEstimator # 联合距离-角度估计
│   │   ├── RangeAngleSearchGrid      # 搜索网格
│   │   ├── delay_music_spectrum()    # 延迟MUSIC
│   │   └── coherent_music_spectrum() # 相干MUSIC
│   ├── metrics.py
│   └── advanced.py
│
└── experiments/
    └── isac_end2end_simulation.py  # 端到端仿真启动脚本
```

## Quick Start (Windows PowerShell)

1. Create and initialize base environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Profile base -EnvName isac_uav
```

2. (Optional) Install Sionna stack:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Profile sionna -EnvName isac_uav
```

3. Verify environment:

```powershell
conda run -n isac_uav python .\scripts\check_env.py --profile sionna
```

4. Run week-1 first plot task:

```powershell
conda run -n isac_uav python .\experiments\plot_ula_pattern.py
```

The output figure will be saved to `results/week1_ula_pattern.png`.

5. Run classical DOA baseline demo:

```powershell
conda run -n isac_uav python .\experiments\run_classical_doa_demo.py
```

6. Run Sionna 2.x smoke test:

```powershell
conda run -n isac_uav python .\experiments\run_sionna_smoke_test.py
```

7. Run Phase-2 DOA benchmark:

```powershell
conda run -n isac_uav python .\experiments\phase2_doa_benchmark.py
```

8. Run Phase-2 communication-sensing tradeoff experiment:

```powershell
conda run -n isac_uav python .\experiments\phase2_tradeoff.py
```

Phase-2 outputs are generated under `results/phase2/`.

## End-to-End ISAC Simulation

Run the complete simulation pipeline:

```powershell
# Full simulation (all channel models + all algorithms)
conda run -n isac_uav python .\experiments\isac_end2end_simulation.py --mode full --num-trials 100

# Channel model comparison
conda run -n isac_uav python .\experiments\isac_end2end_simulation.py --compare-channels --num-trials 100

# Multipath impact study
conda run -n isac_uav python .\experiments\isac_end2end_simulation.py --multipath-study --num-trials 100

# Quick test
conda run -n isac_uav python .\experiments\isac_end2end_simulation.py --mode quick --num-trials 20
```

Outputs: `results/end2end/*.png` (RMSE curves, runtime comparison, channel comparison)

## External References

- `third_party/DOA/`: cloned from [AmgadSalama/DOA](https://github.com/AmgadSalama/DOA) for tutorial reference and cross-checking.

## Sionna Note

This project now targets Sionna 2.x, which uses PyTorch. The local `isac_uav`
environment has been verified with GPU PyTorch `cu128` on the RTX 4060 Laptop
GPU. Use `device="cuda:0"` for Sionna GPU blocks.

## Suggested Workflow

- Keep all scripts version-controlled with Git.
- Save weekly outputs under `results/weekXX/`.
- Use `docs/weekly_report_template.md` for weekly mentor updates.
