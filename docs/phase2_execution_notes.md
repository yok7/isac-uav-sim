# 第二阶段执行说明（已落地）

## 对应老师任务映射

- 任务3（通感一体化波形生成）
  - 模块：`waveform/isac_ofdm.py`
  - 能力：可调 `通信资源比例`、`导频密度`、`子载波间隔`，并输出 OFDM 时域波形与资源掩码。

- 任务4（多天线阵列建模）
  - 模块：`channel/ula_scene.py`
  - 能力：ULA 快拍仿真，支持 LoS / NLoS（含反射路径）两种场景。

- 任务5（先进DOA算法调研、复现与对比）
  - 经典算法：`Beamforming/MUSIC/ESPRIT`（`doa/classical.py`）
  - 前沿代理：`TransDOA-surrogate`、`OGFVBI-surrogate`（`doa/advanced.py`）
  - 说明：当前为可复现代理实现，用于先完成统一实验框架与对比流程；后续可替换为论文官方实现。

- 任务6（通感性能权衡分析）
  - 脚本：`experiments/phase2_tradeoff.py`
  - 输出：通信速率代理、BER代理、DOA-RMSE、Pareto 前沿曲线。

## 一键运行

```powershell
conda run -n isac_uav python .\experiments\phase2_doa_benchmark.py
conda run -n isac_uav python .\experiments\phase2_tradeoff.py
```

## 输出位置

- `results/phase2/phase2_benchmark_raw.csv`
- `results/phase2/phase2_benchmark_summary.csv`
- `results/phase2/phase2_rmse_vs_snr.png`
- `results/phase2/phase2_rmse_vs_snapshots.png`
- `results/phase2/phase2_runtime_comparison.png`
- `results/phase2/phase2_los_nlos_adaptability.png`
- `results/phase2/phase2_tradeoff_full.csv`
- `results/phase2/phase2_tradeoff_pareto.csv`
- `results/phase2/phase2_pareto_rate_vs_rmse.png`
