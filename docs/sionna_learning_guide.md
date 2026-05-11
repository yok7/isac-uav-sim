# Sionna 2.x 学习路线：5G NR OFDM、信道、通感与阵列

## 0. 先明确你要学的模块

当前阶段不需要一口气读完整个 Sionna。建议先只盯住四组 API：

- `sionna.phy.nr`：5G NR PUSCH 配置、发射机、DMRS、MCS、Transport Block。
- `sionna.phy.ofdm`：OFDM ResourceGrid、OFDM 调制解调、导频映射、信道估计。
- `sionna.phy.channel`：Rayleigh、TDL/CDL、OFDMChannel、TimeChannel。
- `sionna.phy.channel.tr38901`：3GPP 天线阵列、CDL/UMi/UMa/RMa 场景。

## 1. 官方仓库里优先看哪些

你已经 clone 了官方仓库，建议按这个顺序看：

- `third_party/sionna/tutorials/phy/Discover_Sionna.ipynb`
- `third_party/sionna/tutorials/phy/Hello_World.ipynb`
- `third_party/sionna/tutorials/phy/5G_NR_PUSCH.ipynb`
- `third_party/sionna/tutorials/phy/OFDM_MIMO_Detection.ipynb`
- `third_party/sionna/tutorials/phy/MIMO_OFDM_Transmissions_over_CDL.ipynb`

前两个是入门，第三个直接对应 5G NR 物理层，后两个对应 OFDM/MIMO/信道。

## 2. 本项目里先跑哪些脚本

先跑最小环境检查：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n isac_uav python scripts/check_env.py --profile sionna
conda run -n isac_uav python scripts/sionna_smoke_test.py
```

然后跑学习脚本：

```powershell
conda run -n isac_uav python experiments/sionna_learning/01_nr_pusch_ofdm_waveform.py
conda run -n isac_uav python experiments/sionna_learning/02_isac_waveform_tradeoff.py
conda run -n isac_uav python experiments/sionna_learning/03_array_uav_scene.py
conda run -n isac_uav python experiments/sionna_learning/04_ofdm_channel_modeling.py
```

输出都在：

```text
results/sionna_learning/
```

## 3. 任务3：通感一体化波形生成怎么学

核心理解：

- 通信数据：占用 PUSCH 的数据 RE，用来承载 Transport Block。
- 感知资源：可以先把 DMRS pilots 当成感知探测资源。
- 通感权衡：更多导频通常提升信道/目标估计质量，但会挤占数据 RE；更大带宽改善距离分辨率，但资源和采样压力更大。

先看脚本：

- `01_nr_pusch_ofdm_waveform.py`：看 Sionna 如何生成 5G NR PUSCH 频域资源网格和时域 OFDM 基带波形。
- `02_isac_waveform_tradeoff.py`：修改 `subcarrier_spacing`、`n_size_grid`、`dmrs.additional_position`，观察吞吐率、DMRS 占比和距离分辨率 proxy 的变化。
- `04_ofdm_channel_modeling.py`：把 PUSCH OFDM 资源网格送入 Rayleigh OFDM 信道，理解 `channel_model -> OFDMChannel -> y,h_freq` 的基本接口。

关键参数：

- `cfg.carrier.subcarrier_spacing`：15/30/60 kHz。
- `cfg.carrier.n_size_grid`：资源块数量，决定占用带宽。
- `cfg.dmrs.additional_position`：DMRS 额外位置，决定导频密度。
- `cfg.tb.mcs_index`：调制编码阶数，影响通信速率。

## 4. 任务4：多天线阵列建模怎么学

核心理解：

- ULA/UPA 是阵元在空间中的几何坐标集合。
- 对某个来波方向，阵列响应向量由每个阵元相对相位组成。
- 阵元间距通常取半波长，避免明显栅瓣。
- 无人机目标可以先抽象为一个远场点目标，位置决定方位角、俯仰角和距离。

先看脚本：

- `03_array_uav_scene.py`：用 Sionna 的 `AntennaArray` 配置 ULA/UPA，同时显式计算阵列响应向量。

你需要掌握的公式：

```text
a(theta, phi)[m] = exp(j * 2*pi/lambda * r_m^T u(theta, phi)) / sqrt(M)
```

其中 `r_m` 是第 m 个阵元坐标，`u(theta, phi)` 是目标方向单位向量。

## 5. 阶段产出建议

可以提交以下内容：

- 可运行代码：`experiments/sionna_learning/*.py`
- 结果图：`results/sionna_learning/*.png`
- 参数扫描表：`results/sionna_learning/02_isac_waveform_tradeoff.csv`
- 简短说明：解释哪些资源用于通信，哪些资源用于感知，以及导频密度/带宽/阵列规模的影响。

## 6. 暂时不要做的事

- 不要把 `third_party/sionna` 源码 `pip install -e` 到当前环境，先保持 pip 安装版 Sionna 2.0.1 稳定。
- 不要一开始就上完整 ray tracing。先把 PHY 层资源网格、信道接口、阵列响应这些基本块打牢。
- 不要先追求复杂指标。第一阶段先能解释 shape、参数、图和趋势。
