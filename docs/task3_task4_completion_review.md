# Task 3/4 completion review

## Task 3: ISAC waveform generation

Status: complete for the current stage after adding an actual single-target
range simulation. The earlier parameter sweep is retained as a theoretical
design scan, not as the final sensing-accuracy result.

Covered items:

- Sionna 2.x/PyTorch environment verified on GPU.
- 5G NR PUSCH baseband waveform generation via `PUSCHConfig` and `PUSCHTransmitter`.
- Frequency-domain resource grid and time-domain OFDM waveform saved.
- DMRS pilots identified as sensing/channel-estimation resources within the same OFDM slot.
- OFDM channel interface verified with `RayleighBlockFading` and `OFDMChannel`.
- Waveform parameters swept:
  - subcarrier spacing: 15/30/60 kHz,
  - resource blocks: 4/8/16,
  - DMRS density: `additional_position = 0/1/2`,
  - frame/symbol allocation: `[0,14]`, `[0,10]`, `[2,12]`.
- Communication metric: transport-block throughput proxy.
- Sensing proxy in the design scan: range-resolution and DMRS-density based
  precision proxy.
- Actual sensing simulation: single UAV-target OFDM echo generated on top of
  the Sionna PUSCH resource grid; range estimated from DMRS REs; Monte Carlo
  range RMSE reported.
- Range+DOA simulation: the same DMRS echo is extended to an 8-element ULA;
  range is estimated from subcarrier phase and DOA is estimated with MUSIC.

Key outputs:

- `results/sionna_learning/01_pusch_resource_grid.png`
- `results/sionna_learning/01_pusch_time_waveform.png`
- `results/sionna_learning/04_channel_response.png`
- `results/sionna_learning/05_task3_isac_waveform_sweep.csv`
- `results/sionna_learning/05_task3_rate_sensing_tradeoff.png`
- `results/sionna_learning/05_task3_dmrs_density_effect.png`
- `results/sionna_learning/07_task3_actual_isac_range_sim.csv`
- `results/sionna_learning/07_task3_actual_rate_vs_range_rmse.png`
- `results/sionna_learning/07_task3_theory_vs_sim_rmse.png`
- `results/sionna_learning/08_task3_task4_isac_range_doa_sim.csv`
- `results/sionna_learning/08_isac_range_doa_rmse.png`
- `results/sionna_learning/08_isac_music_doa_spectrum.png`

Remaining limitation:

- The actual sensing simulation is still single-target and simplified. It does
  not yet include multi-target detection, clutter, self-interference, or
  Doppler/velocity estimation.

## Task 4: Multi-antenna array modeling

Status: complete for the current stage.

Covered items:

- Sionna `AntennaArray` used to configure ULA and UPA.
- Array element coordinates extracted and saved.
- Explicit array response vector implemented:

```text
a(theta, phi)[m] = exp(j * 2*pi/lambda * r_m^T u(theta, phi)) / sqrt(M)
```

- Low-altitude UAV scene configured:
  - BS position: `[0, 0, 10] m`,
  - UAV position: `[120, 45, 80] m`,
  - range: about `146 m`,
  - azimuth: about `20.56 deg`,
  - elevation: about `28.64 deg`.
- ULA azimuth response and UPA azimuth-elevation response generated.

Key outputs:

- `results/sionna_learning/03_array_uav_geometry.png`
- `results/sionna_learning/03_ula_beam_pattern.png`
- `results/sionna_learning/06_task4_ula_upa_geometry.png`
- `results/sionna_learning/06_task4_upa_az_el_response.png`
- `results/sionna_learning/06_task4_array_response.npz`

Remaining limitation:

- The ULA geometry has been connected to the simplified UAV echo model and
  MUSIC DOA estimator. A later stage should extend this to UPA 2D DOA,
  multipath/clutter, and Sionna RT-style propagation.
