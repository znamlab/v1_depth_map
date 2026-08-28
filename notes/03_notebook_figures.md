# 03 - Notebook Figures Tracking

This document tracks the execution status, resource utilization, and action items for all Jupyter notebooks located under `v1_depth_map/figures/`.

---

## 1. Notebooks Execution Status Table

*Execution benchmarks recorded on local / HPC runner via [run_figures_pipeline.py](../run_figures_pipeline.py).*

| Notebook | Status | Execution Time | Peak RAM | Target Figures / Panels | Reason / Actions Needed |
| :--- | :---: | :---: | :---: | :--- | :--- |
| [`figure_depth_selectivity.ipynb`](../v1_depth_map/figures/figure_depth_selectivity.ipynb) | ✅ Ready | 49.7 min | 5.14 GB | Fig 1 (Panels A–P full assembly), Fig 2 panels | Full multi-panel publication assembly cell included |
| [`figure_rf.ipynb`](../v1_depth_map/figures/figure_rf.ipynb) | ✅ Success | 7.9 min | 2.31 GB | Fig 3 / RF panels | Receptive field mapping & 3D RF profiles |
| [`figure_rsof_integration.ipynb`](../v1_depth_map/figures/figure_rsof_integration.ipynb) | ✅ Ready | 13.3 min | 4.98 GB | Fig 2 (Panels A–K full assembly) | Full multi-panel publication assembly cell included |
| [`figure_openloop.ipynb`](../v1_depth_map/figures/figure_openloop.ipynb) | ✅ Success | 11.2 min | 5.08 GB | Open-loop responses | Open-loop vs closed-loop comparison |
| [`figure_depth_cells.ipynb`](../v1_depth_map/figures/figure_depth_cells.ipynb) | ✅ Success | - | - | Depth cell examples | Single-cell exemplars and FOVs |
| [`figsupp_speed.ipynb`](../v1_depth_map/figures/figsupp_speed.ipynb) | ✅ Ready | 1.8 min | 2.50 GB | Supp: Speed & Eye tracking | Running speed, optic flow speed, & pupil/gaze tracking (Panels A–L) |
| [`figsupp_openloop.ipynb`](../v1_depth_map/figures/figsupp_openloop.ipynb) | ✅ Success | 1.5 min | 3.59 GB | Supp: Open loop | Playback comparisons |
| [`figsupp_rf.ipynb`](../v1_depth_map/figures/figsupp_rf.ipynb) | ✅ Success | 3.1 min | 1.24 GB | Supp: RF details | RF gradients and contralateral vs ipsilateral |
| [`figsupp_rsof.ipynb`](../v1_depth_map/figures/figsupp_rsof.ipynb) | ✅ Success | 37.2 min | 2.33 GB | Supp: RS × OF matrices | RS × OF 2D grid responses |
| [`figsupp_size_control.ipynb`](../v1_depth_map/figures/figsupp_size_control.ipynb) | ✅ Ready | 0.2 min | 1.30 GB | Supp: Size control (`\label{sup:size}`) | Size-tuning invariance controls & stats |
| [`figsupp_vis_stim_sync.ipynb`](../v1_depth_map/figures/figsupp_vis_stim_sync.ipynb) | ✅ Ready | 7.0 min | 6.33 GB | Supp: Visual sync (`\label{sup:vis_stim}`) | 100% vector Fig S1 with embedded vector schematic |
| [`revisions/multi_days.ipynb`](../v1_depth_map/revisions/multi_days.ipynb) | ✅ Success | - | - | Supp: Multiday stability | Longitudinal tracking across days (candidate Fig S7). No `figures/` notebook yet — panels still live in `revisions/` |
| [`figsupp_simulation_control.ipynb`](../v1_depth_map/figures/figsupp_simulation_control.ipynb) | ✅ Ready | - | - | Supp: Simulation control | Synthetic dataset validation (candidate Fig S8) |

---

## 2. Detailed Notebook Tasks & Action Items

- [x] **`figsupp_size_control.ipynb`**:
  - [x] Submit 2P rerun without neuropil subtraction for 3 size control sessions (Jobs `52277005`–`52277007`).
  - [x] Run `size_control_all_sessions.py` to produce updated `neurons_df.pickle` with `ast_neuropil=False`.
  - [x] Validate notebook execution (1,822 neurons loaded, 367 depth-tuned).

- [ ] **`figsupp_simulation_control.ipynb`**:
  - [x] Slurm simulation jobs `52276868`–`52276875` completed on NEMO (`rerun_simulation_tdecay2_areanorm.py`).
  - [ ] Sync simulation `.parquet` files to local drive, drop `tread_kwargs=dict(method="model")` override in cell 10, and run notebook.

- [ ] **`figure_rsof_integration.ipynb`**:
  - Apply the cut treadmill filter (`cut_treadmill = True` / `_motor_cut`) to match the latest ridge decoder logic from `run_full_cut.py`.

### 🟡 Medium Priority
- [ ] Verify that all notebooks save outputs to versioned directory `v1_manuscript_figures/ver_rev1/` using [paths.py](../v1_depth_map/paths.py).
- [ ] Ensure `reload = False` is set after initial cache generation for fast execution during figure refinement.

---

## 3. How to Run Notebooks via Automated Pipeline

Run the automated async runner with resource monitoring and automatic timeout management:

```bash
# Run all figure notebooks sequentially with resource monitoring
python run_figures_pipeline.py

# Run a specific notebook or with custom timeout
python run_figures_pipeline.py --notebooks figure_depth_selectivity.ipynb --timeout 3600
```
