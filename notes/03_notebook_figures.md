# 03 - Notebook Figures Tracking

This document tracks the execution status, resource utilization, and action items for all Jupyter notebooks located under `v1_depth_map/figures/`.

---

## 1. Notebooks Execution Status Table

*Execution benchmarks recorded on local / HPC runner via [run_figures_pipeline.py](../run_figures_pipeline.py).*

| Notebook | Status | Execution Time | Peak RAM | Target Figures / Panels | Reason / Actions Needed |
| :--- | :---: | :---: | :---: | :--- | :--- |
| [`figure1_depth_selectivity.ipynb`](../v1_depth_map/figures/figure1_depth_selectivity.ipynb) | ✅ Ready | 49.7 min | 5.14 GB | Fig 1 (Panels A–P full assembly) | Full multi-panel publication assembly cell included; also caches `fig1/neurons_df_all.pickle` for Fig S3 |
| [`figure2_rsof_integration.ipynb`](../v1_depth_map/figures/figure2_rsof_integration.ipynb) | ✅ Ready | 13.3 min | 4.98 GB | Fig 2 (Panels A–E: RS × OF integration models) | Closed-loop integration models; also builds the Fig S5 panel O stacked bar |
| [`figure2_openloop.ipynb`](../v1_depth_map/figures/figure2_openloop.ipynb) | ✅ Success | 11.2 min | 5.08 GB | Fig 2 (Panels F–L: open loop) | Open-loop vs closed-loop comparison |
| [`figure3_depth_cells.ipynb`](../v1_depth_map/figures/figure3_depth_cells.ipynb) | ✅ Success | 2.5 min | 3.85 GB | Fig 3 (Panels A–L: Motorized wheel & depth cells) | Full multi-panel publication assembly cell included (saves `fig_depth_cells.svg`) |
| [`figure4_receptive_fields.ipynb`](../v1_depth_map/figures/figure4_receptive_fields.ipynb) | ✅ Ready | - | - | Fig 4 (Panels A–H: 3D receptive fields) | Multi-depth protocol (`SpheresPermTubeReward_multidepth`); 3D RF isosurfaces & FOV retinotopy |
| [`figsupp1_vis_stim_sync.ipynb`](../v1_depth_map/figures/figsupp1_vis_stim_sync.ipynb) | ✅ Ready | 7.0 min | 6.33 GB | Fig S1 (`\label{sup:vis_stim}`) | 100% vector figure with embedded vector schematic |
| [`figsupp2_speed.ipynb`](../v1_depth_map/figures/figsupp2_speed.ipynb) | ✅ Ready | 1.8 min | 2.50 GB | Fig S2 (Panels A–L: speeds & eye tracking) | Running speed, optic flow speed & pupil/gaze tracking |
| [`figsupp3_depth_pop.ipynb`](../v1_depth_map/figures/figsupp3_depth_pop.ipynb) | ✅ Ready | - | - | Fig S3 (Panels A–L: depth tuning & population) | Reuses `fig1/neurons_df_all.pickle`; emits `statSuppDepthPop*` |
| [`figsupp4_size_control.ipynb`](../v1_depth_map/figures/figsupp4_size_control.ipynb) | ✅ Ready | 0.2 min | 1.30 GB | Fig S4 (`\label{sup:size}`) | Size-tuning invariance controls & stats |
| [`figsupp5_rsof.ipynb`](../v1_depth_map/figures/figsupp5_rsof.ipynb) | ✅ Success | 37.2 min | 2.33 GB | Fig S5 (Panels A–O: RS × OF matrices) | RS × OF 2D grid responses and the five model fits |
| [`figsupp6_openloop.ipynb`](../v1_depth_map/figures/figsupp6_openloop.ipynb) | ✅ Success | 1.5 min | 3.59 GB | Fig S6 (Panels A–D: open loop) | Playback comparisons |
| [`figsupp7_simulation_control.ipynb`](../v1_depth_map/figures/figsupp7_simulation_control.ipynb) | ✅ Ready | 0.9 min | 2.48 GB | Fig S7 (simulation control) | Synthetic dataset validation |
| [`figsupp8_multidepth_receptive_fields.ipynb`](../v1_depth_map/figures/figsupp8_multidepth_receptive_fields.ipynb) | ✅ Ready | - | - | Fig S8 (Panels A–I: multi- vs single-depth RFs) | RF comparisons, spatial correlations & depth consistency; owns all `\statSuppEight*` macros |
| [`figsupp9_v1_depth_map.ipynb`](../v1_depth_map/figures/figsupp9_v1_depth_map.ipynb) | ✅ Success | 7.9 min | 2.31 GB | Fig S9 (Panels A–G: corrected depth map) | Cortical alignment, gradient z-test; owns all `\statSuppNine*` macros |
| [`figsupp9_rf_distance.ipynb`](../v1_depth_map/figures/figsupp9_rf_distance.ipynb) | ✅ Success | 3.1 min | 1.24 GB | Fig S9 (Panels H–I: uncorrected depth) | Uncorrected V1 map; the pairwise RF distance panels moved to Fig 4 I–K |
| [`unused_single_depth_receptive_fields.ipynb`](../v1_depth_map/figures/unused_single_depth_receptive_fields.ipynb) | 💤 Unused | - | - | None in this manuscript version | Single-depth protocol RFs; superseded by the Fig S8 comparison. Skipped by the pipeline unless named with `--notebooks` |
| [`revisions/multi_days.ipynb`](../v1_depth_map/revisions/multi_days.ipynb) | ✅ Success | - | - | Fig 1N–P (multiday stability) | Longitudinal tracking; the manuscript panels now live in `figure1_depth_selectivity.ipynb` |

---

## 2. Detailed Notebook Tasks & Action Items

- [x] **`figsupp4_size_control.ipynb`**:
  - [x] Submit 2P rerun without neuropil subtraction for 3 size control sessions (Jobs `52277005`–`52277007`).
  - [x] Run `size_control_all_sessions.py` to produce updated `neurons_df.pickle` with `ast_neuropil=False`.
  - [x] Validate notebook execution (1,822 neurons loaded, 367 depth-tuned).

- [x] **`figsupp7_simulation_control.ipynb`**:
  - [x] Slurm simulation jobs `52276868`–`52276875` completed on NEMO (`rerun_simulation_tdecay2_areanorm.py`).
  - [x] Sync simulation `.parquet` files to local drive, drop `tread_kwargs=dict(method="model")` override in cell 10, and run notebook.

### 🟡 Medium Priority
- [x] Verify that all notebooks save outputs to versioned directory `v1_manuscript_figures/ver_rev1/` using [paths.py](../v1_depth_map/paths.py).
  - All notebooks derive `SAVE_ROOT` from `get_figures_roots(flexilims_session)`; no hardcoded output paths remain.
- [x] Unify font, font size and SVG export across all `figures/` notebooks - see [04 §3.1](04_manuscript_figures.md#31-figure-style-convention-enforced-across-all-figures-notebooks).
  - Single `style.setup_figure_fonts()` setup call, all sizes sourced from `style.FONTSIZE_DICT`, all 27 saves via `style.savefig(..., fig=...)`.
- [ ] Ensure `reload = False` is set after initial cache generation for fast execution during figure refinement.

---

## 3. How to Run Notebooks via Automated Pipeline

Run the automated async runner with resource monitoring and automatic timeout management:

```bash
# Run all figure notebooks sequentially with resource monitoring
python run_figures_pipeline.py

# Run a specific notebook or with custom timeout
python run_figures_pipeline.py --notebooks figure1_depth_selectivity.ipynb --timeout 3600
```
