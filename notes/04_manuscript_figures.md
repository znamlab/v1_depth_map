# 04 - Manuscript Figures Assembly Tracking

This document tracks the assembly status of all Main and Supplementary figures for the manuscript, including individual panel generation from notebooks, vector assembly (in Adobe Illustrator / Inkscape), and final PDF/SVG compilation.

---

## 1. Main Figures Assembly Status

*Current active version directory: `v1_manuscript_figures/ver_rev1/`.*

| Figure | Description | Generating Notebook(s) | Exported Panel Files | Assembly Status | Tasks & Action Items |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Figure 1** | Depth selectivity in mouse V1 | [`figure_depth_selectivity.ipynb`](../v1_depth_map/figures/figure_depth_selectivity.ipynb) | `fig1.svg`<br>`fig1_4examples.svg`<br>`fig1_4examples_rsq.svg` | ✅ Assembled | - Check font consistency (Arial / Arial Narrow)<br>- Align example raster plots with mean curve overlays |
| **Figure 2** | Depth representation vs speed & flow (RS × OF) | [`figure_depth_selectivity.ipynb`](../v1_depth_map/figures/figure_depth_selectivity.ipynb)<br>[`figure_rsof_integration.ipynb`](../v1_depth_map/figures/figure_rsof_integration.ipynb) | `fig2.svg`<br>`fig2.pdf`<br>`fig2b.pdf`<br>`fig2_supp_best_model_all_depth_selective.pdf` | ✅ Assembled | - Update with treadmill-cut decoder results (`run_full_cut.py`) |
| **Figure 3** | Receptive fields & 3D tuning | [`figure_rf.ipynb`](../v1_depth_map/figures/figure_rf.ipynb) | `fig_3d_rfs.pdf`<br>`fig_rf_example_fov.svg`<br>`fig_rf_examples.pdf`<br>`rf_position_corrected.svg`<br>`v1_map_vrange_0.1_2.0_a0.3.svg` | 🎨 In Progress | - Confirm retinotopic map overlay scaling<br>- Review contralateral vs ipsilateral ROIs |
| **Figure 4** | Depth cell properties & population organization | [`figure_depth_cells.ipynb`](../v1_depth_map/figures/figure_depth_cells.ipynb) | `fig_depth_cells.svg`<br>`fig_depth_cells_examples.svg`<br>`fig_depth_cells_examples_overlaid.svg` | 🎨 In Progress | - Refine example traces and tuning overlays |

---

## 2. Supplementary Figures Assembly Status

| Figure | Description | Generating Notebook | Exported Files in `ver_rev1` | Assembly Status | Tasks & Action Items |
| **Fig S1** | Visual stimulus synchronization (Schematic, Sequence vs Photodiode, Position, Lag histogram) | [`figsupp_vis_stim_sync.ipynb`](../v1_depth_map/figures/figsupp_vis_stim_sync.ipynb) | `fig_supp_vis-stim.svg`<br>`lag_example.svg` | ✅ Ready / Assembled | Panels A–D complete (embedded schematic & layout matched) |
| **Fig S2** | Speed & Eye tracking controls (Running speed, Optic flow, Eye tracking) | [`figsupp_speed.ipynb`](../v1_depth_map/figures/figsupp_speed.ipynb) | `figsupp_speed.svg` | ✅ Ready / Assembled | Panels A–L complete (full frame eye image, unclipped labels) |
| **Fig S5** | Open-loop vs closed-loop responses | [`figsupp_openloop.ipynb`](../v1_depth_map/figures/figsupp_openloop.ipynb) | `fig_supp_openloop.svg` | 🎨 In Progress | Verify playback correlation plots |
| **Fig S6** | Size control experiment | [`figsupp_size_control.ipynb`](../v1_depth_map/figures/figsupp_size_control.ipynb) | `fig_size_control/` | ✅ Assembled | 2P reruns & `neurons_df.pickle` updated (`ast_neuropil=False`) |
| **Fig S7** | Multi-day stability | [`figsupp_multidays.ipynb`](../v1_depth_map/figures/figsupp_multidays.ipynb) | `figsupp_multidays.pdf`<br>`figsupp_multidays_10cells.pdf`<br>`figsupp_multidays_cell1_rsof_matrix.pdf` | 🎨 In Progress | Longitudinal ROI tracking panels |
| **Fig S8** | Simulation control | [`figsupp_simulation_control.ipynb`](../v1_depth_map/figures/figsupp_simulation_control.ipynb) | `fig_supp_simulation_control.svg`<br>`fig_supp_simulation_control.pdf` | 🎨 In Progress | Synthetic dataset validation |

---

## 3. Vector Layout & Style Guidelines

- **Fonts**: Arial / Arial Narrow (embed TTF files from `v1_manuscript_figures/fonts/`).
- **Color Palette**: Use project standard colormaps for depth (e.g., Viridis / Turbo) and model comparisons.
- **Export Standards**:
  - Export vector graphics as `.svg` with editable text.
  - Final manuscript assembly compiled to vectorized multi-page `.pdf`.
