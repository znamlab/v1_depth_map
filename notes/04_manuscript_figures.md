# 04 - Manuscript Figures Assembly Tracking

This document tracks the assembly status of all Main and Supplementary figures for the manuscript, including individual panel generation from notebooks, vector assembly (in Adobe Illustrator / Inkscape), and final PDF/SVG compilation.

---

## 1. Main Figures Assembly Status (`v1_depth_map.tex`)

*Current active version directory: `v1_manuscript_figures/ver_rev1/`.*

| Figure | LaTeX Label & File | Description | Generating Notebook(s) | Exported Panel Files | Assembly Status | Tasks & Action Items |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Figure 1** | `\label{fig:intro}`<br>`figures/fig1.pdf` | Depth selectivity in mouse V1 | [`figure_depth_selectivity.ipynb`](../v1_depth_map/figures/figure_depth_selectivity.ipynb) | `fig1.svg`<br>`fig1_4examples.svg`<br>`fig1_4examples_rsq.svg` | ✅ Assembled | - Check font consistency (Arial / Arial Narrow)<br>- Align example raster plots with mean curve overlays |
| **Figure 2** | `\label{fig:rsof}`<br>`figures/fig2.pdf` | Depth representation vs speed & flow (RS × OF) | [`figure_depth_selectivity.ipynb`](../v1_depth_map/figures/figure_depth_selectivity.ipynb)<br>[`figure_rsof_integration.ipynb`](../v1_depth_map/figures/figure_rsof_integration.ipynb) | `fig2.svg`<br>`fig2.pdf`<br>`fig2b.pdf`<br>`fig2_supp_best_model_all_depth_selective.pdf` | ✅ Assembled | - Update with treadmill-cut decoder results (`run_full_cut.py`) |
| **Figure 3** | `\label{fig:rf}`<br>`figures/fig3.png` | Receptive fields & 3D tuning | [`figure_rf.ipynb`](../v1_depth_map/figures/figure_rf.ipynb) | `fig_3d_rfs.pdf`<br>`fig_rf_example_fov.svg`<br>`fig_rf_examples.pdf`<br>`rf_position_corrected.svg`<br>`v1_map_vrange_0.1_2.0_a0.3.svg` | 🎨 In Progress | - Confirm retinotopic map overlay scaling<br>- Review contralateral vs ipsilateral ROIs |
| **Figure 4** | `\label{fig:v1map}`<br>`figures/fig4.png` | Depth map across V1 / visual space | [`figure_depth_cells.ipynb`](../v1_depth_map/figures/figure_depth_cells.ipynb) | `fig_depth_cells.svg`<br>`fig_depth_cells_examples.svg`<br>`fig_depth_cells_examples_overlaid.svg` | 🎨 In Progress | - Refine example traces and tuning overlays |

---

## 2. Supplementary Figures Assembly Status

### Integrated in `v1_depth_map.tex`

| Figure | LaTeX Label & File | Description | Generating Notebook | Exported Files in `ver_rev1` | Assembly Status | Details & Action Items |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Fig S1** | `\label{sup:vis_stim}`<br>`figures/fig_supp_vis-stim.png` | Visual stimuli in VR (Schematic, Photodiode sync, Position, Lag histogram) | [`figsupp_vis_stim_sync.ipynb`](../v1_depth_map/figures/figsupp_vis_stim_sync.ipynb) | `fig_supp_vis-stim.svg`<br>`fig_supp_vis-stim.pdf` | ✅ Ready / Assembled | Panels A–D 100% vector (embedded vector schematic, no duplicate axes) |
| **Fig S2** | `\label{sup:speeds}`<br>`figures/fig_supp_speeds.png` | Running, optic flow speeds and eye movements | [`figsupp_speed.ipynb`](../v1_depth_map/figures/figsupp_speed.ipynb) | `figsupp_speed.svg` | ✅ Ready / Assembled | Panels A–L complete (full frame eye image, unclipped labels) |
| **Fig S3** | `\label{sup:size}`<br>`figures/fig_supp_size_control.png` | Virtual depth selectivity invariance to stimulus size | [`figsupp_size_control.ipynb`](../v1_depth_map/figures/figsupp_size_control.ipynb) | `fig_size_control/`<br>`fig_size_tuning.svg` | ✅ Ready / Assembled | Panels A–C complete (2P reruns & stats calculated in notebook) |
| **Fig S4** | `\label{sup:rsof}`<br>`figures/fig_supp_rsof.png` | Conjunctive coding of optic flow & running speed | [`figsupp_rsof.ipynb`](../v1_depth_map/figures/figsupp_rsof.ipynb) | `fig_supp_rsof.svg` | 🎨 In Progress | Panels A–J (model fit comparisons & hold-out predictions) |
| **Fig S5** | `\label{sup:openloop}`<br>`figures/fig_supp_openloop.png` | Optic flow and running speed tuning on open-loop trials | [`figsupp_openloop.ipynb`](../v1_depth_map/figures/figsupp_openloop.ipynb) | `fig_supp_openloop.svg` | 🎨 In Progress | Panels A–D (closed-loop vs open-loop tuning & speed correlations) |
| **Fig S6** | `\label{sup:v1map_uncorrected}`<br>`figures/fig_supp_v1map_uncorrected.png` | Distribution of depth preferences across visual field (uncorrected) | [`figsupp_rf.ipynb`](../v1_depth_map/figures/figsupp_rf.ipynb) | `rf_supp/pairwise_distance_all_sessions.svg` | 🎨 In Progress | Panels A–E (pairwise distance analysis & uncorrected RF gradients) |

---

### Additional / Candidate Figures (Referenced in Text or Under Development)

| Candidate Figure | Text Reference in `v1_depth_map.tex` | Description | Generating Notebook / Script | Status | Action Items |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Multi-day Stability** | Line 172: *(Figure 1O-P / Supp)* | Longitudinal tracking of depth selectivity across consecutive days | [`figsupp_multidays.ipynb`](../v1_depth_map/figures/figsupp_multidays.ipynb) | 🎨 In Progress | Single-cell exemplars tracked over multiple days |
| **Simulation Control** | Line 211: *"(FIG SUP)"* (tri-modal distribution absent in synthetic data) | Synthetic neural response model validation & simulation fitting | [`figsupp_simulation_control.ipynb`](../v1_depth_map/figures/figsupp_simulation_control.ipynb) | ✅ Ready | Simulation reruns completed (`decay_tau=2`, area-norm) |
| **Motorized Wheel & History Blur** | Line 207: *"(Figure SUP)"* (residual co-fluctuations along iso-depth lines) | Motorized wheel constant speed periods & stimulus history control | [`figure_rsof_integration.ipynb`](../v1_depth_map/figures/figure_rsof_integration.ipynb) / `treadmill.ipynb` | 🎨 In Progress | Motorized vs free locomotion comparison |

---

## 3. Vector Layout & Style Guidelines

- **Fonts**: Arial / Arial Narrow (embed TTF files from `v1_manuscript_figures/fonts/`).
- **Color Palette**: Use project standard colormaps for depth (e.g., Viridis / Turbo) and model comparisons.
- **Export Standards**:
  - Export vector graphics as `.svg` with editable text.
  - Final manuscript assembly compiled to vectorized multi-page `.pdf`.
