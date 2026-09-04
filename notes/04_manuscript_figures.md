# 04 - Manuscript Figures Assembly Tracking

This document tracks the assembly status of all Main and Supplementary figures for the manuscript, including individual panel generation from notebooks, vector assembly (in Adobe Illustrator / Inkscape), and final PDF/SVG compilation.

---

## 1. Main Figures Assembly Status (`v1_depth_map.tex`)

*Current active version directory: `v1_manuscript_figures/ver_rev1/`.*

| Figure | LaTeX Label & File | Description | Generating Notebook(s) | Exported Panel Files | Assembly Status | Tasks & Action Items |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Figure 1** | `\label{fig:intro}`<br>`figures/fig1.pdf` | Depth selectivity in mouse V1 (Panels A–P) | [`figure_depth_selectivity.ipynb`](../v1_depth_map/figures/figure_depth_selectivity.ipynb) | `fig1.svg`<br>`fig1.pdf`<br>`fig1_4examples.svg` | ✅ Ready / Assembled | Complete 16-panel assembly (A–P) generated directly from notebook |
| **Figure 2** | `\label{fig:rsof}`<br>`figures/fig2.pdf` | Depth representation vs speed & flow (RS × OF) (Panels A–K) | [`figure_rsof_integration.ipynb`](../v1_depth_map/figures/figure_rsof_integration.ipynb) | `fig2.svg`<br>`fig2.pdf` | ✅ Ready / Assembled | Complete 11-panel assembly (A–K) generated directly from notebook |
| **Figure 3** | `\label{fig:depth_cells}`<br>`figures/fig3.pdf` | Depth cells exemplars & single-cell tuning profiles | [`figure_depth_cells.ipynb`](../v1_depth_map/figures/figure_depth_cells.ipynb) | `fig_depth_cells.svg`<br>`fig_depth_cells_examples.svg`<br>`fig_depth_cells_examples_overlaid.svg` | 🎨 In Progress | - Refine single-cell exemplars, FOVs, and tuning overlays |
| **Figure 4** | `\label{fig:rf}`<br>`figures/fig4.png` | Receptive fields & 3D tuning | [`figure_rf.ipynb`](../v1_depth_map/figures/figure_rf.ipynb) | `fig_3d_rfs.pdf`<br>`fig_rf_example_fov.svg`<br>`fig_rf_examples.pdf`<br>`rf_position_corrected.svg` | 🎨 In Progress | - Confirm retinotopic map overlay scaling<br>- Review contralateral vs ipsilateral ROIs |
| **Figure 5** | `\label{fig:v1map}`<br>`figures/fig5.pdf` | Depth map across V1 / visual space (Panels A–F) | [`figure_rf.ipynb`](../v1_depth_map/figures/figure_rf.ipynb) | `fig5.svg` | ✅ Ready / Assembled | Complete 6-panel assembly (A–F) generated directly from notebook |

---

## 2. Supplementary Figures Assembly Status

### Integrated in `v1_depth_map.tex`

| Figure | LaTeX Label & File | Description | Generating Notebook | Exported Files in `ver_rev1` | Assembly Status | Details & Action Items |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Fig S1** | `\label{sup:vis_stim}`<br>`figures/fig_supp_vis-stim.png` | Visual stimuli in VR (Schematic, Photodiode sync, Position, Lag histogram) | [`figsupp_vis_stim_sync.ipynb`](../v1_depth_map/figures/figsupp_vis_stim_sync.ipynb) | `fig_supp_vis-stim.svg`<br>`fig_supp_vis-stim.pdf` | ✅ Ready / Assembled | Panels A–D 100% vector (embedded vector schematic, no duplicate axes) |
| **Fig S2** | `\label{sup:speeds}`<br>`figures/fig_supp_speeds.png` | Running, optic flow speeds and eye movements | [`figsupp_speed.ipynb`](../v1_depth_map/figures/figsupp_speed.ipynb) | `figsupp_speed.svg` | ✅ Ready / Assembled | Panels A–L complete (full frame eye image, unclipped labels) |
| **Fig S3** | `\label{sup:size}`<br>`figures/fig_supp_size_control.png` | Virtual depth selectivity invariance to stimulus size | [`figsupp_size_control.ipynb`](../v1_depth_map/figures/figsupp_size_control.ipynb) | `fig_size_control/`<br>`fig_size_tuning.svg` | ✅ Ready / Assembled | Panels A–C complete (2P reruns & stats calculated in notebook) |
| **Fig S4** | `\label{sup:rsof}`<br>`figures/fig_supp_rsof.png` | Conjunctive coding of optic flow & running speed | [`figsupp_rsof.ipynb`](../v1_depth_map/figures/figsupp_rsof.ipynb) | `fig_supp_rsof.svg` | 🎨 In Progress | Panels A–O. A–L: three example neurons (near / middle / far - ROIs 261 / 402 / 86 of `PZAH8.2h_S20230116`), each with depth, RS and OF tuning (letters 1-2) and the measured RS/OF matrix plus the five model fits (letters 3-4). M–N: preferred OF and preferred RS vs preferred depth. O: proportion of neurons best fit by each model per response-amplitude bin |
| **Fig S5** | `\label{sup:openloop}`<br>`figures/fig_supp_openloop.png` | Optic flow and running speed tuning on open-loop trials | [`figsupp_openloop.ipynb`](../v1_depth_map/figures/figsupp_openloop.ipynb) | `fig_supp_openloop.svg` | 🎨 In Progress | Panels A–D (closed-loop vs open-loop tuning & speed correlations) |
| **Fig S6** | `\label{sup:v1map_uncorrected}`<br>`figures/fig_supp_v1map_uncorrected.png` | Distribution of depth preferences across visual field (uncorrected) | [`figsupp_rf.ipynb`](../v1_depth_map/figures/figsupp_rf.ipynb) | `rf_supp/pairwise_distance_all_sessions.svg` (A–C)<br>`rf_supp/v1map_uncorrected.svg` (D–E) | 🎨 In Progress | Panels A–E (pairwise distance analysis & uncorrected RF gradients). D–E mirror Fig 5E/F on `preferred_depth_closedloop` instead of `preferred_depth_corrected` |

| **Fig S7** | `\label{sup:simulation}`<br>`figures/fig_supp_simulation.png` | Simulation control: synthetic neural response validation & absence of tri-modal distribution | [`figsupp_simulation_control.ipynb`](../v1_depth_map/figures/figsupp_simulation_control.ipynb) | `fig_supp_simulation_control.svg` | ✅ Ready / Assembled | Simulation reruns completed (`decay_tau=2`, area-norm) |
| **Supp Fig** | `\label{sup:prop_depth}` | Depth selectivity of the V1 population: three further example neurons (near / middle / far), proportion of depth-tuned neurons across sessions, and preferred depth for the 5- vs 8-depth cohorts | [`figsupp3_depth_pop.ipynb`](../v1_depth_map/figures/figsupp3_depth_pop.ipynb) | `figsupp3_depth_pop.svg` | ✅ Ready / Assembled | Panels A-L generated directly from the notebook (A-I examples, J session histogram, K/L cohort preferred depth on a shared y axis, dashed lines at the presented depths). Examples are ROIs 261 / 638 / 742 of `PZAH8.2h_S20230116` - the best-fit depth-tuned neuron in each depth band, excluding the ROIs already shown in Fig. 1F-I. Population panels reuse `fig1/neurons_df_all.pickle` cached by `figure1_depth_selectivity.ipynb` |
| **Supp Fig** | `\label{sup:prop_sig_rf}` | Distribution of proportion of cells with significant receptive fields | [`figure_rf.ipynb`](../v1_depth_map/figures/figure_rf.ipynb) / [`figsupp_single_depth_receptive_fields.ipynb`](../v1_depth_map/figures/figsupp_single_depth_receptive_fields.ipynb) | `prop_sig_rf_hist.svg` | 🎨 In Progress | Session-wise histograms & ipsi vs contra significant RF proportions |
| **Supp Fig** | `\label{sup:multidepth_rf}` | Multi-depth vs. single-depth receptive fields and preferred depth consistency | [`figsupp_multidepth_receptive_fields.ipynb`](../v1_depth_map/figures/figsupp_multidepth_receptive_fields.ipynb) | `fig_supp_multidepth_rf/` | 🎨 In Progress | Single-depth vs multi-depth 2D RF comparisons, correlation distributions & depth consistency |
| **Supp Fig** | `\label{sup:best_model_amp}` | Fraction / proportion of best RS × OF model fit per dF/F amplitude bin | [`figsupp_rsof.ipynb`](../v1_depth_map/figures/figsupp_rsof.ipynb) / [`figure_rsof_integration.ipynb`](../v1_depth_map/figures/figure_rsof_integration.ipynb) | panel O of `fig_supp_rsof.svg` | 🎨 In Progress | Evaluate whether best model distribution varies with response magnitude / signal-to-noise. Now assembled inside Fig S4 (panel O) rather than a standalone file; `figure_rsof_integration.ipynb` still builds the same stacked bar on its own for quick inspection |

---

## 3. Vector Layout & Style Guidelines

- **Fonts**: Arial / Arial Narrow (embed TTF files from `v1_manuscript_figures/fonts/`).
- **Color Palette**: Use project standard colormaps for depth (e.g., Viridis / Turbo) and model comparisons.
- **Export Standards**:
  - Export vector graphics as `.svg` with editable text.
  - Final manuscript assembly compiled to vectorized multi-page `.pdf`.

### 3.1 Figure style convention (enforced across all `figures/` notebooks)

Everything below lives in `cottage_analysis.plotting.style`. Do not re-implement any
of it in a notebook.

**Setup** - one call, as the notebook's second cell. It replaces the old
`arial_font_path` boilerplate, and no notebook should set `pdf.fonttype`,
`svg.fonttype`, `font.family` or `mathtext.default` itself:

```python
from cottage_analysis.plotting import style
from cottage_analysis.plotting.style import CM, FONTSIZE_DICT

style.setup_figure_fonts()
```

It registers **every** Arial face from the fonts directory (regular, **bold**, italic,
plus Arial Narrow), applies the rcParams, and sets `font.family = "Arial"` - a single
family name, which is what Illustrator resolves most reliably. Registering the bold
face matters: `findfont` does not fail when asked for a weight it cannot supply, it
silently returns the closest match, so registering `arial.ttf` alone made
`fontweight="bold"` produce non-bold panel letters on any machine without a system
Arial Bold (NEMO, most Linux runners). The fonts directory is looked up in
`style.FONT_SEARCH_DIRS` (external drive, then NEMO); a missing one warns rather than
raising, so an unmounted drive cannot break a run.

**Font sizes** - the canonical dict is `style.FONTSIZE_DICT`:

| key | pt | used for |
| :--- | :---: | :--- |
| `panel` | 10 | panel letters (always bold) |
| `title` | 7 | axes titles |
| `label` | 7 | axis labels |
| `tick` | 5 | tick labels |
| `legend` | 5 | legend text |

Use it as-is. A figure that genuinely needs to deviate must say so as an explicit
merge, so the deviation stays reviewable - never re-type the whole dict:

```python
fontsize_dict = FONTSIZE_DICT | {"title": 8}   # and say why, in a comment
```

Note that an override only reaches text drawn by a `fontsize_dict`-aware plotting
helper. Anything drawn by plain matplotlib in the same figure still takes its size
from the rcParams, so an override can make a figure internally inconsistent rather
than uniformly larger - check which helper actually reads the key before adding one.

**Current state: there are no overrides.** Every notebook and helper resolves to the
shared dict. The ones that used to exist were all removed as unjustified: `tick: 6`
(7 sites across the RF notebooks), `title: 8` (3 sites), `legend: 5.5` (1 site) and
`title: 5` (1 site). The last of those was dead code - it only reached
`plot_RS_OF_matrix`, which was called without a `title=` argument, so it set the size
of an empty string.

**Panel letters and cm layout** - use the shared helpers, so panel letters cannot
drift from `FONTSIZE_DICT["panel"]`:

```python
from cottage_analysis.plotting.style import rect_cm, panel_letter

ax = fig.add_axes(rect_cm(fig, x_cm, y_cm, w_cm, h_cm))
panel_letter(fig, "A", 0.1, 5.8)   # bold, at FONTSIZE_DICT["panel"]
```

**Saving** - `style.savefig` is the only permitted save path, and the figure is always
passed explicitly (a bare `plt.gcf()` is brittle when notebook cells are re-run out of
order):

```python
style.savefig(SAVE_ROOT / "fig.svg", fig=fig, bbox_inches="tight", dpi=300)
```

For `.svg` it rewrites matplotlib's CSS `font` shorthand
(`font: 700 10px 'Arial'`) into longhand
(`font-family: 'Arial'; font-size: 10px; font-weight: 700`). Illustrator parses the
two-token `<size> <family>` form but discards the three-token
`<weight> <size> <family>` form and falls back to 12 pt - so **bold text opens at
12 pt regardless of the real size**, which in practice means the panel letters. Other
formats pass straight through. The call is idempotent, and `style.fix_svg_fonts(path)`
applies the same fix to an SVG written by other means (as `figsupp_vis_stim_sync`
does after splicing in its vector schematic with `ElementTree`).

Only matplotlib < 3.10 emits the shorthand; 3.10 writes longhand itself. The figure
kernel is currently on 3.9.2, so the fix is load-bearing.

**Checking an export:**

```bash
grep -l 'style="font:' *.svg                      # must return nothing
grep -o "font-family: '[^']*'" fig.svg | sort -u  # must be 'Arial' only
grep -o "font-weight: [0-9]*" fig.svg | sort -u   # 700 present for panel letters
```
