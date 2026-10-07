# 04 - Manuscript Figures Assembly Tracking

This document tracks the assembly status of all Main and Supplementary figures for the manuscript, including individual panel generation from notebooks, vector assembly (in Adobe Illustrator / Inkscape), and final PDF/SVG compilation.

---

## 1. Main Figures Assembly Status (`v1_depth_map.tex`)

*Current active version directory: `v1_manuscript_figures/ver_rev1/`.*

| Figure | LaTeX Label & File | Description | Generating Notebook(s) | Exported Panel Files | Assembly Status | Tasks & Action Items |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Figure 1** | `\label{fig:intro}`<br>`figures/fig1.pdf` | Depth selectivity from motion parallax in L2/3 of mouse V1 (Panels A–P) | [`figure1_depth_selectivity.ipynb`](../v1_depth_map/figures/figure1_depth_selectivity.ipynb) | `fig1.svg`<br>`fig1.pdf` | ✅ Ready / Assembled | Complete 16-panel assembly (A–P) generated directly from notebook |
| **Figure 2** | `\label{fig:rsof}`<br>`figures/fig2.pdf` | Depth selectivity from conjunctive coding of optic flow & running speed (Panels A–L) | [`figure2_rsof_integration.ipynb`](../v1_depth_map/figures/figure2_rsof_integration.ipynb) (A–E)<br>[`figure2_openloop.ipynb`](../v1_depth_map/figures/figure2_openloop.ipynb) (F–L) | `fig2.svg`<br>`fig2.pdf`<br>`fig_openloop.svg`<br>`fig_openloop.pdf` | ✅ Ready / Assembled | Two producing notebooks; each owns its own `stats_*.yaml` |
| **Figure 3** | `\label{fig:depth_cells}`<br>`figures/fig3.pdf` | V1 neurons are selective for visuomotor gains (Panels A–L) | [`figure3_depth_cells.ipynb`](../v1_depth_map/figures/figure3_depth_cells.ipynb) | `fig_depth_cells.svg` | ✅ Ready / Assembled | Complete 12-panel assembly (A–L) generated directly from notebook |
| **Figure 4** | `\label{fig:rf}`<br>`figures/fig4.pdf` | Depth-selective neurons have three-dimensional receptive fields (Panels A–K) | [`figure4_receptive_fields.ipynb`](../v1_depth_map/figures/figure4_receptive_fields.ipynb) | `fig_receptive_fields_full.{svg,pdf,png}`<br>`fig_receptive_fields_examples.svg`<br>`fig_receptive_fields_3d_rfs_polished.{svg,pdf,png}`<br>`fig_receptive_fields_fov.svg`<br>`fig_receptive_fields_rf_distance.svg` | ✅ Ready / Assembled | Was Figure 5's companion in the previous version; the old Figure 5 (V1 depth map) is now Fig S9. Panels I–K (pairwise RF / depth distance vs distance between cells, hierarchical mouse/session bootstrap CI) moved here from Fig S9 A–C and read the `hey2_3d-vision_foodres_20220101` cache `rf_supp/pairwise_distance_all_sessions.pkl`. Exports the I–K counts (`\statFigFourRFDist*`: neurons, sessions, mice, pairs), spike RF fits only |

---

## 2. Supplementary Figures Assembly Status

### Integrated in `v1_depth_map.tex`

| Figure | LaTeX Label & File | Description | Generating Notebook | Exported Files in `ver_rev1` | Assembly Status | Details & Action Items |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Fig S1** | `\label{sup:vis_stim}`<br>`figures/fig_supp_vis-stim.png` | Visual stimuli in VR (schematic, photodiode sync, position, lag histogram) | [`figsupp1_vis_stim_sync.ipynb`](../v1_depth_map/figures/figsupp1_vis_stim_sync.ipynb) | `fig_supp_vis-stim.svg`<br>`fig_supp_vis-stim.pdf` | ✅ Ready / Assembled | Panels A–D 100% vector (embedded vector schematic, no duplicate axes) |
| **Fig S2** | `\label{sup:speeds}`<br>`figures/fig_supp_speeds.png` | Running, optic flow speeds and eye movements across virtual depths | [`figsupp2_speed.ipynb`](../v1_depth_map/figures/figsupp2_speed.ipynb) | `figsupp_speed.svg` | ✅ Ready / Assembled | Panels A–L complete (full frame eye image, unclipped labels) |
| **Fig S3** | `\label{sup:prop_depth}`<br>`figures/figsupp3_depth_pop.png` | Virtual depth tuning of individual V1 neurons and population distributions | [`figsupp3_depth_pop.ipynb`](../v1_depth_map/figures/figsupp3_depth_pop.ipynb) | `figsupp3_depth_pop.svg` | ✅ Ready / Assembled | Panels A–L (A–I examples, J session histogram, K/L cohort preferred depth). Examples are ROIs 261 / 638 / 742 of `PZAH8.2h_S20230116`. Population panels reuse `fig1/neurons_df_all.pickle` cached by `figure1_depth_selectivity.ipynb`. Emits no statistics |
| **Fig S4** | `\label{sup:size}`<br>`figures/fig_supp_size_control.png` | Virtual depth selectivity is invariant to stimulus size | [`figsupp4_size_control.ipynb`](../v1_depth_map/figures/figsupp4_size_control.ipynb) | `fig_size_control/fig_size_tuning.svg` | ✅ Ready / Assembled | Panels A–C complete (2P reruns & stats calculated in notebook) |
| **Fig S5** | `\label{sup:rsof}`<br>`figures/fig_supp_rsof.png` | Conjunctive coding of optic flow & running speed | [`figsupp5_rsof.ipynb`](../v1_depth_map/figures/figsupp5_rsof.ipynb) | `fig_supp_rsof.svg` | 🎨 In Progress | Panels A–O. A–L: three example neurons (ROIs 261 / 402 / 86 of `PZAH8.2h_S20230116`). M–N: preferred OF and preferred RS vs preferred depth. O: proportion of neurons best fit by each model per response-amplitude bin, built from `figure2_rsof_integration.ipynb`'s code |
| **Fig S6** | `\label{sup:openloop}`<br>`figures/fig_supp_openloop.png` | Optic flow and running speed tuning on open-loop trials | [`figsupp6_openloop.ipynb`](../v1_depth_map/figures/figsupp6_openloop.ipynb) | `fig_supp_openloop.svg` | 🎨 In Progress | Panels A–D (closed-loop vs open-loop tuning & speed correlations) |
| **Fig S7** | `\label{sup:simulation}`<br>`figures/fig_supp_simulation.png` | Motorized wheel controls for calcium indicator dynamics | [`figsupp7_simulation_control.ipynb`](../v1_depth_map/figures/figsupp7_simulation_control.ipynb) | `fig_supp_simulation_control.svg` | ✅ Ready / Assembled | Simulation reruns completed (`decay_tau=2`, area-norm) |
| **Fig S8** | `\label{sup:multidepth_rf}`<br>`figures/fig_supp_multidepth_rf.png` | Receptive fields mapped under single- and multi-depth conditions are similar | [`figsupp8_multidepth_receptive_fields.ipynb`](../v1_depth_map/figures/figsupp8_multidepth_receptive_fields.ipynb) | `fig_supp_multidepth_rf/figsupp_multidepth_receptive_fields_spks.{svg,pdf,png}` | 🎨 In Progress | Panels A–I. A–C: single- vs multi-depth RF examples. D: single/multi-depth RF correlation vs ipsilateral control. E: proportion of depth-tuned neurons with significant RFs per session, multi-depth sessions stacked on single-depth ones, black triangle = median of all sessions. F: single- vs multi-depth RF peak depth. G–I: example FOV retinotopy. Exports the E session counts and medians (`\statSuppEight*`), spike RF fits only |
| **Fig S9** | `\label{sup:v1map}`<br>`figures/fig_supp_v1map.png` | Distribution of depth preferences across the visual field | [`figsupp9_v1_depth_map.ipynb`](../v1_depth_map/figures/figsupp9_v1_depth_map.ipynb) (A–G)<br>[`figsupp9_rf_distance.ipynb`](../v1_depth_map/figures/figsupp9_rf_distance.ipynb) (H–I) | `fig5.svg` (A–G, name unchanged)<br>`rf_supp/v1map_uncorrected.svg` (H–I) | 🎨 In Progress | Was **Figure 5** before the Science re-submission. The former A–C (pairwise RF distance) are now Fig 4 I–K. All manuscript numbers come from `figsupp9_v1_depth_map.ipynb` as `\statSuppNine*` macros; `figsupp9_rf_distance.ipynb` emits none |

### Not in the current manuscript version

| Notebook | Description | Status |
| :--- | :--- | :--- |
| [`unused_single_depth_receptive_fields.ipynb`](../v1_depth_map/figures/unused_single_depth_receptive_fields.ipynb) | RFs and retinotopy under the single-depth protocol (`SpheresPermTubeReward`) | 💤 Unused - superseded by the single-vs-multi comparison in Fig S8. Kept for reference; skipped by `run_figures_pipeline.py` unless named with `--notebooks` |

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
applies the same fix to an SVG written by other means (as `figsupp1_vis_stim_sync`
does after splicing in its vector schematic with `ElementTree`).

Only matplotlib < 3.10 emits the shorthand; 3.10 writes longhand itself. The figure
kernel is currently on 3.9.2, so the fix is load-bearing.

**Checking an export:**

```bash
grep -l 'style="font:' *.svg                      # must return nothing
grep -o "font-family: '[^']*'" fig.svg | sort -u  # must be 'Arial' only
grep -o "font-weight: [0-9]*" fig.svg | sort -u   # 700 present for panel letters
```
