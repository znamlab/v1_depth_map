# 05 - Manuscript Statistics Lineage & Pipeline

This document describes how statistics and sample sizes are generated, tracked, and propagated into the LaTeX manuscript (`v1_depth_map.tex`). It serves as the canonical reference for agents and developers modifying figure notebooks or updating text/captions.

---

## 1. Core Philosophy & Design Rules

1. **No External Extractors / No Logic Duplication**:
   Statistics are generated **inside the Jupyter figure notebooks** that create the plots. An external extraction script re-running analysis logic is strictly forbidden because changes to thresholds, outlier criteria, or session subsets in a notebook would cause text numbers to silently drift out of sync with figure panels.
2. **Notebooks Own Their Numbers, By Reference**:
   Every number cited in the text or captions must be bound to the exact runtime variables used to generate the corresponding axes, histograms, scatter plots, or model fits. An export cell **reads** variables the panel cells already computed; it never re-runs a test. Re-running a bootstrap with a different `n_boots`, or substituting a different test, reintroduces exactly the drift this pipeline exists to prevent.
3. **One YAML Per Notebook**:
   The emission unit is the *notebook*, not the figure. A figure can have several producing notebooks (Fig. 2 panels A-E come from `figure2_rsof_integration.ipynb`, F-L from `figure_openloop.ipynb`), and keying the file by figure made them overwrite each other. Each file records which notebook wrote it, when, and whether the numbers came from a notebook run or were entered by hand.
4. **Pure Aggregator Compiler**:
   The compiler script ([`build_manuscript_stats.py`](../v1_depth_map/stats/build_manuscript_stats.py)) has zero domain or statistical logic. It reads the YAML files emitted by the notebooks, validates them, formats a human-readable audit dashboard ([`MANUSCRIPT_STATS.md`](../v1_depth_map/stats/MANUSCRIPT_STATS.md)), and compiles standard LaTeX macros ([`manuscript_stats.tex`](../v1_depth_map/stats/manuscript_stats.tex)).
5. **Journal Compatibility & Co-Author Friendliness**:
   The LaTeX definitions use standard `\providecommand` and `\renewcommand` declarations. They require **zero external LaTeX packages**, ensuring 100% compatibility with *Science* editorial conversion tools and Overleaf collaboration. The compiler enforces this (see §2, Step 2).

---

## 2. End-to-End Data Flow

```mermaid
flowchart TD
    subgraph Data["1. Data Caches & Sessions"]
        RAW["Processed 2P Caches\n(/Volumes/BlackPasspo/v1_depth_map/processed/...)"]
    end

    subgraph Notebooks["2. Figure Notebooks (Jupyter)"]
        N1["figure1_depth_selectivity.ipynb"]
        N2["figure2_rsof_integration.ipynb"]
        N3["figure_openloop.ipynb"]
        N4["figure_depth_cells.ipynb"]
        N5["figure_rf.ipynb"]
        NS["figsupp1_vis_stim_sync.ipynb\nfigsupp2_speed.ipynb\nfigsupp4_size_control.ipynb"]
        RAW --> N1 & N2 & N3 & N4 & N5 & NS

        N1 -->|"export_figure_stats(..., figure='fig1')"| Y1["stats_figure1_depth_selectivity.yaml"]
        N2 -->|"export_figure_stats(..., figure='fig2')"| Y2["stats_figure2_rsof_integration.yaml"]
        N3 -->|"export_figure_stats(..., figure='fig2')"| Y3["stats_figure_openloop.yaml"]
        N4 -->|"export_figure_stats(..., figure='fig3')"| Y4["stats_figure_depth_cells.yaml"]
        N5 -->|"export_figure_stats(..., figure='fig4_5')"| Y5["stats_figure_rf.yaml"]
        NS -->|"export_figure_stats(..., figure='supp')"| YS["stats_figsupp*.yaml"]
    end

    subgraph StatsDir["3. Per-Notebook YAML Files"]
        Y1 & Y2 & Y3 & Y4 & Y5 & YS
    end

    subgraph Compiler["4. Central Aggregator"]
        BUILD["v1_depth_map/stats/build_manuscript_stats.py\n(validate -> group by figure -> render)"]
        Y1 & Y2 & Y3 & Y4 & Y5 & YS --> BUILD
        BUILD --> MD["MANUSCRIPT_STATS.md\n(Human-Readable Audit Dashboard)"]
        BUILD --> TEX["v1_depth_map/stats/manuscript_stats.tex\n(LaTeX Macro Definitions)"]
    end

    subgraph Overleaf["5. Manuscript (Overleaf)"]
        TEX -->|"Copied by hand, once checked"| OL_TEX["Overleaf/manuscript_stats.tex"]
        OL_TEX -->|\\input{manuscript_stats.tex}| MANUSCRIPT["v1_depth_map.tex\n(Cites \\statMacro{})"]
    end
```

---

## 3. Detailed Step-by-Step Architecture

### Step 1: Emitting Statistics from Figure Notebooks

At the end of a figure notebook (immediately following figure assembly, vector SVG export, and statistical printouts), the notebook constructs a dictionary of metrics and calls `export_figure_stats()`:

```python
from v1_depth_map.stats import export_figure_stats

stats_data = {
    "<metric_key>": {
        "raw": <numeric_or_primitive_value>,
        "formatted": "<publication_ready_string>",
        "latex_macro": "<MacroNameWithoutBackslash>",
        "description": "<Contextual note on where this appears in text/caption>",
    },
    ...
}
export_figure_stats("<notebook_stem>", stats_data, figure="<figure_key>")
```

- **First argument**: the notebook's own file stem. It decides the YAML file name, so each notebook owns exactly one file and cannot clobber another's.
- **`figure`**: the manuscript figure the metrics belong to (`fig1`, `fig2`, `fig3`, `fig4_5`, `supp`). It only decides how the metrics are grouped in the dashboard and in the `.tex`.

#### Field Schema
- `raw`: The raw unrounded number (integer, float) for programmatic comparison.
- `formatted`: The exact string to render in LaTeX prose (including thousands commas, math delimiters for $\pm$ or $\%$, and scientific notation where needed).
- `latex_macro`: The camel-case macro name (e.g., `statFigOneTotalNeurons`). **Letters only** - a LaTeX control word cannot contain digits or underscores.
- `description`: Plain-English explanation indicating the figure panel or manuscript line where the statistic is used.

#### On-Disk Format

Each file in `v1_depth_map/stats/generated/` carries a `_meta` block written by the hook:

```yaml
_meta:
  notebook: figure_openloop
  figure: fig2
  generated_at: '2026-09-19T14:03:11'
  source: notebook        # 'manual' if the numbers were typed in by hand
metrics:
  openloop_n_neurons:
    raw: 1229
    formatted: 1,229
    latex_macro: statFigTwoOpenLoopNeurons
    description: Depth-selective neurons compared between closed and open loop (Fig 2H-J)
```

`source: manual` marks numbers that have **not** come from a notebook run. They compile normally but are counted and flagged in the dashboard and in the `.tex` header, so nobody mistakes a hand-transcribed value for a notebook-derived one. A notebook run overwrites its own file and clears the flag.

#### Notebook Ownership

| Notebook | `figure` | Macro prefix | Covers |
| :--- | :--- | :--- | :--- |
| `figure1_depth_selectivity.ipynb` | `fig1` | `statFigOne*` | Depth selectivity population, 5- vs 8-depth cohort splits, multiday tracking |
| `figure2_rsof_integration.ipynb` | `fig2` | `statFigTwoModel*`, `statFigTwoPval*` | RS/OF model comparison, all 10 pairwise bootstrap p-values (Fig 2A-E) |
| `figure_openloop.ipynb` | `fig2` | `statFigTwo*` | Closed vs open loop, bootstrap correlations, decoder sessions (Fig 2F-L) |
| `figure_depth_cells.ipynb` | `fig3` | `statFigThree*` | Motorized wheel, elongation ratios, axial von Mises mixture |
| `figure_rf.ipynb` | `fig4_5` | `statFigFour*`, `statFigFive*` | 3D receptive field counts, visual space gradient p-values, retinotopic correlations |
| `figsupp1_vis_stim_sync.ipynb` | `supp` | `statSuppDisplay*`, `statSuppFrame*` | Photodiode sync lag and display frame rates |
| `figsupp2_speed.ipynb` | `supp` | `statSuppEye*` | Pupil / gaze tracking sessions and mice |
| `figsupp4_size_control.ipynb` | `supp` | `statSuppSizeControl*` | Stimulus size invariance |

---

### Step 2: Compiling Markdown & LaTeX Outputs

The compiler [`v1_depth_map/stats/build_manuscript_stats.py`](../v1_depth_map/stats/build_manuscript_stats.py) aggregates every YAML file in `v1_depth_map/stats/generated/`, **validates them, and writes nothing if validation fails**:

- a `latex_macro` defined by two different notebooks,
- a `formatted` value with an odd number of unescaped `$` (math mode left open),
- a macro name that is not letters-only,
- a `formatted` value using a command that needs an external package (`\text` → use `\mathrm`, `\bm` → `\mathbf`, `\SI`/`\num`, `\degree` → `^\circ`, `\nicefrac` → `\frac`).

Warnings (which do not block) cover missing descriptions, hand-entered numbers, and YAML files older than the notebook that owns them.

It then generates two artifacts:

1. **`MANUSCRIPT_STATS.md`**: A central human-readable audit table grouped by figure, with a `Source` column per metric and a banner counting hand-entered values. Authors and reviewers can inspect all numbers without opening LaTeX or notebook files.
2. **`manuscript_stats.tex`**: Publication-ready LaTeX definitions:
   ```latex
   % Total layer 2/3 excitatory neurons recorded
   \providecommand{\statFigOneTotalNeurons}{59,937}
   \renewcommand{\statFigOneTotalNeurons}{59,937}

   % Percentage of layer 2/3 neurons with depth selectivity
   \providecommand{\statFigOnePctDepthNeurons}{41.3\%}
   \renewcommand{\statFigOnePctDepthNeurons}{41.3\%}
   ```
   - Using both `\providecommand` and `\renewcommand` ensures definitions can be safely included in multiple sub-files without collisions.
   - **Written to the repo only.** The compiler does not touch Overleaf. Once you have checked the dashboard (in particular that no value is still flagged `manual`), copy the file across by hand:
     ```bash
     cp v1_depth_map/stats/manuscript_stats.tex \
        "/Users/blota/Library/CloudStorage/Dropbox-TheFrancisCrick/Antonin Blot/Apps/Overleaf/v1_depth_map Science re-submission/manuscript_stats.tex"
     ```
     Keeping this manual means a routine figure re-run can never push half-checked numbers into a manuscript co-authors are editing.

---

### Step 3: Referencing in the Manuscript (`v1_depth_map.tex`)

In `v1_depth_map.tex`:
1. **Include definitions** in the preamble (under custom commands):
   ```latex
   \input{manuscript_stats.tex}
   ```
2. **Cite macros in text and captions** using empty braces (`{}`) to preserve trailing whitespace:
   ```latex
   Across the population, \statFigOnePctDepthNeurons{} of cells
   (\statFigOneDepthNeurons{} of \statFigOneTotalNeurons{} neurons) exhibited
   significant depth selectivity...
   ```
   ```latex
   \caption{\textbf{(J)} Distribution of preferred virtual depths of depth-selective
   neurons (N = \statFigOneDepthNeurons{} neurons)...}
   ```

> **Status:** as of 2026-09-19 the manuscript does **not** cite any `\stat...` macro yet - every number in `v1_depth_map.tex` is still a literal. Introducing them is a separate, explicitly-requested task (see §5).

---

## 4. Pipeline Integration & Execution

### Automated Pipeline Run
When running the full figures pipeline:
```bash
python run_figures_pipeline.py
```
1. Each notebook executes and writes its updated `stats_<notebook>.yaml`.
2. At the conclusion of all notebook runs, `run_figures_pipeline.py` automatically calls `compile_manuscript_stats()`.
3. `MANUSCRIPT_STATS.md` and `manuscript_stats.tex` are refreshed **in the repo**. If validation fails, nothing is written and the pipeline prints a warning naming each offending metric.
4. Copying `manuscript_stats.tex` to Overleaf stays manual - see Step 2.

### Manual / Fast Compilation
To re-compile the Markdown dashboard and LaTeX macros from existing YAML outputs (sub-second runtime, zero dependencies on 2P data or external drives):
```bash
python -m v1_depth_map.stats.build_manuscript_stats
```

To validate only, without writing anything (exits non-zero on failure):
```bash
python -m v1_depth_map.stats.build_manuscript_stats --check
```

### Tests
```bash
pytest v1_depth_map/stats/tests/
```
Three suites, all of which run in seconds with no 2P data:
- `test_build_manuscript_stats.py` - the compiler's validation and grouping rules.
- `test_export_cells.py` - executes every notebook's export cell against stub dataframes, so a broken export is caught in a second instead of after a multi-hour notebook run.
- `test_latex_smoke.py` - typesets every generated macro with a real `pdflatex` run and **no packages loaded**, which is the ground truth for design rule 5. Skipped if `pdflatex` is absent.

---

## 5. Guide for Future Agents & Developers

### Adding a New Statistic
When adding a new analysis or panel to an existing figure:
1. Open the corresponding figure notebook (e.g., `figure_rf.ipynb`).
2. Find the cell that already computes the statistic for the panel, and make sure the value is held in a named variable that survives to the export cell.
3. Add the metric to the `stats_<name>` dictionary before `export_figure_stats()`, reading that variable:
   ```python
   stats_rf["my_new_metric"] = {
       "raw": float(my_val),
       "formatted": f"{my_val:.2f}",
       "latex_macro": "statFigFiveMyNewMetric",
       "description": "Short explanation of the metric and where it appears",
   }
   ```
4. Run the notebook (or `python -m v1_depth_map.stats.build_manuscript_stats` if the YAML was already updated).
5. The macro `\statFigFiveMyNewMetric{}` is immediately available in LaTeX.

### Adding a New Producing Notebook
1. Call `export_figure_stats("<notebook_stem>", stats, figure="<figure_key>")` at the end of the cell that computes the numbers.
2. Add a row to the ownership table in §3.
3. Add its stub entry to `_stubs()` and `OWNERS` in `v1_depth_map/stats/tests/test_export_cells.py`.

### Rules of Thumb & Common Pitfalls
- **Never recompute**: read the variable the panel used. If two analyses in a notebook bind to the same name (`r`, `distribution`, `pval` is a common pattern with `hierarchical_bootstrap_stats`), rename them (`r_rs`/`r_of`, `pval_rs`/`pval_of`) so the export cell cannot silently pick up the last one.
- **`hierarchical_bootstrap_stats` returns an array**: `r` has one entry per element of `xcol`. `f"{r:.3f}"` raises `TypeError: unsupported format string passed to numpy.ndarray.__format__` - index it first, `float(r[0])`.
- **Formatting**: Always wrap mathematical expressions (e.g., $\pm$, $p$-values, and units) with dollar signs (`$`) in the `"formatted"` string (e.g., `"$26.6 \pm 1.4~\mathrm{ms}$"` or `"$p < 0.0001$"`), so they render without errors regardless of whether the macro is placed in LaTeX text mode or math mode. Include the whole expression: `" < 0.0001$"` is missing its opening `$p` and leaves math mode open.
- **No package-dependent commands**: use `\mathrm`, not `\text` (which needs `amsmath`). The compiler rejects the ones listed in §2.
- **Scope Limit**: Do NOT automate static experimental constants (such as the 930 nm laser wavelength, 4 mm cranial window diameter, or 10 s trial timeout). Only automate dynamic sample sizes ($N$), percentages, model fit fractions, correlations, and test statistics that depend on data filtering or analysis parameters.
- **Never Modify `v1_depth_map.tex` without Explicit User Instruction**: Step 3 (macro replacement in the text) should only be executed upon specific request from the user to prevent merge conflicts with active Overleaf edits by co-authors.
