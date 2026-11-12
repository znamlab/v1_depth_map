# Motorized-wheel RS/OF orientation controls (Fig 3K–M)

Reviewer-only controls for the tuning-ellipse orientation distribution on the motorized wheel
(Fig 3K–M). They are not manuscript figures. Their outputs are saved under
`<processed>/v1_manuscript_figures/ver_rev1/` as `fig_reviewer_*`.

| Notebook | Fit script | Question |
| --- | --- | --- |
| `fig_reviewer_treadmill_halves.ipynb` | `precompute_data/fit_treadmill_halves.py` (+ `.sbatch`, NEMO) | Do residual responses to the acceleration ramp drive the orientations? Each trial's plateau window is cut into its first and second half and refit. |
| `fig_reviewer_simulation_recovery.ipynb` | `precompute_data/fit_revision_simulation_recovery.py` (local) | Can the fit recover a known orientation? This is a positive control that complements the circular null of Fig S7D. |

The data are on BlackPasspo. The halves fits are in
`processed/colasa_3d-vision_revisions/analysis/treadmill_halves/`. The recovery fits are in the
per-session `simulated_recovery_fits_*.parquet` files next to `neurons_df`.

There are no stats exports, and it is still undecided whether these go in the reviewer response or the SI.
