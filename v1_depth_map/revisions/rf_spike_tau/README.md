# Deconvolution tau and RF lag (revisions, Oct 2026)

- `tau_estimation.ipynb` — how the deconvolution tau was chosen (1.1 s GCaMP6s / PZAG, 0.25 s GCaMP6f / PZAH): decay of
  isolated transients per session. Runs from `data/` only.
- `tau_isolated_transients.py` (+ `.sbatch`) — per-cell tau of isolated transients → `data/isolated_transients/`.
- `tau_diagnostics.py` — RF-free checks of a tau (reconvolution residual and spike autocorrelation).
- `latency_sweep.py`, `single_depth_lag_sweep.py` — RF R² vs stimulus lag.
- `refit_rf_variant.py` — multidepth RF refits with stimulus model variants (snapshot / calcium kernel).

The re-deconvolution itself is `precompute_data/deconvolve_tau.py`.
