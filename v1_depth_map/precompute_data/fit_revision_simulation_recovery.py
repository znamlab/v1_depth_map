"""Recoverability of RS/OF tuning-ellipse orientation on the motorized wheel.

Positive control for Fig 3K-M, complementing the circular null of Fig S7: give every
neuron of the Fig 3K population an RS/OF Gaussian with a *known* orientation, simulate
its dF/F along the real RS/OF trajectory, add realistic noise, refit with exactly the
real trial-average plateau configuration, and compare recovered to true orientation.

Ground truth
------------
Each neuron keeps its own real trial-average plateau fit
(`rsof_popt_closedloop_g2d_treadmill_trial_average_plateau`): amplitude, centre, offset
and both sigmas, hence its real elongation. Only the orientation of the major axis is
replaced (`sigma_x` set to the major axis, `theta` to the target angle), under four
scenarios, plus the circular control of the paper (Fig S7D) for comparison:

    circular  major axis reduced to the minor one, as in Fig S7D (`true_angle` is NaN)
    ratio45   every neuron at 45 deg (ratio / depth tuning)
    cardinal  50% at 0 deg, 50% at 90 deg (random, seeded)
    uniform   uniform on [-45, 135) deg - the true -> recovered transfer function
    matched   drawn from the axial von Mises mixture fitted to the real Fig 3M angles
              (same selection, cut, k range and seed as figure3_depth_cells.ipynb)

Angles are drawn once (`ANGLE_SEED`) and are the same for every noise seed.

Noise
-----
The S7 simulation is noise-free; here `circular` with `noise_seed = -1` is that control
(same seed popts, kernel and fit configuration, k=1 only). Here each neuron gets its own residual added: real
dF/F minus the noise-free simulation of its real (uncircularised) fit, concatenated
over trials and circularly shifted by a random offset (10-90% of the trace length), so
its amplitude, autocorrelation and slow drift are kept while any stimulus locking is
broken. Offsets depend only on (noise seed, session, roi), so all scenarios share the
same noise for a given seed. `noise_seed = -1` is the noise-free fit.

The population is the Fig 3K one (significant trial-average g2d fit AND depth-tuned in
closed loop, 285 neurons over 4 sessions). Only those ROIs are simulated, and only the
k=1 fit is run (orientation and elongation come from it; no CV test R^2).

Output, per session beside `neurons_df.pickle` (one row per roi x scenario x seed):

    simulated_recovery_fits_treadmill_trial_average_plateau_2_0.15.parquet
    simulation_recovery_current.json   (provenance)

Typical use::

    python fit_revision_simulation_recovery.py --dry-run
    python fit_revision_simulation_recovery.py --sessions PZAH17.1e_S20250403 --n-noise 1
    python fit_revision_simulation_recovery.py            # all four, 10 noise seeds
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

from cottage_analysis.analysis import fit_gaussian_blob
from cottage_analysis.pipelines import pipeline_utils

from v1_depth_map.figure_utils import von_mises as vm
from v1_depth_map.figure_utils.treadmill import (
    add_trial_average_rsof_columns,
    load_treadmill_population_neurons_df,
)
from v1_depth_map.precompute_data.fit_revision_treadmill import (
    COMMON_PARAMS,
    FILTER_DATASETS,
    PHOTODIODE_PROTOCOL,
    PROJECT,
    SESSIONS,
    _cottage_analysis_sha,
    assert_site_root,
    target_config,
)

TARGET, METHOD = "treadmill", "plateau"
TA = "_treadmill_trial_average_plateau"
GROUNDTRUTH_COL = f"rsof_popt_closedloop_g2d{TA}"

# Same calcium model as fit_revision_simulation.py (the S7 control).
DECAY_TAU = 2
RISE_TAU = 0.15
KERNEL_NORMALIZATION = "area"

SCENARIOS = ("circular", "ratio45", "cardinal", "uniform", "matched")
ANGLE_SEED = 0
DEFAULT_N_NOISE = 10
SHIFT_RANGE = (0.1, 0.9)  # circular shift, as a fraction of the concatenated trace

# Fig 3M settings, for the `matched` scenario's mixture.
ELONGATION_CUTOFF = 1.4
VM_MAX_K = 5
VM_SEED = 42

PROVENANCE_FILENAME = "simulation_recovery_current.json"


def output_filename():
    return (
        f"simulated_recovery_fits_treadmill_trial_average_{METHOD}"
        f"_{DECAY_TAU}_{RISE_TAU}.parquet"
    )


def wrap_axial_deg(deg):
    """Wrap an axial angle to [-45, 135), matching fit_gb.get_gaussian_angle."""
    return (np.asarray(deg) + 45) % 180 - 45


# --------------------------------------------------------------------------------------
# Population and ground truth
# --------------------------------------------------------------------------------------
def select_population():
    """The Fig 3K population, selected exactly as in figure3_depth_cells.ipynb."""
    flexilims_session = flz.get_flexilims_session(PROJECT)
    ndf, *_ = load_treadmill_population_neurons_df(
        flexilims_session, load_simulated=False
    )
    add_trial_average_rsof_columns(
        ndf, ta_suffix=TA, null_method="empirical", verbose=False
    )
    sel = ndf[
        ndf[f"rsof_test_rsq_closedloop_g2d{TA}_sig"]
        & ndf["is_depth_neuron"].fillna(False)
    ].dropna(subset=[f"g2d_theta{TA}", f"g2d_semimajor{TA}", f"g2d_semiminor{TA}"])
    sel = sel[["roi_uid", "session", "roi", GROUNDTRUTH_COL, f"g2d_theta{TA}"]].copy()
    sel["real_elongation"] = ndf.loc[sel.index, f"g2d_semimajor{TA}"].astype(
        float
    ) / ndf.loc[sel.index, f"g2d_semiminor{TA}"].astype(float)
    sel = sel.rename(columns={f"g2d_theta{TA}": "real_angle"})
    return sel.sort_values(["session", "roi"]).reset_index(drop=True)


def fit_real_mixture(sel):
    """Axial von Mises mixture of the real Fig 3M angles (best k by BIC)."""
    angles = sel.loc[sel.real_elongation > ELONGATION_CUTOFF, "real_angle"].to_numpy(
        dtype=float
    )
    res = vm.model_selection(
        np.radians(angles), max_k=VM_MAX_K, seed=VM_SEED, verbose=False
    )
    k = min(res, key=lambda kk: res[kk]["bic"])
    r = res[k]
    return dict(
        k=int(k),
        pi_k=np.asarray(r["pi_k"], float),
        mu_k=np.asarray(r["mu_k"], float),  # radians, axial space
        kappa_k=np.asarray(r["kappa_k"], float),  # doubled-angle space
    )


def draw_angles(n, mixture, seed=ANGLE_SEED):
    """Target angles (deg, [-45, 135)) for every scenario, one per neuron."""
    rng = np.random.default_rng(seed)
    comp = rng.choice(len(mixture["pi_k"]), size=n, p=mixture["pi_k"])
    # von Mises on the doubled angle, halved back to the axial space.
    x2 = rng.vonmises(2 * mixture["mu_k"][comp], mixture["kappa_k"][comp])
    return {
        "ratio45": np.full(n, 45.0),
        "cardinal": rng.permutation(np.resize([0.0, 90.0], n)),
        "uniform": rng.uniform(-45, 135, size=n),
        "matched": wrap_axial_deg(np.degrees(x2 / 2)),
        # No orientation: rotate_popt circularises on NaN. Drawn last so the other
        # scenarios' draws are unchanged.
        "circular": np.full(n, np.nan),
    }


def rotate_popt(popt, angle_deg):
    """Same Gaussian with its major axis at `angle_deg`.

    A NaN angle gives the circular Gaussian instead, built exactly as
    `simulate_calcium_responses(make_circular=True)` does for Fig S7 (both log-variances
    set to the smaller one, theta unchanged).
    """
    out = np.asarray(popt, dtype=float).copy()
    if np.isnan(angle_deg):
        out[3] = out[4] = min(out[3], out[4])
        return out
    major, minor = max(out[3], out[4]), min(out[3], out[4])
    out[3], out[4] = major, minor
    out[5] = np.radians(angle_deg)
    return out


# --------------------------------------------------------------------------------------
# Simulation and fitting
# --------------------------------------------------------------------------------------
def simulate_session(session_name, sel_sess, cfg):
    """One load: real dF/F of the selected ROIs plus the noise-free simulations.

    The popt list holds len(sel_sess) * (1 + len(SCENARIOS)) Gaussians - block 0 the real
    fits (for the residual), then one block per scenario. `simulate_calcium_responses`
    returns one column per popt, independent of the ROI count, so this costs a few
    hundred columns rather than (1 + n_scenarios) x all ROIs.
    """
    popt_blocks = [list(sel_sess[GROUNDTRUTH_COL].map(np.asarray))]
    for scen in SCENARIOS:
        popt_blocks.append(
            [
                rotate_popt(p, a)
                for p, a in zip(sel_sess[GROUNDTRUTH_COL], sel_sess[f"angle_{scen}"])
            ]
        )
    popt_list = [p for block in popt_blocks for p in block]

    t0 = time.time()
    _, _, _, trials_df = pipeline_utils.load_session(
        project=PROJECT,
        session_name=session_name,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        regenerate_frames=False,
        filter_datasets=FILTER_DATASETS,
        protocol_base=cfg["protocol_base"],
        recording_type="two_photon",
        tread_kwargs={
            "method": METHOD,
            "sim_popt_list": popt_list,
            "sim_tau_decay": DECAY_TAU,
            "sim_tau_rise": RISE_TAU,
            "sim_make_circular": False,
            "sim_kernel_normalization": KERNEL_NORMALIZATION,
        },
    )
    print(f"### loaded and simulated in {(time.time() - t0) / 60:.1f} min")

    is_multidepth = trials_df.recording_name.str.contains("multidepth")
    assert not is_multidepth.any(), f"multidepth trials in {session_name}"
    trials_df = trials_df.reset_index(drop=True)
    assert "fake_dff_stim" in trials_df.columns, "sim_* kwargs did not reach the sync"

    rois = sel_sess.roi.to_numpy(dtype=int)
    n = len(rois)
    real = [d[:, rois] for d in trials_df.dff_stim]
    fake = trials_df.fake_dff_stim.tolist()
    for r, f in zip(real, fake):
        assert f.shape == (r.shape[0], n * (1 + len(SCENARIOS)))
    trials_df = trials_df.drop(columns=["dff_stim", "fake_dff_stim"])
    return trials_df, real, fake, n


def make_noise(real, fake, n, seed, session_idx, roi_ids):
    """Per-ROI residual, circularly shifted; list of per-trial (frames x n) arrays."""
    lengths = [r.shape[0] for r in real]
    resid = np.concatenate([r - f[:, :n] for r, f in zip(real, fake)], axis=0)
    # NaN only where the simulation skipped a frame (log RS/OF undefined); those frames
    # are not running frames and never enter the fit, but they must not carry a NaN
    # onto a running frame once shifted.
    resid = np.nan_to_num(resid, nan=0.0)
    total = resid.shape[0]
    shifted = np.empty_like(resid)
    for j, roi in enumerate(roi_ids):
        rng = np.random.default_rng([seed, session_idx, int(roi)])
        offset = int(
            rng.integers(int(SHIFT_RANGE[0] * total), int(SHIFT_RANGE[1] * total))
        )
        shifted[:, j] = np.roll(resid[:, j], offset)
    return np.split(shifted, np.cumsum(lengths)[:-1], axis=0)


def _fit_job(args):
    trials_df, dff_list, cfg, label = args
    trials_df = trials_df.copy()
    trials_df["dff_stim"] = dff_list
    t0 = time.time()
    fit_df = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df,
        model="gaussian_2d",
        choose_trials=None,
        trial_sfx="",
        k_folds=1,
        max_rs2motor_diff=cfg["max_rs2motor_diff"],
        trial_average=cfg["trial_average"],
        param_range=cfg["param_range"],
        **COMMON_PARAMS,
    )
    print(f"    {label}: {time.time() - t0:.0f} s", flush=True)
    return (
        fit_df["rsof_popt_closedloop_g2d"].tolist(),
        fit_df["rsof_rsq_closedloop_g2d"].tolist(),
    )


def run_session(session_name, session_idx, sel_sess, cfg, noise_seeds, n_workers):
    trials_df, real, fake, n = simulate_session(session_name, sel_sess, cfg)
    rois = sel_sess.roi.to_numpy(dtype=int)

    jobs, keys = [], []
    for seed in noise_seeds:
        noise = (
            [np.zeros((r.shape[0], n)) for r in real]
            if seed < 0
            else make_noise(real, fake, n, seed, session_idx, rois)
        )
        for i_scen, scen in enumerate(SCENARIOS):
            cols = slice(n * (1 + i_scen), n * (2 + i_scen))
            dff = [f[:, cols] + z for f, z in zip(fake, noise)]
            jobs.append((trials_df, dff, cfg, f"{scen} seed={seed}"))
            keys.append((scen, seed))

    print(f"### {len(jobs)} fits of {n} ROIs on {n_workers} workers")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        outputs = list(pool.map(_fit_job, jobs))

    rows = []
    for (scen, seed), (popts, rsqs) in zip(keys, outputs):
        for j in range(n):
            rows.append(
                dict(
                    roi=int(rois[j]),
                    roi_uid=sel_sess.roi_uid.iloc[j],
                    scenario=scen,
                    noise_seed=int(seed),
                    true_angle=float(sel_sess[f"angle_{scen}"].iloc[j]),
                    real_angle=float(sel_sess.real_angle.iloc[j]),
                    real_elongation=float(sel_sess.real_elongation.iloc[j]),
                    popt_groundtruth=rotate_popt(
                        sel_sess[GROUNDTRUTH_COL].iloc[j],
                        sel_sess[f"angle_{scen}"].iloc[j],
                    ),
                    popt_simulated=np.asarray(popts[j], dtype=float),
                    rsq_simulated=float(rsqs[j]),
                )
            )
    return pd.DataFrame(rows)


def write_provenance(session_dir, cfg, mixture, noise_seeds, n_rois):
    record = {
        "script": Path(__file__).name,
        "target": TARGET,
        "method": METHOD,
        "groundtruth_col": GROUNDTRUTH_COL,
        "scenarios": list(SCENARIOS),
        "angle_seed": ANGLE_SEED,
        "noise_seeds": [int(s) for s in noise_seeds],
        "shift_range": list(SHIFT_RANGE),
        "matched_mixture": {k: np.asarray(v).tolist() for k, v in mixture.items()},
        "n_rois": n_rois,
        "decay_tau": DECAY_TAU,
        "rise_tau": RISE_TAU,
        "kernel_normalization": KERNEL_NORMALIZATION,
        "param_range": cfg["param_range"],
        "trial_average": cfg["trial_average"],
        "max_rs2motor_diff": cfg["max_rs2motor_diff"],
        "min_sigma": COMMON_PARAMS["min_sigma"],
        "rs_thr": COMMON_PARAMS["rs_thr"],
        "niter": COMMON_PARAMS["niter"],
        "cottage_analysis_sha": _cottage_analysis_sha(),
    }
    path = Path(session_dir) / PROVENANCE_FILENAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def session_dir(session_name):
    flexilims_session = flz.get_flexilims_session(PROJECT)
    ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        allow_multiple=False,
    )
    return ds.path_full.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", nargs="+", default=SESSIONS)
    parser.add_argument("--n-noise", type=int, default=DEFAULT_N_NOISE)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--site", choices=("local", "nemo"), default="local")
    args = parser.parse_args()

    assert_site_root(args.site)
    cfg = target_config(TARGET, METHOD)
    noise_seeds = [-1] + list(range(args.n_noise))

    # Population and angles are drawn on all four sessions together, so the matched
    # mixture and the 50/50 cardinal split are population-level, whatever --sessions is.
    sel = select_population()
    mixture = fit_real_mixture(sel)
    for scen, angles in draw_angles(len(sel), mixture).items():
        sel[f"angle_{scen}"] = angles
    print(f"population: {len(sel)} neurons")
    print(sel.groupby("session").size().to_string())
    mu = wrap_axial_deg(np.degrees(mixture["mu_k"]))
    print(
        f"matched mixture: k={mixture['k']}, "
        + ", ".join(
            f"{m:.0f} deg ({100 * p:.0f}%)" for m, p in zip(mu, mixture["pi_k"])
        )
    )
    print(
        f"noise seeds: {noise_seeds}; {len(SCENARIOS) * len(noise_seeds)} fits/session"
    )

    t_start = time.time()
    for session_name in args.sessions:
        out_path = session_dir(session_name) / output_filename()
        if args.dry_run:
            state = "exists" if out_path.exists() else "would write"
            print(f"{session_name}: {state} {out_path}")
            continue
        if out_path.exists() and not args.redo:
            print(f"--- skip (exists): {out_path}")
            continue
        print(f"\n=== {session_name}")
        sel_sess = sel[sel.session == session_name].reset_index(drop=True)
        results = run_session(
            session_name,
            SESSIONS.index(session_name),
            sel_sess,
            cfg,
            noise_seeds,
            args.workers,
        )
        tmp = out_path.with_suffix(".parquet.partial")
        results.to_parquet(tmp, index=False)
        tmp.replace(out_path)
        write_provenance(out_path.parent, cfg, mixture, noise_seeds, len(sel_sess))
        print(f"### wrote {out_path} ({len(results)} rows)")
    print(f"\ntotal: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
