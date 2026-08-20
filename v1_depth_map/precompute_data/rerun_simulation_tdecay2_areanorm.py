"""Re-run the RS/OF simulated-response pipeline with the area-normalized kernel
and decay_tau=2 (aligned with `figsupp_simulation_control.ipynb`'s TDECAY).

`cottage_analysis.analysis.spheres.simulation.make_biexponential_kernel` (and
`make_exponential_kernel`) take a `normalization` flag: "area" normalizes the kernel's
sum to 1 (unit gain, so a sustained input reaches the same steady state at the output),
"max" normalizes its peak to 1. "area" is the default and is what this script uses -- a
"max"-normalized run was tried and rejected because it inflates `fake_dff` amplitudes by
roughly the area-normalized kernel's sum-to-peak ratio (~37x at decay_tau=2 / 15 Hz),
which is not comparable to the real dF/F.

This script regenerates the precomputed
`simulated_responses_fit_{treadmill,spheres}_*.parquet` files (containing `fake_dff`)
for the "motor" (treadmill) sessions of the colasa_3d-vision_revisions project, via
`treadmill.simulate_and_fit_session` / `spheres.simulate_and_fit_session` (both
submitted to slurm, matching how they were originally generated in
`v1_depth_map/revisions/preprocess_rev_sessions.ipynb`, cells 26-27), using:
  - the default kernel normalization ("area"), and
  - decay_tau=2 instead of the previously-used decay_tau=4.

Note the output filename encodes only decay_tau/rise_tau/circularity, not the
normalization, so this overwrites whatever `_2_0.15_circular.parquet` is already there.

Note also that `simulate_and_fit_session` cuts trials with the *current* default
onset detector ("plateau"), whereas the April-2026 artifacts were cut with "model".
After running this, the `tread_kwargs=dict(method="model")` override in
`figsupp_simulation_control.ipynb` (cell 10) must be dropped, or its frame-count
assert will fire.
"""

import os
from pathlib import Path

import flexiznam as flz

from cottage_analysis.analysis import treadmill
from cottage_analysis.analysis.spheres import spheres
from v1_depth_map.revisions.revision_sessions import sessions

PROJECT = "colasa_3d-vision_revisions"
DECAY_TAU = 2  # matches figsupp_simulation_control.ipynb's TDECAY
RISE_TAU = 0.15
MAKE_CIRCULAR = True
KERNEL_NORMALIZATION = "area"  # the default; passed explicitly for clarity
USE_SLURM = True


def get_motor_sessions(flexilims_session):
    """Motor (treadmill) sessions that actually exist on flexilims for this project."""
    motor_sessions = []
    for session_name, protocol in sessions.items():
        if protocol != "motor":
            continue
        sess = flz.get_entity(name=session_name, project_id=PROJECT, datatype="session")
        if sess is None:
            print(f"Session {session_name} doesn't exist on flexilims, skipping.")
            continue
        motor_sessions.append(session_name)
    return motor_sessions


def make_slurm_folder(session_name):
    slurm_folder = Path(os.path.expanduser("~/slurm_logs"))
    slurm_folder.mkdir(exist_ok=True)
    slurm_folder = slurm_folder / session_name
    slurm_folder.mkdir(exist_ok=True)
    return slurm_folder


def main():
    flexilims_session = flz.get_flexilims_session(project_id=PROJECT)
    motor_sessions = get_motor_sessions(flexilims_session)
    print(
        f"Re-running simulation for {len(motor_sessions)} motor sessions: "
        f"{motor_sessions}"
    )

    for session_name in motor_sessions:
        slurm_folder = make_slurm_folder(session_name)
        circ_sfx = "circular" if MAKE_CIRCULAR else "elliptical"

        # 1. Motorised wheel (treadmill) -- produces
        # simulated_responses_fit_treadmill_{DECAY_TAU}_{RISE_TAU}_{circ_sfx}.parquet
        treadmill.simulate_and_fit_session(
            session_name,
            decay_tau=DECAY_TAU,
            rise_tau=RISE_TAU,
            make_circular=MAKE_CIRCULAR,
            kernel_normalization=KERNEL_NORMALIZATION,
            project=PROJECT,
            use_slurm=USE_SLURM,
            slurm_folder=slurm_folder,
            filter_datasets={"anatomical_only": 3, "annotated": True},
            scripts_name=f"simul_tdecay{DECAY_TAU}_areanorm_{session_name}_{circ_sfx}_treadmill",
        )

        # 2. Free locomotion (spheres) -- produces
        # simulated_responses_fit_spheres_{DECAY_TAU}_{RISE_TAU}_{circ_sfx}.parquet
        spheres.simulate_and_fit_session(
            session_name,
            decay_tau=DECAY_TAU,
            rise_tau=RISE_TAU,
            make_circular=MAKE_CIRCULAR,
            kernel_normalization=KERNEL_NORMALIZATION,
            project=PROJECT,
            use_slurm=USE_SLURM,
            slurm_folder=slurm_folder,
            filter_datasets={"annotated": True},
            scripts_name=f"simul_tdecay{DECAY_TAU}_areanorm_{session_name}_{circ_sfx}_spheres",
        )

    print(
        f"\nSubmitted simulation re-run (decay_tau={DECAY_TAU}, kernel_normalization="
        f"'{KERNEL_NORMALIZATION}') for all motor sessions. Check slurm logs under "
        "~/slurm_logs/<session_name>/ for job status."
    )


if __name__ == "__main__":
    main()
