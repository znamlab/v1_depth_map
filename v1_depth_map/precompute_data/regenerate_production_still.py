"""Regenerate the production decoder speed bins with the methods stationary definition.

For each session, the production decoder_results.pickle is moved to BACKUP_ROOT, or
renamed in place with BACKUP_SUFFIX (same drive, so the original is kept untouched), and
rewritten with:
- `y_preds_all_*` rebuilt from the per-fold outputs (the stored ones are the zero
  placeholder), checked against the stored confusion matrix;
- `acc_speed_bins_*` / `conmat_speed_bins_*` recomputed by
  population_depth_decoder.find_acc_speed_bins, whose "Still" bin now uses
  find_stationary_samples (max raw running speed over the preceding 1 s < 5 cm/s).
No refit: accuracy and confusion matrices are unchanged (asserted). Rerunning skips
the move and starts again from the backup.

Run on 2026-10-31 on the 84 sessions in regenerate_production_still_sessions.txt (the
Fig. 1L-M decoder sessions), with cottage_analysis cb5e66e:
- locally: --fold-root <per-fold outputs copied from NEMO> --backup-root
  /Volumes/BlackPasspo/v1_depth_map/processed/backups/decoder_results_pre_still_fix_20261031
- on NEMO: regenerate_production_still.sbatch (run dir ver_rev1/still_regen/), originals
  kept as decoder_results.pickle.pre_still_fix_20261031_backup next to the new file.
Both gave identical stationary bins for all 120 runs (84 closed loop, 36 open loop).

    python regenerate_production_still.py MANIFEST --fold-root DIR \
        (--backup-root DIR | --backup-suffix SUFFIX) [--csv FILE]
writes a summary CSV of the old and new stationary bins (default BACKUP_ROOT/
still_regeneration.csv).
"""

import argparse
import pickle
import shutil
from pathlib import Path

import flexiznam as flz
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from cottage_analysis.analysis import population_depth_decoder as pdd

PROJECT = "hey2_3d-vision_foodres_20220101"
# as in cottage_analysis.pipelines.depth_decoder_pipeline
PARAMS = dict(
    continuous_still=1,
    still_time=1,
    still_thr=0.05,
    speed_bins=[0.05, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],
    k_folds=5,
    random_state=42,
)

DOWNSAMPLE_WINDOW = 0.5  # depth_decoder_pipeline params
STILL_DEFINITION = "max raw RS over preceding still_time s < still_thr (methods)"


def get_frame_rate(session_name, flexilims_session):
    ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        flexilims_session=flexilims_session,
        return_dataseries=False,
        filter_datasets={"anatomical_only": 3},
    )[0]
    return ds.extra_attributes["fs"]


def rebuild_predictions(decoder_dict, recording_type, fold_dir):
    """Production y_pred in sample order, from the per-fold outputs."""
    trials_df = decoder_dict[f"trials_df_{recording_type}"]
    y_test = decoder_dict[f"y_test_all_{recording_type}"]
    *_, test_frame_indices, _, _ = pdd.split_train_test_val(
        trials_df.copy(), k_folds=PARAMS["k_folds"], random_state=PARAMS["random_state"]
    )
    y_pred = np.full_like(y_test, -1)
    for i, idx in enumerate(test_frame_indices):
        with open(
            fold_dir / f"decoder_outputs_{recording_type}_fold{i}.pickle", "rb"
        ) as f:
            y_pred[idx] = pickle.load(f)["y_pred"]
    ndepths = len(trials_df.depth.unique())
    conmat = confusion_matrix(y_test, y_pred, labels=np.arange(ndepths))
    assert np.array_equal(
        conmat, decoder_dict[f"conmat_{recording_type}"]
    ), f"{recording_type}: fold outputs do not match the stored confusion matrix"
    return y_pred


def regenerate(
    session_name, fold_root, flexilims_session, backup_root=None, backup_suffix=None
):
    mouse, sess = session_name.split("_")
    prod_path = (
        flz.get_data_root("processed", flexilims_session=flexilims_session)
        / PROJECT
        / mouse
        / sess
        / "decoder_results.pickle"
    )
    if backup_suffix:
        backup_path = prod_path.with_name(prod_path.name + backup_suffix)
    else:
        backup_path = backup_root / mouse / sess / "decoder_results.pickle"
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(prod_path, backup_path)
    with open(backup_path, "rb") as f:
        decoder_dict = pickle.load(f)
    params = decoder_dict.get("params") or {}
    for key in ["rolling_window", "downsample_window"]:
        assert params.get(key, DOWNSAMPLE_WINDOW) == DOWNSAMPLE_WINDOW, (key, params)

    recording_types = [
        r for r in ["closedloop", "openloop"] if f"trials_df_{r}" in decoder_dict
    ]
    old = {}
    for r in recording_types:
        decoder_dict[f"y_preds_all_{r}"] = rebuild_predictions(
            decoder_dict, r, fold_root / mouse / sess
        )
        old[r] = {
            k: decoder_dict[f"{k}_{r}"]
            for k in ["accuracy", "conmat", "acc_speed_bins", "conmat_speed_bins"]
        }
    decoder_dict["still_definition"] = STILL_DEFINITION
    tmp_path = prod_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(decoder_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.rename(prod_path)

    frame_rate = get_frame_rate(session_name, flexilims_session)
    rows = []
    for r in recording_types:
        decoder_dict = pdd.find_acc_speed_bins(
            decoder_dict_path=prod_path,
            recording_type=f"_{r}",
            speed_bins=PARAMS["speed_bins"],
            continuous_still=PARAMS["continuous_still"],
            still_thr=PARAMS["still_thr"],
            still_time=PARAMS["still_time"],
            frame_rate=frame_rate,
            downsample_window=DOWNSAMPLE_WINDOW,
            use_slurm=False,
        )
        assert decoder_dict[f"accuracy_{r}"] == old[r]["accuracy"]
        assert np.array_equal(decoder_dict[f"conmat_{r}"], old[r]["conmat"])
        # running speed bins do not depend on the stationary definition
        np.testing.assert_array_equal(
            decoder_dict[f"acc_speed_bins_{r}"][1:], old[r]["acc_speed_bins"][1:]
        )
        new_still = decoder_dict[f"conmat_speed_bins_{r}"][0]
        old_still = old[r]["conmat_speed_bins"][0]
        rows.append(
            dict(
                session=session_name,
                recording_type=r,
                n_still_old=np.nansum(old_still),
                n_still_new=np.nansum(new_still),
                acc_still_old=old[r]["acc_speed_bins"][0],
                acc_still_new=decoder_dict[f"acc_speed_bins_{r}"][0],
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--fold-root", type=Path, required=True)
    backup = parser.add_mutually_exclusive_group(required=True)
    backup.add_argument("--backup-root", type=Path)
    backup.add_argument("--backup-suffix")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    csv = args.csv or args.backup_root / "still_regeneration.csv"
    flexilims_session = flz.get_flexilims_session(PROJECT)
    sessions = args.manifest.read_text().split()
    rows = []
    for i, session_name in enumerate(sessions):
        print(f"[{i + 1}/{len(sessions)}] {session_name}")
        rows += regenerate(
            session_name,
            args.fold_root,
            flexilims_session,
            backup_root=args.backup_root,
            backup_suffix=args.backup_suffix,
        )
        pd.DataFrame(rows).to_csv(csv, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("recording_type")[df.columns[2:]].median())


if __name__ == "__main__":
    main()
