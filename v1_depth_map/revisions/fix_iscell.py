"""Post-reextraction iscell fix for annotated revision sessions.

`2p reextract` re-runs the built-in Suite2p classifier, which discards curated cells.
The curated masks contain only validated cells, so every ROI with a non-NaN trace should
be marked as a cell.

Backs each file up to ~/offset_check_work/backups/ before writing. Report-only unless
--apply is passed.
"""

import argparse
from pathlib import Path

import flexiznam as flz
import numpy as np

BACKUP_DIR = Path.home() / "offset_check_work" / "backups"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="colasa_3d-vision_revisions")
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument(
        "--apply", action="store_true", help="actually write iscell.npy"
    )
    args = parser.parse_args()

    flm = flz.get_flexilims_session(args.project)

    for sess in args.sessions:
        ds = flz.get_datasets(
            project_id=args.project,
            dataset_type="suite2p_rois",
            origin_name=sess,
            filter_datasets={"annotated": True},
            allow_multiple=False,
            flexilims_session=flm,
        )
        print(f"\n=== {sess}\n    {ds.path_full}")
        for iscellfile in sorted(Path(ds.path_full).rglob("iscell.npy")):
            f_path = iscellfile.parent / "F.npy"
            if not f_path.exists():
                print(
                    f"    {iscellfile.parent.name}: no F.npy next to iscell, skipping"
                )
                continue
            f_raw = np.load(f_path, mmap_mode="r")
            valid_cell = ~np.all(np.isnan(np.asarray(f_raw)), axis=1)
            iscell = np.load(iscellfile)
            before = int(iscell[:, 0].sum())
            after = int(valid_cell.sum())
            if before == after and np.array_equal(
                iscell[:, 0].astype(bool), valid_cell
            ):
                print(f"    {iscellfile.parent.name}: already correct ({after} cells)")
                continue
            print(
                f"    {iscellfile.parent.name}: {iscell.shape[0]} ROIs, "
                f"iscell {before} -> {after} cells (+{after - before})"
            )
            if not args.apply:
                continue
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            tag = "_".join(iscellfile.parts[-4:]).replace("/", "_")
            backup = BACKUP_DIR / f"{sess}_{tag}"
            if not backup.exists():
                np.save(backup, iscell)
                print(f"      backed up original to {backup}")
            iscell[:, 0] = valid_cell.astype(iscell.dtype)
            np.save(iscellfile, iscell)
            check = np.load(iscellfile)
            print(
                f"      WROTE -> iscell[:,0] sums to {int(check[:, 0].sum())} "
                f"(re-read from disk)"
            )


if __name__ == "__main__":
    main()
