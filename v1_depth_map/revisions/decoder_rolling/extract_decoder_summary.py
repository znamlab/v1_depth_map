"""Extract a small per-session summary from the depth decoder `decoder_results.pickle` files.

Used to compare the previous production decoder (0.5 s rolling average of dF/F and running
speed before the 0.5 s downsampling; Oct 31 2026 files on BlackPasspo) with the Nov 16 2026
rerun without rolling average (cottage_analysis 9165c80; files on NEMO). Each file is about
1.2 GB; the summary keeps only accuracies, confusion matrices and best C.

For each session and loop type (closedloop / openloop) it records accuracy, confusion
matrix, best C per fold, accuracy and confusion matrix per speed bin (Still, 10 bins from
0.05 to 2 m/s, > 2 m/s), the number of test samples, and the checks from
notes/07_decoder_rerun_2026-11-16.md: accuracy == accuracy_score(y_test, y_preds), 5 best C,
speed-bin results present. Read-only on the decoder files.

Usage:
    python extract_decoder_summary.py --processed ROOT --sessions FILE --out OUT.pickle
FILE has one session name (MOUSE_SESSION) per line.
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

LOOPS = ("closedloop", "openloop")


def summarize_file(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    rows = []
    for loop in LOOPS:
        if f"accuracy_{loop}" not in d and f"y_test_all_{loop}" not in d:
            continue
        y_test = np.asarray(d.get(f"y_test_all_{loop}", []))
        y_pred = np.asarray(d.get(f"y_preds_all_{loop}", []))
        acc = d.get(f"accuracy_{loop}", np.nan)
        best_c = d.get(f"best_C_{loop}")
        if best_c is None:
            best_c = d.get(f"best_params_all_{loop}", {}).get("C")
        acc_bins = d.get(f"acc_speed_bins_{loop}")
        recomputed = (
            accuracy_score(y_test, y_pred)
            if len(y_test) and len(y_test) == len(y_pred)
            else np.nan
        )
        rows.append(
            dict(
                loop=loop,
                accuracy=float(acc) if acc is not None else np.nan,
                accuracy_recomputed=recomputed,
                n_test=len(y_test),
                conmat=np.asarray(d.get(f"conmat_{loop}", np.nan)),
                best_C=list(best_c) if best_c is not None else None,
                acc_speed_bins=(
                    np.asarray(acc_bins, float) if acc_bins is not None else None
                ),
                conmat_speed_bins=(
                    np.asarray(d[f"conmat_speed_bins_{loop}"], float)
                    if f"conmat_speed_bins_{loop}" in d
                    else None
                ),
                check_accuracy=bool(np.isclose(acc, recomputed))
                if acc is not None
                else False,
                check_5_best_C=best_c is not None and len(best_c) == 5,
                check_speed_bins=acc_bins is not None,
            )
        )
    params = d.get("params", {})
    return rows, {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--sessions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--filename", default="decoder_results.pickle")
    a = p.parse_args()

    sessions = [s.strip() for s in a.sessions.read_text().splitlines() if s.strip()]
    rows = []
    t0 = time.time()
    for i, session in enumerate(sessions):
        mouse, sess = session.split("_", 1)
        path = a.processed / mouse / sess / a.filename
        base = dict(session=session, path=str(path))
        if not path.exists():
            rows.append(dict(base, error="missing"))
            print(f"[{i + 1}/{len(sessions)}] {session}: missing", flush=True)
            continue
        base["mtime"] = pd.Timestamp(os.path.getmtime(path), unit="s")
        try:
            loop_rows, params = summarize_file(path)
        except Exception as err:  # truncated pickles etc.: record and go on
            rows.append(dict(base, error=repr(err)))
            print(f"[{i + 1}/{len(sessions)}] {session}: {err!r}", flush=True)
            continue
        for r in loop_rows:
            rows.append(dict(base, params=params, **r))
        print(
            f"[{i + 1}/{len(sessions)}] {session}: {[r['loop'] for r in loop_rows]} "
            f"{time.time() - t0:.0f} s",
            flush=True,
        )
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    df.to_pickle(tmp)
    tmp.rename(a.out)
    print(f"wrote {a.out}: {len(df)} rows")


if __name__ == "__main__":
    main()
