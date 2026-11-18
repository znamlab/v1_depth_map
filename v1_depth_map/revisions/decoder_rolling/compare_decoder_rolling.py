"""Compare depth decoder results with and without the 0.5 s rolling average (per session).

Inputs are the summaries written by extract_decoder_summary.py:
- previous production (Oct 31 2026, 0.5 s rolling average then 0.5 s downsampling, Still
  bin = methods 1 s rule), from BlackPasspo;
- Nov 16 2026 rerun (cottage_analysis 9165c80: no rolling average, same downsampling, same
  C grid), from NEMO.

A session is compared only if its new file was written by the rerun (mtime after
RERUN_START), it has no decoder job left on NEMO (`--unfinished` list), and both loop types
pass the checks of notes/07 (accuracy matches y_test/y_preds, 5 best C, speed bins).

Writes one figure per session plus a population summary and a CSV to --out-dir.

Usage:
    python compare_decoder_rolling.py --old OLD.pickle --new NEW.pickle \
        --unfinished FILE --out-dir DIR
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RERUN_START = pd.Timestamp("2026-11-16 17:52")  # first orchestrator submission (UTC)
LOOPS = ("closedloop", "openloop")
BIN_LABELS = (
    ["Still"]
    + [
        f"{lo:g}-{hi:g}"
        for lo, hi in zip(
            [0.05, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8],
            [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],
        )
    ]
    + [">2"]
)
OLD_COLOR, NEW_COLOR = "0.45", "tab:red"
MIN_BIN_N = 20  # speed bins with fewer test samples are not plotted / not in the median


def bin_counts(conmats):
    """Number of test samples in each speed bin, from its confusion matrix."""
    if conmats is None:
        return None
    return np.array([np.nansum(c) for c in np.asarray(conmats, float)])


def masked_bins(acc, conmats):
    acc = np.asarray(acc, float).copy()
    n = bin_counts(conmats)
    if n is not None:
        acc[n < MIN_BIN_N] = np.nan
    return acc


def fmt_c(cs):
    return "[" + ", ".join(f"{float(c):g}" for c in cs) + "]" if cs is not None else "?"


def valid_new(rows):
    return (
        len(rows) > 0
        and rows.get("error", pd.Series([np.nan] * len(rows))).isna().all()
        and (rows.mtime > RERUN_START).all()
        and rows.check_accuracy.all()
        and rows.check_5_best_C.all()
        and rows.check_speed_bins.all()
    )


def norm_conmat(c):
    c = np.asarray(c, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return c / c.sum(axis=1, keepdims=True)


def plot_session(session, old, new, out_dir):
    loops = [lp for lp in LOOPS if lp in set(old.loop) and lp in set(new.loop)]
    fig, axes = plt.subplots(
        len(loops),
        4,
        figsize=(15, 3.6 * len(loops)),
        squeeze=False,
        gridspec_kw=dict(width_ratios=[1.8, 1, 1, 1]),
    )
    for i, loop in enumerate(loops):
        o = old[old.loop == loop].iloc[0]
        n = new[new.loop == loop].iloc[0]
        ax = axes[i, 0]
        x = np.arange(len(BIN_LABELS))
        for r, color, lab in [
            (o, OLD_COLOR, "rolling 0.5 s"),
            (n, NEW_COLOR, "no rolling"),
        ]:
            ax.plot(
                x,
                masked_bins(r.acc_speed_bins, r.conmat_speed_bins),
                "o-",
                color=color,
                ms=3,
                label=f"{lab}",
            )
            ax.axhline(r.accuracy, color=color, ls="--", lw=1)
        n_old, n_new = bin_counts(o.conmat_speed_bins), bin_counts(n.conmat_speed_bins)
        for xi, (a, b) in enumerate(zip(n_old, n_new)):
            ax.text(
                xi,
                0.97,
                f"{a:.0f}\n{b:.0f}",
                ha="center",
                va="top",
                fontsize=5,
                color="k" if min(a, b) >= MIN_BIN_N else "0.6",
            )
        n_depths = np.asarray(o.conmat).shape[0]
        ax.axhline(1 / n_depths, color="k", ls=":", lw=0.8)
        ax.set_xticks(x, BIN_LABELS, rotation=60, fontsize=7)
        ax.set_xlabel("running speed bin (m/s)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{loop}: overall {o.accuracy:.3f} -> {n.accuracy:.3f} (dashed; dotted = chance)\n"
            f"n per bin rolling / no rolling at top; bins with n < {MIN_BIN_N} not drawn",
            fontsize=7,
        )
        ax.legend(fontsize=7, loc="upper right")
        co, cn = norm_conmat(o.conmat), norm_conmat(n.conmat)
        for ax, c, title, cmap, lim in [
            (axes[i, 1], co, f"rolling\nbest C {fmt_c(o.best_C)}", "viridis", (0, 1)),
            (
                axes[i, 2],
                cn,
                f"no rolling\nbest C {fmt_c(n.best_C)}",
                "viridis",
                (0, 1),
            ),
            (
                axes[i, 3],
                cn - co,
                "difference (no rolling - rolling)",
                "RdBu_r",
                (-0.3, 0.3),
            ),
        ]:
            im = ax.imshow(c, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(title, fontsize=7)
            ax.set_xlabel("decoded depth #", fontsize=7)
            ax.set_ylabel("true depth #", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{session}  (n test samples: "
        + ", ".join(
            f"{lp} {old[old.loop == lp].n_test.iloc[0]} -> {new[new.loop == lp].n_test.iloc[0]}"
            for lp in loops
        )
        + ")",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{session}.png", dpi=120)
    plt.close(fig)


def plot_population(comp, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, loop in enumerate(LOOPS):
        c = comp[comp.loop == loop]
        if c.empty:
            continue
        ax = axes[0, i]
        ax.scatter(c.acc_old, c.acc_new, s=12, color="k")
        lim = [0, max(c.acc_old.max(), c.acc_new.max()) * 1.05]
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xlabel("accuracy, rolling 0.5 s")
        ax.set_ylabel("accuracy, no rolling")
        d = c.acc_new - c.acc_old
        ax.set_title(
            f"{loop}: n={len(c)}, median diff {d.median():+.3f} "
            f"[IQR {d.quantile(.25):+.3f}, {d.quantile(.75):+.3f}]",
            fontsize=8,
        )
        ax = axes[1, i]
        diffs = np.vstack([nb - ob for ob, nb in zip(c.bins_old, c.bins_new)])
        x = np.arange(len(BIN_LABELS))
        for row in diffs:
            ax.plot(x, row, color="0.8", lw=0.5)
        ax.plot(x, np.nanmedian(diffs, 0), "o-", color=NEW_COLOR, label="median")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, BIN_LABELS, rotation=60, fontsize=7)
        ax.set_xlabel("running speed bin (m/s)")
        ax.set_ylabel("accuracy, no rolling - rolling")
        ax.set_title(
            f"per session (grey); bins with n < {MIN_BIN_N} in either run excluded",
            fontsize=8,
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "population_summary.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--unfinished", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    old, new = pd.read_pickle(a.old), pd.read_pickle(a.new)
    unfinished = set(a.unfinished.read_text().split())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "sessions").mkdir(exist_ok=True)

    status, comp = [], []
    for session in old.session.unique():
        o, n = old[old.session == session], new[new.session == session]
        if session in unfinished:
            status.append((session, "decoder jobs still running"))
            continue
        if not valid_new(n):
            status.append((session, "new results missing, stale or failing checks"))
            continue
        if not (
            o.get("error", pd.Series([np.nan] * len(o))).isna().all()
            and o.check_accuracy.all()
        ):
            status.append((session, "old results failing checks"))
            continue
        plot_session(session, o, n, a.out_dir / "sessions")
        status.append((session, "compared"))
        for loop in LOOPS:
            if loop in set(o.loop) and loop in set(n.loop):
                ro, rn = o[o.loop == loop].iloc[0], n[n.loop == loop].iloc[0]
                comp.append(
                    dict(
                        session=session,
                        loop=loop,
                        acc_old=ro.accuracy,
                        acc_new=rn.accuracy,
                        n_test_old=ro.n_test,
                        n_test_new=rn.n_test,
                        best_C_old=fmt_c(ro.best_C),
                        best_C_new=fmt_c(rn.best_C),
                        bins_old=masked_bins(ro.acc_speed_bins, ro.conmat_speed_bins)
                        + 0 * masked_bins(rn.acc_speed_bins, rn.conmat_speed_bins),
                        bins_new=masked_bins(rn.acc_speed_bins, rn.conmat_speed_bins)
                        + 0 * masked_bins(ro.acc_speed_bins, ro.conmat_speed_bins),
                        **{
                            f"still_{k}": r.acc_speed_bins[0]
                            for k, r in [("old", ro), ("new", rn)]
                        },
                    )
                )
    comp = pd.DataFrame(comp)
    pd.DataFrame(status, columns=["session", "status"]).to_csv(
        a.out_dir / "status.csv", index=False
    )
    comp.drop(columns=["bins_old", "bins_new"]).to_csv(
        a.out_dir / "comparison.csv", index=False
    )
    comp.to_pickle(a.out_dir / "comparison.pickle")
    plot_population(comp, a.out_dir)
    print(pd.DataFrame(status, columns=["session", "status"]).status.value_counts())
    for loop in LOOPS:
        c = comp[comp.loop == loop]
        if len(c):
            d = c.acc_new - c.acc_old
            ds = c.still_new - c.still_old
            print(
                f"{loop}: n={len(c)}  old median {c.acc_old.median():.3f}  new median "
                f"{c.acc_new.median():.3f}  diff median {d.median():+.3f} "
                f"(new > old in {(d > 0).sum()}/{len(c)}); Still diff median {ds.median():+.3f}"
            )


if __name__ == "__main__":
    main()
