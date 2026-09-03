"""Regenerate the hey2 session table for notes/01_preprocessing.md.

The hand-maintained table in §4 drifted from the list the batch scripts actually run:
40 figure sessions were missing from it and 24 rows marked "Fine" were not figure
sessions. This rebuilds it from the same `get_sessions()` call that
`batch_analysis/batch_analysis/analyze_all_sessions.py` uses, so the two cannot disagree.

Manual annotations in the existing table (exclusion reasons, sanity-plot ticks) are
parsed out and carried into the Notes column -- they are not derivable and must survive.

Metadata + file stats only, no bulk array reads, so this is fine on a login node.

Usage:
    conda activate cottage_analysis
    python gen_session_table.py notes/01_preprocessing.md > new_table.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

import flexiznam as flz
import pandas as pd
from cottage_analysis.summary_analysis import get_session_list

PROJECT = "hey2_3d-vision_foodres_20220101"
FIGURE_MICE = [
    "PZAH6.4b",
    "PZAG3.4f",
    "PZAH8.2h",
    "PZAH8.2i",
    "PZAH8.2f",
    "PZAH10.2d",
    "PZAH10.2f",
]
# the filter the figure notebooks pass to load_session()
TRACE_FILTER = {"anatomical_only": 3, "ast_neuropil": False}
# protocols sync_all_recordings() iterates over
LOADED_PROTOCOL = "SpheresPermTubeReward"

# Superseded annotations to drop or replace when carrying the old table forward.
# The "needs ast:False run" flag was wrong: those sessions are excluded by
# trialnum_min=10, so no notebook loads them and nothing needs re-running.
NOTE_REWRITES = {
    "⛔ **Needs ast:False run** (see §2.1)": "no ast:False traces — excluded by trialnum_min",
    "❌ Needs ast:False run": "no ast:False traces — excluded by trialnum_min",
}

ROW = re.compile(
    r"\|\s*`([A-Za-z0-9.]+_S\d+)`\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|"
)
SESSION_LITERAL = re.compile(r"\bPZA[A-Z]?\d+\.\d+[a-z]?_S\d{8}\d*\b")
NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "figures"


def sessions_named_in_notebooks(notebook_dir=NOTEBOOK_DIR):
    """Sessions hard-coded in any figure notebook, as {session: [notebooks]}.

    Some sessions are loaded by a notebook but are not in `get_sessions()` -- the three
    size-control sessions are the standing example. Scanning for literals keeps them in
    the table on their own merit, instead of surviving only as leftover manual notes.
    """
    found = {}
    for path in sorted(Path(notebook_dir).glob("*.ipynb")):
        try:
            nb = json.loads(path.read_text())
        except Exception:
            continue
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for name in SESSION_LITERAL.findall("".join(cell.get("source", []))):
                found.setdefault(name, set()).add(path.stem)
    return {k: sorted(v) for k, v in found.items()}


# Column headers whose content is hand-written and must survive a regeneration, and
# those that this script recomputes and must NOT be carried forward.
CARRY_HEADERS = {"status / exclude reason", "notes", "sanity plot checked"}
DERIVED_HEADERS = {
    "in figure list",
    "suite2p rois",
    "ast:false available",
    "suite2p traces",
    "ast_neuropil=false",
    "mouse",
    "recs",
    "2p recs",
}
DROP_VALUES = {"", "✅ fine", "— no", "✅ yes", "❌ no", "-", "[ ]"}


def _cells(line):
    """Split a markdown table row into stripped cell strings."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_manual_notes(notes_path):
    """Carry forward hand-written columns, keyed by session name.

    Reads the column headers rather than assuming positions, so this stays correct when
    run against either the original hand-written table (Status / Sanity Plot Checked) or
    a table this script previously generated (Notes). Without that, regenerating from
    generated output silently drops every annotation and copies computed columns into
    Notes instead -- the generator has to be idempotent.
    """
    manual = {}
    headers = None
    for line in Path(notes_path).read_text().splitlines():
        cells = _cells(line) if line.lstrip().startswith("|") else None
        if cells and any(h.lower() in ("session name", "session") for h in cells[:1]):
            headers = [c.lower() for c in cells]
            continue
        if cells and set("".join(cells)) <= set(":- "):
            continue  # separator row
        m = ROW.match(line)
        if not m or headers is None:
            continue
        session = m.group(1)
        bits = []
        for header, value in zip(headers, cells):
            if header in DERIVED_HEADERS or header not in CARRY_HEADERS:
                continue
            value = NOTE_REWRITES.get(value, value).strip()
            if value.lower() in DROP_VALUES:
                continue
            if header == "sanity plot checked":
                if not value.startswith("[x]"):
                    continue
                value = f"sanity: {value[3:].strip() or 'checked'}"
            bits.append(value)
        if bits:
            # de-duplicate while preserving order, in case a note was already merged in
            seen, uniq = set(), []
            for b in bits:
                if b not in seen:
                    seen.add(b)
                    uniq.append(b)
            manual[session] = "; ".join(uniq)
    return manual


def loadable_recordings(session_id, flm):
    """Recordings sync_all_recordings() would iterate for this session."""
    recs = flz.get_entities(
        datatype="recording",
        origin_id=session_id,
        query_key="recording_type",
        query_value="two_photon",
        flexilims_session=flm,
    )
    if recs is None or (isinstance(recs, pd.DataFrame) and recs.empty):
        return pd.DataFrame(), 0
    total = len(recs)
    sel = recs[recs.name.str.contains(LOADED_PROTOCOL)]
    sel = sel[~sel.name.str.contains("multidepth")]
    if "exclude_reason" in sel.columns:
        sel = sel[sel["exclude_reason"].isna()]
    return sel, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes", help="path to 01_preprocessing.md (for manual notes)")
    parser.add_argument(
        "--notebook-dir",
        default=str(NOTEBOOK_DIR),
        help="directory of figure notebooks to scan for hard-coded session names "
        f"(default: {NOTEBOOK_DIR})",
    )
    args = parser.parse_args()
    if not Path(args.notebook_dir).is_dir():
        parser.error(f"notebook dir not found: {args.notebook_dir}")

    manual = parse_manual_notes(args.notes)
    flm = flz.get_flexilims_session(PROJECT)

    mice = flz.get_entities("mouse", flexilims_session=flm)
    mice = mice[mice.name.isin(FIGURE_MICE)]
    figure_list = set(
        get_session_list.get_sessions(
            flm,
            exclude_openloop=False,
            exclude_pure_closedloop=False,
            v1_only=True,
            trialnum_min=10,
            mouse_list=mice,
        )
    )
    print(f"figure list: {len(figure_list)} sessions", file=sys.stderr)

    sessions = flz.get_entities("session", flexilims_session=flm)
    nb_sessions = sessions_named_in_notebooks(args.notebook_dir)
    print(
        f"sessions hard-coded in figure notebooks: {len(nb_sessions)} "
        f"({len(set(nb_sessions) - figure_list)} of them outside get_sessions())",
        file=sys.stderr,
    )
    # every session a notebook can load: the figure list, sessions named directly in a
    # notebook, and anything the previous table already tracked
    wanted = sorted(figure_list | set(nb_sessions) | set(manual))

    rows = []
    for name in wanted:
        ent = sessions[sessions.name == name]
        if ent.empty:
            continue
        ent = ent.iloc[0]
        sel, total = loadable_recordings(ent["id"], flm)
        n_ok = 0
        for rec_name in sel.name:
            try:
                ds = flz.get_datasets(
                    flexilims_session=flm,
                    origin_name=rec_name,
                    dataset_type="suite2p_traces",
                    filter_datasets=TRACE_FILTER,
                    allow_multiple=False,
                    return_dataseries=False,
                )
            except Exception:
                ds = None
            if ds is not None and (Path(ds.path_full) / "plane0" / "dff.npy").exists():
                n_ok += 1
        rois = flz.get_datasets(
            project_id=PROJECT,
            dataset_type="suite2p_rois",
            origin_name=name,
            filter_datasets={"anatomical_only": 3},
            allow_multiple=False,
            flexilims_session=flm,
        )
        note = manual.get(name, "")
        if name in nb_sessions and name not in figure_list:
            ref = "loaded by " + ", ".join(f"`{n}`" for n in nb_sessions[name])
            if ref not in note:
                note = f"{note}; {ref}" if note else ref
        rows.append(
            dict(
                session=name,
                mouse=name.split("_")[0],
                recs=total,
                in_figures="✅ Yes" if name in figure_list else "— No",
                traces="✅ Yes" if rois is not None else "❌ No",
                astfalse=(f"{n_ok}/{len(sel)}" if len(sel) else "—"),
                notes=note,
            )
        )

    print(
        "*Generated by "
        "[`gen_session_table.py`](../v1_depth_map/revisions/gen_session_table.py) — "
        "`python gen_session_table.py notes/01_preprocessing.md`. "
        "**In figure list** is `get_sessions(v1_only=True, trialnum_min=10)`, the same call "
        "`analyze_all_sessions.py` uses. **2P recs** counts two-photon recordings — the "
        "old hand-written table counted behavioural ones, so the numbers differ. "
        "**ast:False available** counts loadable `SpheresPermTubeReward` recordings that "
        "resolve to a real `dff.npy`.*"
    )
    print()
    print(
        "| Session Name | Mouse | 2P recs | In figure list | Suite2p ROIs | ast:False available | Notes |"
    )
    print("| :--- | :--- | ---: | :---: | :---: | :---: | :--- |")
    for r in sorted(rows, key=lambda r: (r["mouse"], r["session"])):
        print(
            f"| `{r['session']}` | {r['mouse']} | {r['recs']} | {r['in_figures']} | "
            f"{r['traces']} | {r['astfalse']} | {r['notes']} |"
        )

    n_fig = sum(1 for r in rows if r["in_figures"].startswith("✅"))
    absent = sorted(set(nb_sessions) - {r["session"] for r in rows})
    print(f"\nrows: {len(rows)}, flagged in figure list: {n_fig}", file=sys.stderr)
    if absent:
        print(
            f"notebook-referenced but not in project {PROJECT} "
            f"(expected for colasa sessions, which live in their own table): {absent}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
