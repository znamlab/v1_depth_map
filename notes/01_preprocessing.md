# 01 - Preprocessing Tracking

This document tracks raw 2-photon and behavioral preprocessing across all sessions, Suite2p motion correction/ROI extraction, calcium trace extraction without neuropil subtraction (`ast_neuropil = False`), optical offset estimation, and visual sanity plots.

> [!NOTE]
> **Preprocessing is complete — nothing here is outstanding.** Every session the manuscript figures load
> has `ast_neuropil = False` traces and a validated optical offset, and all sanity plots have been
> checked. This file is now a reference: how the extraction was done, and what state each session is in.
> Remaining work on this dataset is downstream fitting, tracked in
> [`02_processing_fits.md`](02_processing_fits.md).

---

## 1. Status & Settled Findings

- **Total Registered Sessions**: 183 sessions across 19 mice
- **Key Requirement**: For the latest analysis pipeline and paper revisions, all 2P sessions must be processed with `ast_neuropil = False` (`anatomical_only = 3`, `ast_neuropil = False`).
- **Optical Offset Fix**: Commit `42b8a0b` (2026-06-03) updated `estimate_offset()` in `calcium_utils.py:323-364` to pick the middle TIFF (`tiffs[len(tiffs) // 2]`) instead of the first frame.

**Settled:**
- Every recording the notebooks load resolves to a correct ast:False `dff.npy` — **390/390** hey2 and
  **41/41** colasa, verified with the exact filters the notebooks pass.
- The stale `ast_neuropil=True` labels are **harmless** — the loader reads the recording-level
  `suite2p_traces` flag, not the parent `suite2p_rois` flag that carries the stale value. 42 parent
  `suite2p_rois` datasets still carry `ast_neuropil = True`; nothing reads the flag.
- The offset fix changes revision-session offsets by **≲0.7%** — no catastrophic cases (§3.1).
- `PZAG16.3c_S20250401` has been re-extracted with post-fix offsets (Job `52280899`) and its `iscell`
  repaired to 639 curated cells.

---

## 2. 2P Extraction Procedures (reference)

How each project's traces were produced. Kept for reproducing a run, not as pending work.

### Execution Commands:

#### 1. Standard Sessions: dF/F Rerun without Neuropil:
```bash
# Sbatch script using run_dff_noneuropil.sh
2p calcium -p hey2_3d-vision_foodres_20220101 -s <SESSION_NAME> -c overwrite --no-run-suite2p --no-run-neuropil -t 0.7 --run-split
```

---

### 2.1 Revision Sessions (`colasa_3d-vision_revisions`): Why and How to Use `2p reextract`

> [!IMPORTANT]
> **Why `colasa` sessions require `2p reextract`**:
> Automated Suite2p segmentation and cell classification performed poorly on the revision datasets. For all `colasa_3d-vision_revisions` sessions, Suite2p was run **only for motion correction** and generating mean images (`meanImg.tif`, `meanImgE.tif`).
> ROI masks were generated outside Suite2p using **Cellpose** on `meanImgE` and manually curated in **Napari** (see [`preprocess_rev_sessions.ipynb`](../v1_depth_map/revisions/preprocess_rev_sessions.ipynb)).
> Consequently, traces must be extracted from the registered binaries (`data.bin`) using `2p reextract` with the curated mask PNG, creating the annotated dataset (`suite2p_rois_annotated_0`).

#### Mask File Location:
Curated mask PNGs are stored at:
```
/camp/lab/znamenskiyp/home/shared/projects/colasa_3d-vision_revisions/cellpose_data/curated_masks/{SESSION_NAME}_meanImgE_curated_masks.png
```
*(On local drive: `/Volumes/BlackPasspo/v1_depth_map/processed/colasa_3d-vision_revisions/cellpose_data/curated_masks/...`)*

#### How `2p reextract` Works:
1. Loads the curated mask image and maps non-zero pixels to ROI definitions.
2. Extracts raw traces (`F.npy`, `Fneu.npy`) from registered `data.bin` on `ga100` GPU partition.
3. Automatically runs `process_concatenated_traces`:
   - Calculates the middle-frame optical offsets (`estimate_offsets`).
   - Applies `correct_offset()` to trace arrays.
   - Detrends, calculates dF/F (`ast_neuropil=False`), and deconvolves spikes (`spks.npy`).
4. Splits recordings into `suite2p_traces_annotated` datasets and registers them in Flexilims.

#### Command Syntax:
```bash
# In conda env '2p-preprocess' (submits a Slurm job on ga100 GPU partition):
2p reextract \
  -s <SESSION_NAME> \
  -p colasa_3d-vision_revisions \
  -m /camp/lab/znamenskiyp/home/shared/projects/colasa_3d-vision_revisions/cellpose_data/curated_masks/<SESSION_NAME>_meanImgE_curated_masks.png \
  --conflicts overwrite
```

#### Post-Reextraction Step:
Because curated masks contain only validated cells, set all non-NaN ROIs to cells in `iscell.npy`:
```python
# (As performed in preprocess_rev_sessions.ipynb, cell 12)
suite2p_annotated = flz.get_datasets(
    project_id="colasa_3d-vision_revisions",
    dataset_type="suite2p_rois",
    origin_name=sess,
    filter_datasets={"annotated": True},
    allow_multiple=False,
)
for iscellfile in suite2p_annotated.path_full.rglob("iscell.npy"):
    f_raw = np.load(iscellfile.parent / "F.npy")
    valid_cell = ~np.all(np.isnan(f_raw), axis=1)
    iscell = np.load(iscellfile)
    iscell[:, 0] = valid_cell.astype(iscell.dtype)
    np.save(iscellfile, iscell)
```

> [!IMPORTANT]
> `2p reextract` **always** re-runs the Suite2p classifier and clobbers curated `iscell`. The post-step
> above is mandatory after every revision-session re-extraction, not optional. On
> `PZAG16.3c_S20250401` it silently dropped 181 of 639 curated cells.
> [`fix_iscell.py --sessions <names> --apply`](../v1_depth_map/revisions/fix_iscell.py) does it and is
> idempotent.

---

## 3. Optical Offset & Sanity Plots QC

Optical offsets calibrate the visual stimulus alignment with the screen and microscope frames.


### 3.1 Optical Offset Status — ✅ ALL ACTIVE SESSIONS VALIDATED

All sessions loaded by the manuscript figures have valid optical offsets:
- **`colasa_3d-vision_revisions` (29 sessions)**: Audited against middle-frame offset recomputes (Job `52281594`). All sessions show \(\le 0.7\%\) difference (\(\Delta \le 8\) counts on ~2160 baseline).
  - **`PZAG16.3c_S20250401`**: Re-extracted with post-fix middle-frame offsets (`[-18.72, -20.06]`) baked into traces (Job `52280899`).
  - **`PZAG16.3b_S20250401`**: Verified and marked good — offset shift (~15 counts) is negligible for trace dynamics and \(\Delta F/F\).
  - **`PZAH17.1e_S20250306`**: Verified and marked good for Multidays — bimodal dark pixels due to FOV vignetting; 100% positive \(F_0\) and <1% negative \(\Delta F/F\). (Excluded from main figures only due to <10 trials/depth and no RF).
- **`hey2_3d-vision_foodres_20220101` (50 sessions)**: All active sessions have valid optical offsets. The only session with an offset flag (`PZAH10.2d_S20230706`) is permanently excluded.

### Sanity Plot Checklist
- [x] Verify optical offset sanity plots for all active sessions in `hey2_3d-vision_foodres_20220101`.
- [x] Verify optical offset sanity plots for all revision sessions in `colasa_3d-vision_revisions`.
- [x] Confirm bad offset exclusion rules in the automated analysis pipelines.

---

## 4. Sessions Status Table

### Project: `colasa_3d-vision_revisions`

| Session Name | Mouse | Recs | Status / Exclude Reason | Suite2p Traces | ASt_neuropil=False | Sanity Plot Checked |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PZAG16.3a_S20250213` | PZAG16.3a | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAG16.3a_S20250218` | PZAG16.3a | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAG16.3b_S20250218` | PZAG16.3b | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAG16.3b_S20250224` | PZAG16.3b | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3b_S20250225` | PZAG16.3b | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3b_S20250226` | PZAG16.3b | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3b_S20250310` | PZAG16.3b | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3b_S20250313` | PZAG16.3b | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3b_S20250317` | PZAG16.3b | 7 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3b_S20250401` | PZAG16.3b | 8 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3c_S20250218` | PZAG16.3c | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAG16.3c_S20250219` | PZAG16.3c | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3c_S20250220` | PZAG16.3c | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3c_S20250221` | PZAG16.3c | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3c_S20250310` | PZAG16.3c | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3c_S20250313` | PZAG16.3c | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3c_S20250317` | PZAG16.3c | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG16.3c_S20250401` | PZAG16.3c | 6 | ✅ Re-extracted (Job `52280899`), `iscell` fixed | ✅ Yes | ✅ Yes | [x] Offsets baked in |
| `PZAG17.3a_S20250218` | PZAG17.3a | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAG17.3a_S20250227` | PZAG17.3a | 4 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG17.3a_S20250228` | PZAG17.3a | 9 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG17.3a_S20250303` | PZAG17.3a | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG17.3a_S20250305` | PZAG17.3a | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG17.3a_S20250306` | PZAG17.3a | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG17.3a_S20250319` | PZAG17.3a | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG17.3a_S20250402` | PZAG17.3a | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAG17.3b_S20250218` | PZAG17.3b | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAH17.1c_S20250213` | PZAH17.1c | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAH17.1c_S20250214` | PZAH17.1c | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAH17.1c_S20250217` | PZAH17.1c | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAH17.1e_S20250218` | PZAH17.1e | 1 | Habituation / Single rec | ❌ No | ❌ No | - |
| `PZAH17.1e_S20250304` | PZAH17.1e | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAH17.1e_S20250305` | PZAH17.1e | 8 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAH17.1e_S20250306` | PZAH17.1e | 5 | ✅ Fine for Multidays (Day 3; Excluded from main figs: <10 trials/depth & no RF) | ✅ Yes | ✅ Yes | [x] |
| `PZAH17.1e_S20250307` | PZAH17.1e | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAH17.1e_S20250311` | PZAH17.1e | 5 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAH17.1e_S20250313` | PZAH17.1e | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAH17.1e_S20250318` | PZAH17.1e | 6 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |
| `PZAH17.1e_S20250403` | PZAH17.1e | 7 | ✅ Fine | ✅ Yes | ✅ Yes | [x] |

---

### Project: `hey2_3d-vision_foodres_20220101` (Primary Figures Dataset)

*Generated by [`gen_session_table.py`](../v1_depth_map/revisions/gen_session_table.py) — `python gen_session_table.py notes/01_preprocessing.md`. **In figure list** is `get_sessions(v1_only=True, trialnum_min=10)`, the same call `analyze_all_sessions.py` uses. **2P recs** counts two-photon recordings — the old hand-written table counted behavioural ones, so the numbers differ. **ast:False available** counts loadable `SpheresPermTubeReward` recordings that resolve to a real `dff.npy`.*

| Session Name | Mouse | 2P recs | In figure list | Suite2p ROIs | ast:False available | Notes |
| :--- | :--- | ---: | :---: | :---: | :---: | :--- |
| `PZAG3.4f_S20220419` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220421` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220426` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220503` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220504` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220505` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220510` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220511` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220512` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAG3.4f_S20220520` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAG3.4f_S20220523` | PZAG3.4f | 3 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAG3.4f_S20220524` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAG3.4f_S20220526` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAG3.4f_S20220527` | PZAG3.4f | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230526` | PZAH10.2d | 5 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2d_S20230531` | PZAH10.2d | 1 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2d_S20230602` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230608` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230613` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230623` | PZAH10.2d | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230626` | PZAH10.2d | 5 | — No | ✅ Yes | 0/1 | ❌ Excluded (not V1) |
| `PZAH10.2d_S20230627` | PZAH10.2d | 1 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2d_S20230628` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230703` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230704` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230706` | PZAH10.2d | 5 | — No | ✅ Yes | 1/1 | ❌ Excluded (`exclude_reason` = Bad offset estimation); sanity: Bad offsets |
| `PZAH10.2d_S20230707` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230725` | PZAH10.2d | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230728` | PZAH10.2d | 6 | — No | ✅ Yes | 0/2 | no ast:False traces — excluded by trialnum_min |
| `PZAH10.2d_S20230818` | PZAH10.2d | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230821` | PZAH10.2d | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230822` | PZAH10.2d | 1 | — No | ✅ Yes | — | ✅ Size control re-run done (Job `52277005`); loaded by `figsupp_size_control` |
| `PZAH10.2d_S20230920` | PZAH10.2d | 7 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2d_S20230922` | PZAH10.2d | 7 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2f_S20230601` | PZAH10.2f | 5 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2f_S20230606` | PZAH10.2f | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2f_S20230609` | PZAH10.2f | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2f_S20230615` | PZAH10.2f | 7 | ✅ Yes | ✅ Yes | 3/3 |  |
| `PZAH10.2f_S20230622` | PZAH10.2f | 5 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2f_S20230623` | PZAH10.2f | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2f_S20230626` | PZAH10.2f | 5 | — No | ✅ Yes | 0/1 | ❌ Excluded (not V1) |
| `PZAH10.2f_S20230627` | PZAH10.2f | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2f_S20230703` | PZAH10.2f | 5 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2f_S20230706` | PZAH10.2f | 5 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2f_S20230707` | PZAH10.2f | 5 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2f_S20230727` | PZAH10.2f | 8 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH10.2f_S20230807` | PZAH10.2f | 5 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2f_S20230815` | PZAH10.2f | 3 | — No | ✅ Yes | 3/3 | ✅ Size control re-run done (Job `52277006`); loaded by `figsupp_size_control` |
| `PZAH10.2f_S20230817` | PZAH10.2f | 9 | ✅ Yes | ✅ Yes | 4/4 |  |
| `PZAH10.2f_S20230822` | PZAH10.2f | 4 | ✅ Yes | ✅ Yes | 4/4 |  |
| `PZAH10.2f_S20230907` | PZAH10.2f | 1 | — No | ✅ Yes | — | ✅ Size control re-run done (Job `52277007`); loaded by `figsupp_size_control` |
| `PZAH10.2f_S20230908` | PZAH10.2f | 4 | ✅ Yes | ✅ Yes | 4/4 |  |
| `PZAH10.2f_S20230912` | PZAH10.2f | 8 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH10.2f_S20230914` | PZAH10.2f | 8 | ✅ Yes | ✅ Yes | 3/3 |  |
| `PZAH10.2f_S20230924` | PZAH10.2f | 7 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH6.4b_S20220419` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220426` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220429` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220503` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220505` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220506` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220511` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220512` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220516` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH6.4b_S20220519` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH6.4b_S20220524` | PZAH6.4b | 2 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH8.2f_S20230109` | PZAH8.2f | 1 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2f_S20230117` | PZAH8.2f | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2f_S20230126` | PZAH8.2f | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2f_S20230131` | PZAH8.2f | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2f_S20230202` | PZAH8.2f | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2f_S20230206` | PZAH8.2f | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH8.2f_S20230214` | PZAH8.2f | 7 | ✅ Yes | ✅ Yes | 3/3 |  |
| `PZAH8.2f_S20230223` | PZAH8.2f | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH8.2f_S20230313` | PZAH8.2f | 7 | ✅ Yes | ✅ Yes | 3/3 |  |
| `PZAH8.2h_S20230113` | PZAH8.2h | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2h_S20230116` | PZAH8.2h | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2h_S20230126` | PZAH8.2h | 6 | — No | ✅ Yes | 0/1 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2h_S20230127` | PZAH8.2h | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2h_S20230202` | PZAH8.2h | 7 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH8.2h_S20230224` | PZAH8.2h | 7 | ✅ Yes | ✅ Yes | 3/3 |  |
| `PZAH8.2h_S20230302` | PZAH8.2h | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH8.2h_S20230303` | PZAH8.2h | 8 | ✅ Yes | ✅ Yes | 4/4 |  |
| `PZAH8.2h_S20230310` | PZAH8.2h | 5 | — No | ✅ Yes | 0/5 | ❌ Excluded (too few trials) |
| `PZAH8.2h_S20230314` | PZAH8.2h | 7 | — No | ✅ Yes | 0/3 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2h_S20230321` | PZAH8.2h | 6 | — No | ✅ Yes | 0/2 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2i_S20221208` | PZAH8.2i | 6 | — No | ✅ Yes | 0/1 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2i_S20221209` | PZAH8.2i | 6 | — No | ✅ Yes | 0/1 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2i_S20221213` | PZAH8.2i | 6 | — No | ✅ Yes | 0/1 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2i_S20221215` | PZAH8.2i | 6 | — No | ✅ Yes | 0/1 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2i_S20230110` | PZAH8.2i | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2i_S20230116` | PZAH8.2i | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2i_S20230117` | PZAH8.2i | 6 | ✅ Yes | ✅ Yes | 1/1 |  |
| `PZAH8.2i_S20230127` | PZAH8.2i | 6 | — No | ✅ Yes | 0/1 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2i_S20230203` | PZAH8.2i | 3 | ✅ Yes | ✅ Yes | 3/3 |  |
| `PZAH8.2i_S20230209` | PZAH8.2i | 5 | ✅ Yes | ✅ Yes | 3/3 |  |
| `PZAH8.2i_S20230216` | PZAH8.2i | 9 | ✅ Yes | ✅ Yes | 5/5 |  |
| `PZAH8.2i_S20230220` | PZAH8.2i | 9 | ✅ Yes | ✅ Yes | 5/5 |  |
| `PZAH8.2i_S20230324` | PZAH8.2i | 6 | — No | ✅ Yes | 0/2 | no ast:False traces — excluded by trialnum_min |
| `PZAH8.2i_S20230330` | PZAH8.2i | 6 | ✅ Yes | ✅ Yes | 2/2 |  |
| `PZAH8.2i_S20230404` | PZAH8.2i | 6 | ✅ Yes | ✅ Yes | 2/2 |  |

> [!NOTE]
> Rows marked `— No` under **In figure list** are not returned by `get_sessions()` and so are not loaded
> by the depth-figure pipeline. They are still tracked here when a notebook names them directly — the
> Notes column says which. `— No` means "not in `get_sessions()`", not "unused".
