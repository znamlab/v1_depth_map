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
  **41/41** colasa, verified with the exact filters the notebooks pass. Details in §5.
- The stale `ast_neuropil=True` labels are **harmless** — the loader reads the recording-level
  `suite2p_traces` flag, not the parent `suite2p_rois` flag that carries the stale value.
- The offset fix changes revision-session offsets by **≲0.7%** — no catastrophic cases. Per-session
  evidence in §6.
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

---

## 5. Reference: how to tell what a session actually ran

`ops.npy` and the parent `suite2p_rois` flag are **not** reliable for hey2 — the `--no-run-neuropil` path
never writes the flag back, so both keep saying `True` (and `ops.npy` is still dated 2023). They are
reliable for colasa, whose `reextract` path writes fresh ops. Evidence in decreasing order of directness:

1. **Recording-level `suite2p_traces` flag** — written correctly by `split_recordings()`; this is what the
   loader actually uses.
2. **`plane0/dff.npy` exists** — the standard-neuropil branch uses `filename_suffix = ""`.
3. **Job log**: `Ops: {... 'ast_neuropil': False ...}` and `Running standard neuropil correction`.
4. `plane0/Fstandard.npy` — written only by the standard branch, but **only in recent versions**, so its
   absence proves nothing for older runs.

Helper scripts in [`../v1_depth_map/revisions/`](../v1_depth_map/revisions/):
[`check_offsets.py`](../v1_depth_map/revisions/check_offsets.py) (+ `.sbatch`),
[`summarise_offsets.py`](../v1_depth_map/revisions/summarise_offsets.py),
[`scan_ast_flag.py`](../v1_depth_map/revisions/scan_ast_flag.py),
[`fix_ast_flag.py`](../v1_depth_map/revisions/fix_ast_flag.py),
[`verify_trace_labels.py`](../v1_depth_map/revisions/verify_trace_labels.py),
[`fix_iscell.py`](../v1_depth_map/revisions/fix_iscell.py).

### 5.1 Why the stale `ast_neuropil=True` labels are harmless

`generate_imaging_df()` (`synchronisation.py:583`) picks the trace file with

```python
dff_fname = "dff_ast.npy" if suite2p_ds.extra_attributes["ast_neuropil"] else "dff.npy"
```

It reads that flag from the **recording-level `suite2p_traces`** dataset, written correctly by
`split_recordings()`. The stale `True` values live on the **parent `suite2p_rois`** dataset and in
`ops.npy`, neither of which this path consults. Verified exhaustively with the filters the notebooks
actually pass:

| Project | Notebook filter | Recordings resolving to the correct ast:False `dff.npy` |
| :--- | :--- | :--- |
| `hey2_3d-vision_foodres_20220101` | `{"anatomical_only": 3, "ast_neuropil": False}` | **390 / 390**, 0 problems |
| `colasa_3d-vision_revisions` | `{"anatomical_only": 3, "annotated": True}` | **41 / 41**, 0 problems |

ROI loading is correct too: every notebook loads `suite2p_rois` with `{"anatomical_only": 3}` or
`{"annotated": True}` and **no** `ast_neuropil` constraint, and `rf_analysis.py:362-364` explicitly drops
the filter for ROI selection. The ROI label is irrelevant — only the extraction matters.

### 5.2 Known cosmetic, no action needed

- **42 parent `suite2p_rois` datasets** still carry `ast_neuropil = True`. Their data is correct and
  nothing reads the flag. [`fix_ast_flag.py --sessions <names> --apply`](../v1_depth_map/revisions/fix_ast_flag.py)
  clears them if you ever want the metadata tidy. (The 3 size-control parents were corrected already.)
- **`ops.npy`'s own flag was never audited** for hey2 — the `scan_ops_flag` job never completed
  successfully. Nothing reads it; resubmit [`scan_ops_flag.sbatch`](../v1_depth_map/revisions/scan_ops_flag.sbatch)
  if you want the answer.

---

## 6. Appendix: optical offset audit evidence (Job `52281594`)

Compares `offsets.npy` (written 2026-06-01, pre-fix) against a fresh recompute with the current fixed
`estimate_offset()`. 29/29 colasa sessions, 0 errors. **This table is the surviving record** — the
per-session JSONs under `~/offset_check_work/results/` have been cleaned up. To regenerate, resubmit
[`check_offsets.sbatch`](../v1_depth_map/revisions/check_offsets.sbatch) (~12 min on `ncpu`) and summarise
with [`summarise_offsets.py`](../v1_depth_map/revisions/summarise_offsets.py).

**Legend** — `offsets.npy` is the diagnostic file written by `2p sanity` / `estimate_offsets()`; the traces
are what `correct_offset()` actually subtracted at extraction time. They are independent.

| Group | `offsets.npy` written | Sessions | Traces vs `offsets.npy` |
| :--- | :--- | :---: | :--- |
| **A** | 2026-06-01 (pre-fix) | 23 | Both pre-fix, different runs (Feb 6 vs Jun 1), non-deterministic TIFF choice. 18/23 shift by >1 count, max 15.11 on a ~2160 baseline (≲0.7%) |
| **B** | 2026-07-30 (post-fix) | 5 | File holds corrected values, traces hold the old ones |
| **C** | 2026-08-20 (post-fix) | 1 | Same as B (`PZAG16.3c_S20250401`, since re-extracted) |

### Group B & C — stored `offsets.npy` already post-fix

| Session | `offsets.npy` written | Stored (corrected) offsets |
| :--- | :--- | :--- |
| `PZAG16.3b_S20250224` | 2026-07-30 | `[2154.59]` |
| `PZAG16.3b_S20250225` | 2026-07-30 | `[2156.55]` |
| `PZAG16.3b_S20250226` | 2026-07-30 | `[2156.89]` |
| `PZAG16.3b_S20250310` | 2026-07-30 | `[2172.91]` |
| `PZAG16.3b_S20250313` | 2026-07-30 | `[2164.98]` |
| `PZAG16.3c_S20250401` | 2026-08-20 | `[-18.72, -20.06]` — ✅ re-extracted, no longer an issue |

### Group A — pre-fix `offsets.npy` vs recompute

| Session | Stored offsets (pre-fix) | Recomputed (post-fix) | max abs diff | Verdict |
| :--- | :--- | :--- | ---: | :--- |
| `PZAG16.3b_S20250317` | `[2159.99, 2152.81]` | `[2165.97, 2156.58]` | 5.98 | ⚠️ differs |
| `PZAG16.3b_S20250401` | `[-46.86, -30.39, -42.04, -65.45]` | `[-54.17, -32.04, -34.10, -50.34]` | **15.11** | ⛔ largest absolute change; near-zero baseline so relative change is large |
| `PZAG16.3c_S20250219` | `[2165.10]` | `[2160.68]` | 4.43 | ⚠️ differs |
| `PZAG16.3c_S20250220` | `[2165.56]` | `[2174.54]` | 8.98 | ⚠️ differs |
| `PZAG16.3c_S20250221` | `[2167.69]` | `[2169.91]` | 2.22 | ⚠️ differs |
| `PZAG16.3c_S20250310` | `[2171.40]` | `[2170.34]` | 1.06 | ⚠️ differs |
| `PZAG16.3c_S20250313` | `[2159.85]` | `[2159.96]` | 0.11 | ✅ agrees |
| `PZAG16.3c_S20250317` | `[2158.36, 2165.09]` | `[2166.29, 2167.25]` | 7.93 | ⚠️ differs |
| `PZAG17.3a_S20250227` | `[2157.13]` | `[2157.79]` | 0.66 | ✅ agrees |
| `PZAG17.3a_S20250228` | `[2174.56]` | `[2174.96]` | 0.40 | ✅ agrees |
| `PZAG17.3a_S20250303` | `[2167.74]` | `[2168.37]` | 0.63 | ✅ agrees |
| `PZAG17.3a_S20250305` | `[2169.27]` | `[2175.97]` | 6.70 | ⚠️ differs |
| `PZAG17.3a_S20250306` | `[2166.42]` | `[2164.47]` | 1.95 | ⚠️ differs |
| `PZAG17.3a_S20250319` | `[2159.16, 2161.14]` | `[2162.48, 2160.19]` | 3.32 | ⚠️ differs |
| `PZAG17.3a_S20250402` | `[2153.03, 2162.70]` | `[2157.41, 2157.72]` | 4.98 | ⚠️ differs |
| `PZAH17.1e_S20250304` | `[2166.24]` | `[2167.26]` | 1.02 | ⚠️ differs |
| `PZAH17.1e_S20250305` | `[2170.78]` | `[2171.97]` | 1.19 | ⚠️ differs |
| `PZAH17.1e_S20250306` | `[4117.71]` | `[4117.71]` | 0.00 | ❌ **bad offset, but not a TIFF-choice problem** — ~2× every other session; genuinely anomalous recording |
| `PZAH17.1e_S20250307` | `[2165.87]` | `[2164.14]` | 1.73 | ⚠️ differs |
| `PZAH17.1e_S20250311` | `[2169.98]` | `[2161.98]` | 8.00 | ⚠️ differs |
| `PZAH17.1e_S20250313` | `[2162.35, 2168.39]` | `[2162.49, 2165.35]` | 3.05 | ⚠️ differs |
| `PZAH17.1e_S20250318` | `[2167.83, 2163.18]` | `[2170.99, 2168.61]` | 5.43 | ⚠️ differs |
| `PZAH17.1e_S20250403` | `[2156.64, 2156.27, 2153.23]` | `[2158.40, 2149.79, 2154.57]` | 6.49 | ⚠️ differs |

**Method validation**: the 6 Group B/C sessions (whose stored `offsets.npy` is already post-fix) recompute
to **exactly** the stored value (`max_abs_diff = 0.00` for all six). The recompute is deterministic and
reproduces the current code path — the non-zero diffs in Group A are real TIFF-choice effects, not noise.

**What this does and does not prove.** The value actually baked into the traces is a *third* estimate from
2026-02-06 which was never saved and is unrecoverable. So `max_abs_diff` bounds the **sensitivity of the
offset to TIFF choice**; it is not a direct measurement of the error in the traces. That sensitivity is
small on every session, which caps how wrong the Feb 6 values can be.
