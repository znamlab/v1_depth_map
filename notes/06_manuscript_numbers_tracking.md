# 06 - Manuscript Numbers Tracking & Audit

This tracking document catalogs **every number cited in the manuscript** (`v1_depth_map.tex`), cross-references each value against the automated statistics pipeline (`MANUSCRIPT_STATS.md` / `manuscript_stats.tex` / generated YAML manifests), tracks all newly exported metrics, and inventories all compiled statistics that are not currently cited in the text.

> [!IMPORTANT]
> **No changes were made to the manuscript (`v1_depth_map.tex`).** Per user instructions, this document serves exclusively as an audit and tracking file in `notes/` to prepare for eventual synchronization.

---

## 1. Executive Summary & Audit Dashboard

### Overall Metrics Status

| Category | Count | Description | Action Required |
| :--- | :---: | :--- | :--- |
| 🔴 **Outdated in Paper (Mismatch)** | **17** | Numbers in `v1_depth_map.tex` that differ from the latest notebook runs | Update `v1_depth_map.tex` with latest stats |
| 🟢 **Up to Date (Exact Match)** | **41** | Dynamic results in the paper that match `MANUSCRIPT_STATS.md` | Ready to bind to `\stat...{}` macros |
| 🟣 **Placeholders in Paper** | **4** | Literal `STATS` / `STAT` tokens in text awaiting values (drifting gratings) | Replace with corresponding `\statRevGrat...` macros |
| ⚪ **Static Parameters / Constants** | **45+** | Setup geometry, laser wavelength, filter bands, durations, bounds | Keep as hardcoded literals (by design rule) |
| 🔵 **Stats Metrics Not Cited in Paper** | **62** | Precomputed metrics in YAML manifests not directly cited in prose | Available for expanded text or reference |

**Total Metrics Compiled in Stats Pipeline**: **196** across 14 notebook YAML manifests (100% notebook-generated, 0 hand-entered).

---

## 2. Newly Added Metrics Exported to Stats Manifest

The following metrics, previously cited as hardcoded literals or missing macros, have now been added to the figure notebooks and compiled into `MANUSCRIPT_STATS.md` and `manuscript_stats.tex`:

| Macro Name | Formatted | Raw | Source Notebook | Manuscript Location & Description |
| :--- | :---: | :---: | :--- | :--- |
| `\statRevGratOtherSftfNeurons` | `1,052` | 1052 | `figsupp10_gratings.ipynb` | Fig S10E (L978): Non-depth-selective SF/TF-tuned neurons ($1,442 - 390 = 1,052$) |
| `\statFigOneMaleMice` | `1` | 1 | `figure1_depth_selectivity.ipynb` | Methods (L527): Male mice recorded in layer 2/3 of V1 |
| `\statFigOneFemaleMice` | `6` | 6 | `figure1_depth_selectivity.ipynb` | Methods (L527): Female mice recorded in layer 2/3 of V1 |
| `\statFigTwoConjunctiveThreshold` | `0.0016` | 0.00158 | `figure2_rsof_integration.ipynb` | Methods (L638): Empirical null $R^2$ threshold for 2D conjunctive model (cited as $\approx 0.002$) |
| `\statSuppNineAPCorrPval` | `$p < 0.0001$` | $1.0 \times 10^{-74}$ | `figsupp9_v1_depth_map.ipynb` | Methods / S9 (L273): p-value of correlation between preferred depth and AP location in V1 |

---

## 3. Critical Discrepancies (Outdated Numbers in Paper)

The table below lists all numbers in the paper text and captions that have silently drifted from the current outputs of the figure notebooks (due to re-runs, updated cell inclusion, or spike RF fits):

| # | Manuscript Line | Context / Caption Snippet | Value in Paper (`v1_depth_map.tex`) | Value in Stats File (`manuscript_stats.tex`) | Relevant Macro | Source Notebook & Cause |
| :-: | :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | L148, L161, L286 | Total depth-selective neurons & population % | **27,094 (45.2%)** | **27,127 (45.3%)** | `\statFigOneDepthNeurons`<br>`\statFigOnePctDepthNeurons` | `figure1_depth_selectivity.ipynb`<br>Updated depth classification output (+33 cells) |
| **2** | L195, L196 | Open-loop comparison depth-selective neurons | **3,948** | **3,961** | `\statFigTwoOpenLoopNeurons` | `figure2_openloop.ipynb`<br>Slightly updated cell selection (+13 cells) |
| **3** | L210 | Closed vs open loop peak amplitude ratio p-value | **$p = 0.743$** | **$p = 0.735$** | `\statFigTwoAmpRatioPval` | `figure2_openloop.ipynb`<br>Bootstrap resample variation |
| **4** | L210 | Preferred running speed closed vs open loop correlation | **$r = 0.469$** | **$r = 0.470$** | `\statFigTwoRSCorrR` | `figure2_openloop.ipynb`<br>Rounding / updated cell selection |
| **5** | L210 | Preferred optic flow closed vs open loop correlation | **$r = 0.535$** | **$r = 0.536$** | `\statFigTwoOFCorrR` | `figure2_openloop.ipynb`<br>Rounding / updated cell selection |
| **6** | L265 | Pairwise RF distance depth-tuned neurons | **13,425** | **16,974** | `\statFigFourRFDistNeurons` | `figure4_receptive_fields.ipynb`<br>Spike RF re-run expanded analyzed cells |
| **7** | L545-L546 | Average display lag and cohort sessions | **$26.6 \pm 1.4~\text{ms}$**<br>(5 mice, 60 sessions) | **$26.4 \pm 0.1~\mathrm{ms}$**<br>(5 mice, 59 sessions) | `\statSuppDisplayLag`<br>`\statFigOneEightDepthSessions` | `figsupp1_vis_stim_sync.ipynb`<br>Lag refined; total sessions is 84 (25+59=84) |
| **8** | L634 | Neurons exceeding $R^2$ threshold for model comparison | **21,318 of 27,094** | **21,457 of 27,127** | `\statFigTwoModelNeurons`<br>`\statFigOneDepthNeurons` | `figure2_rsof_integration.ipynb`<br>Updated model fit population |
| **9** | L638 | Closed-loop conjunctive model significant fits | **17,953 of 27,094** | **18,049 of 27,127** | `\statFigTwoClosedLoopConjunctiveNeurons` | `figure2_rsof_integration.ipynb`<br>Updated model fit population |
| **10** | L641 | Open-loop conjunctive model significant fits | **3,948 of 9,429** | **3,961 of 9,462** | `\statFigTwoOpenLoopNeurons`<br>`\statFigTwoOpenLoopTotalDepthNeurons` | `figure2_openloop.ipynb`<br>Updated open-loop dataset |
| **11** | L866 | Fig S3L: 8-depth cohort depth-tuned neurons | **7,332** | **7,365** | `\statSuppDepthPopEightDepthNeurons` | `figsupp3_depth_pop.ipynb`<br>Refined depth-tuned count (+33 cells) |
| **12** | L937 | Fig S8D: RF correlation comparison neurons & stats | **$N = 895$**<br>Contra $r = 0.286$<br>Ipsi $r = 0.042$<br>$p = 2.18 \times 10^{-124}$ | **$N = 1,009$**<br>Contra $r = 0.334$<br>Ipsi $r = 0.031$<br>$p < 0.0001$ | `\statSuppEightRFCorrNeurons`<br>`\statSuppEightRFCorrMedianContra`<br>`\statSuppEightRFCorrMedianIpsi`<br>`\statSuppEightRFCorrPval` | `figsupp8_multidepth_receptive_fields.ipynb`<br>Hierarchical bootstrap over mice/sessions |
| **13** | L939 | Fig S8F: Multi- vs single-depth peak depth stats | **$N = 603$**<br>$r = 0.587$, $p = 4.05 \times 10^{-57}$<br>ratio = 0.865, IQR: 0.631--1.290 | **$N = 1,009$**<br>$r = 0.561$, $p < 0.0001$<br>ratio = 0.939, IQR: 0.669--1.506 | `\statSuppEightPeakDepthNeurons`<br>`\statSuppEightPeakDepthR`<br>`\statSuppEightPeakDepthPval`<br>`\statSuppEightPeakDepthMedianRatio`<br>`\statSuppEightPeakDepthRatioIQR` | `figsupp8_multidepth_receptive_fields.ipynb`<br>Hierarchical bootstrap over mice/sessions |
| **14** | L955, L956 | Fig S9B-E: Spatial distribution RF neurons | **14,173** (84 sessions) | **16,974** (84 sessions) | `\statSuppNineRFNeuronsTotal` | `figsupp9_v1_depth_map.ipynb`<br>Spike RF switch increased RF-significant cell count |
| **15** | L957 | Fig S9F: Near / Mid / Far corrected neurons | Near: **2,476**<br>Mid: **5,832**<br>Far: **5,865** | Near: **3,315**<br>Mid: **7,288**<br>Far: **6,371** | `\statSuppNineNearCorrected`<br>`\statSuppNineMidCorrected`<br>`\statSuppNineFarCorrected` | `figsupp9_v1_depth_map.ipynb`<br>Reflected from 16,974 spike RF total |
| **16** | L958 | Fig S9G-H: Near / Mid / Far uncorrected neurons | Near: **3,078**<br>Mid: **6,744**<br>Far: **4,351** | Near: **4,132**<br>Mid: **8,100**<br>Far: **4,742** | `\statSuppNineNearUncorrected`<br>`\statSuppNineMidUncorrected`<br>`\statSuppNineFarUncorrected` | `figsupp9_v1_depth_map.ipynb`<br>Reflected from 16,974 spike RF total |
| **17** | L984, L987, L989 | Fig S10J, M, O: Drifting grating sample sizes & stats | J: **$N = 333$**, $p_{\mathrm{Holm}} = 0.0060$<br>M: **$N = 142$** (TF)<br>O: **$N = 176$** (SF) | J: **$N = 330$**, $p_{\mathrm{Holm}} = 0.0025$<br>M: **$N = 218$** (TF)<br>O: **$N = 342$** (SF) | `\statRevGratSfOfN`<br>`\statRevGratSfOfPHolm`<br>`\statRevGratSpeedCurvesTfN`<br>`\statRevGratSpeedCurvesSfN` | `figsupp10_gratings.ipynb`<br>Methods (L770) had 218 and 342, but caption (L987/L989) had 142 and 176! |

---

## 4. Section-by-Section Complete Audit

### Section 1: Main Text — Introduction & Virtual Reality Depth Selectivity (Figure 1)

| Line | Manuscript Text Context | Value in Paper | Macro Name in Stats File | Value in Stats File | Status | Notes |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **L145** | Locomotion threshold | $>5\text{ cm/s}$ | — | — | ⚪ Static | Behavioral classification criterion |
| **L145** | Stationary threshold | $<5\text{ cm/s}$ (preceding 1 s) | — | — | ⚪ Static | Behavioral classification criterion |
| **L146** | Example session scale bar | $100\ \mu\text{m}$ | — | — | ⚪ Static | Plotting geometry |
| **L146** | Confidence interval | 95% | — | — | ⚪ Static | Plotting specification |
| **L148** | Fig 1J: Depth-selective neurons | $N = 27,094\text{ neurons}$ | `\statFigOneDepthNeurons` | $27,127$ | 🔴 Outdated | Updated notebook run yields 27,127 |
| **L149** | Fig 1K: Stimulus analysis window | $[-1\text{ m}, +1\text{ m}]$ | — | — | ⚪ Static | Analysis corridor bounds |
| **L149** | Fig 1K: Virtual depths in cohort | 8 virtual depths | `\statFigOneEightDepthSessions` | 59 sessions | 🟢 Match | Cohort condition description |
| **L150** | Fig 1L: Running speed decoder bin | $40\text{--}60\text{ cm/s}$ | — | — | ⚪ Static | Decoding evaluation bin |
| **L150** | Fig 1L: Confusion matrix sessions | $N = 59\text{ sessions}$ | `\statFigOneEightDepthSessions` | 59 | 🟢 Match | Exact match |
| **L151** | Fig 1M: Decoder error sessions | $N = 84\text{ sessions}$ | `\statFigOneTotalSessions` | 84 | 🟢 Match | Exact match |
| **L152** | Fig 1N: Experience cohort mice | $N = 4\text{ mice}$ | `\statFigOneMultidayMice` | 4 | 🟢 Match | Exact match |
| **L152** | Fig 1N: Experience cohort sessions | $19\text{ sessions}$ | `\statFigOneMultidaySessions` | 19 | 🟢 Match | Exact match |
| **L153** | Fig 1O: Consecutive days tracked | 5 consecutive days | — | — | ⚪ Static | Protocol duration |
| **L154** | Fig 1P: Consecutive day pairs | $N = 374\text{ pairs}$ | `\statFigOneMultidayPairs` | 374 | 🟢 Match | Exact match |
| **L154** | Fig 1P: Tracked unique neurons | $234\text{ neurons}$ | `\statFigOneMultidayTrackedNeurons` | 234 | 🟢 Match | Exact match |
| **L159** | 5-depth range & sessions | 6 to 600 cm, 25 sessions | `\statFigOneFiveDepthSessions` | 25 | 🟢 Match | Exact match |
| **L159** | 8-depth range & sessions | 5 to 640 cm, 59 sessions | `\statFigOneEightDepthSessions` | 59 | 🟢 Match | Exact match |
| **L159** | Corridor length | 6 m | — | — | ⚪ Static | Corridor length parameter |
| **L159** | Reward probability | 60–80% | — | — | ⚪ Static | Reward delivery parameter |
| **L161** | Total neurons recorded | 59,937 neurons | `\statFigOneTotalNeurons` | 59,937 | 🟢 Match | Exact match |
| **L161** | Total sessions recorded | 84 sessions | `\statFigOneTotalSessions` | 84 | 🟢 Match | Exact match |
| **L161** | Total mice recorded | 7 mice | `\statFigOneTotalMice` | 7 | 🟢 Match | Exact match |
| **L161** | Depth-selective cell count | 27,094 | `\statFigOneDepthNeurons` | 27,127 | 🔴 Outdated | Updated run: 27,127 |
| **L161** | Depth-selective cell fraction | 45.2% | `\statFigOnePctDepthNeurons` | 45.3% | 🔴 Outdated | Raw: 45.259% rounds to 45.3% |
| **L163** | Stationary speed criterion | $<5\text{ cm/s}$ in 1 s | — | — | ⚪ Static | Behavioral classification criterion |
| **L165** | Virtual size invariance angles | 5, 10 or 20 degrees | — | — | ⚪ Static | Stimulus parameter |

---

### Section 2: Main Text — Visuomotor Integration Models & Open-Loop Replay (Figure 2)

| Line | Manuscript Text Context | Value in Paper | Macro Name in Stats File | Value in Stats File | Status | Notes |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **L174** | Optic flow reference azimuth | 90 degrees | — | — | ⚪ Static | Definition of nominal corridor OF |
| **L190** | Fig 2E: Model comparison sessions | $N = 84\text{ sessions}$ | `\statFigTwoModelSessions` | 84 | 🟢 Match | Exact match |
| **L195** | Fig 2H: Closed vs open loop neurons | $N = 3,948\text{ neurons}$ | `\statFigTwoOpenLoopNeurons` | 3,961 | 🔴 Outdated | Refined open-loop inclusion (+13 cells) |
| **L196** | Fig 2I-J: RS and OF tuning neurons | $N = 3,948\text{ neurons}$ | `\statFigTwoOpenLoopNeurons` | 3,961 | 🔴 Outdated | Refined open-loop inclusion (+13 cells) |
| **L197** | Fig 2K: Closed vs open decoder sessions | $N = 34\text{ sessions}$ | `\statFigTwoDecoderTotalSessions` | 34 | 🟢 Match | Exact match |
| **L198** | Fig 2L: 8-depth confusion matrix sessions | $N = 27\text{ sessions}$ | `\statFigTwoDecoderEightDepthSessions` | 27 | 🟢 Match | Exact match |
| **L202** | Fig 2E: Additive & conjunctive model p-val | all $p < 0.0001$ | `\statFigTwoModelComparisonPval` | $\$p < 0.0001\$$ | 🟢 Match | Largest of the 6 key pairwise p-values |
| **L210** | Peak response ratio closed/open median | 0.98 | `\statFigTwoAmpRatioMedian` | 0.98 | 🟢 Match | Exact match |
| **L210** | Peak response ratio closed/open p-value | $p = 0.743$ | `\statFigTwoAmpRatioPval` | $\$p = 0.735\$$ | 🔴 Outdated | Resample variation (0.743 vs 0.735) |
| **L210** | Preferred RS closed vs open correlation | $r = 0.469$, $p < 0.0001$ | `\statFigTwoRSCorrR`<br>`\statFigTwoRSCorrPval` | $\$r = 0.470\$$, $\$p < 0.0001\$$ | 🔴 Outdated | $r = 0.469$ vs $0.470$ |
| **L210** | Preferred OF closed vs open correlation | $r = 0.535$, $p < 0.0001$ | `\statFigTwoOFCorrR`<br>`\statFigTwoOFCorrPval` | $\$r = 0.536\$$, $\$p < 0.0001\$$ | 🔴 Outdated | $r = 0.535$ vs $0.536$ |

---

### Section 3: Main Text — Motorized Wheel & Visuomotor Gain Modulation (Figure 3)

| Line | Manuscript Text Context | Value in Paper | Macro Name in Stats File | Value in Stats File | Status | Notes |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **L216** | Constant speed plateau duration | at least 2 s | — | — | ⚪ Static | Motorized protocol constant |
| **L232** | Fig 3F: Motorized vs free depth neurons | $n = 285\text{ neurons}$ | `\statFigThreeMotorizedDepthNeurons` | 285 | 🟢 Match | Exact match |
| **L237** | Fig 3K: Motorized 2D fit neurons | $n = 285\text{ neurons}$ | `\statFigThreePolarNeurons` | 285 | 🟢 Match | Exact match |
| **L237** | Fig 3K: Motorized sessions | 4 sessions | `\statFigThreePolarSessions` | 4 | 🟢 Match | Exact match |
| **L237** | Fig 3K: Motorized mice | 4 mice | `\statFigThreePolarMice` | 4 | 🟢 Match | Exact match |
| **L237** | Fig 3K: Elongation $> 1.4$ ratio | $266/285$ | `\statFigThreeElongationCutoffRatio` | 266/285 | 🟢 Match | Exact match |
| **L238** | Fig 3L: Elongated neurons | $n = 266$ | `\statFigThreeElongatedNeurons` | 266 | 🟢 Match | Exact match |
| **L238** | Fig 3L: Axial von Mises mixture components | $k = 3$ | `\statFigThreeVonMisesK` | 3 | 🟢 Match | Exact match |
| **L238** | Fig 3L: Von Mises component weights | $3^\circ$ (14%), $43^\circ$ (76%), $90^\circ$ (11%) | `\statFigThreeVonMisesComponents` | $\$3^\circ\$$ (14%), $\$43^\circ\$$ (76%), $\$90^\circ\$$ (11%) | 🟢 Match | Exact match |
| **L244** | Text: Orientation boundary angles | $100^\circ$, $-10^\circ$ | — | — | ⚪ Static | Quadrant boundary definitions |
| **L244** | Text: Peak at $0^\circ$ (OF axis) | 14%, $N = 266$ neurons, 4 sessions, 4 mice | `\statFigThreeVonMisesComponents` | 14% | 🟢 Match | Exact match |
| **L244** | Text: Peak at $90^\circ$ (RS axis) | 11% | `\statFigThreeVonMisesComponents` | 11% | 🟢 Match | Exact match |
| **L244** | Text: Peak at $\sim 45^\circ$ (Diagonal) | 76% | `\statFigThreeVonMisesComponents` | 76% | 🟢 Match | Exact match |

---

### Section 4: Main Text — 3D Receptive Fields & Drifting Gratings (Figure 4)

| Line | Manuscript Text Context | Value in Paper | Macro Name in Stats File | Value in Stats File | Status | Notes |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **L260** | Reconstructed stimulus spatial resolution | 5 degrees | — | — | ⚪ Static | Grid binning resolution |
| **L261** | Rendered RF iso-response shells | 50, 70 and 90% of max | — | — | ⚪ Static | Visual contour thresholds |
| **L262** | Fig 4E: Multi-depth RF population neurons | $N = 1,009\text{ neurons}$ | `\statFigFourPopNeurons` | 1,009 | 🟢 Match | Exact match |
| **L262** | Fig 4E: Multi-depth RF sessions | 5 sessions | `\statFigFourPopSessions` | 5 | 🟢 Match | Exact match |
| **L262** | Fig 4E: Multi-depth RF mice | 4 mice | `\statFigFourPopMice` | 4 | 🟢 Match | Exact match |
| **L263** | Fig 4F: Scale bar | $100\ \mu\text{m}$ | — | — | ⚪ Static | Visual scale bar |
| **L265** | Fig 4I-K: Pairwise RF distance neurons | $N = 13,425\text{ neurons}$ | `\statFigFourRFDistNeurons` | 16,974 | 🔴 Outdated | Spike RF update expanded sample to 16,974 |
| **L265** | Fig 4I-K: Pairwise RF distance sessions | 84 sessions | `\statFigFourRFDistSessions` | 84 | 🟢 Match | Exact match |
| **L265** | Pairwise exclusion distance | $10\ \mu\text{m}$ | — | — | ⚪ Static | ROI exclusion threshold |
| **L282** | Text: Depth vs grating speed correlation | **(STATS, Figure \ref{sup:gratings}G)** | `\statRevGratSpeedDepthRho`<br>`\statRevGratSpeedDepthPHolm` | $\rho = -0.16$, $p_{\mathrm{Holm}} = 0.0276$ | 🟣 Placeholder | Needs stats inserted in main text |
| **L282** | Text: Depth vs grating TF correlation | **(STAT, Figure \ref{sup:gratings}H)** | `\statRevGratTfDepthRho`<br>`\statRevGratTfDepthPHolm` | $\rho = 0.06$, $p_{\mathrm{Holm}} = 1.0$ | 🟣 Placeholder | Needs stats inserted in main text |
| **L282** | Text: Depth vs grating SF correlation | **(STAT, Figure \ref{sup:gratings}I)** | `\statRevGratSfDepthRho`<br>`\statRevGratSfDepthPHolm` | $\rho = 0.36$, $p_{\mathrm{Holm}} = 0.0291$ | 🟣 Placeholder | Needs stats inserted in main text |
| **L282** | Text: SF vs optic flow speed correlation | **(STAT, Figure \ref{sup:gratings}J)** | `\statRevGratSfOfRho`<br>`\statRevGratSfOfPHolm` | $\rho = -0.41$, $p_{\mathrm{Holm}} = 0.0025$ | 🟣 Placeholder | Needs stats inserted in main text |
| **L286** | Discussion: Fraction of depth-selective neurons | 45.2% | `\statFigOnePctDepthNeurons` | 45.3% | 🔴 Outdated | Matches Fig 1 update (45.3%) |

---

### Section 5: Materials and Methods

| Line | Subsection | Context / Snippet | Value in Paper | Macro in Stats File | Value in Stats File | Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **L527** | Animals | Total transgenic mice | 7 | `\statFigOneTotalMice` | 7 | 🟢 Match | Cohort sample size |
| **L527** | Animals | Cre-line distribution | 6 Emx1, 1 CamKII | `\statSuppDepthPopSixfMice`<br>`\statSuppDepthPopSixsMice` | 6<br>1 | 🟢 Match | Exact match |
| **L527** | Animals | Sex breakdown | 1 male, 6 females | `\statFigOneMaleMice`<br>`\statFigOneFemaleMice` | 1<br>6 | 🟢 Match | Newly added macros |
| **L530** | Cranial window | Isoflurane anesthesia | 1.5–2.5% | — | — | ⚪ Static | Anesthesia concentration |
| **L530** | Cranial window | Dexamethasone dose | 0.02 ml at 2 mg/ml | — | — | ⚪ Static | Drug dosage |
| **L530** | Cranial window | Craniotomy & coverslip diameter | 4 mm | — | — | ⚪ Static | Surgical diameter |
| **L530** | Cranial window | Age at imaging | 3–7 months (up to 10) | — | — | ⚪ Static | Animal age |
| **L533** | VR setup | Number of monitors | 4 screens | — | — | ⚪ Static | Apparatus specification |
| **L533** | VR setup | Horizontal / vertical visual field | $\sim 240^\circ \times 80^\circ$ | — | — | ⚪ Static | Angular visual field |
| **L533** | VR setup | Eye tracking camera rate | $\sim 15\text{ Hz}$ | — | — | ⚪ Static | Acquisition frame rate |
| **L535** | VR stimuli | Background luminance | $11.9\text{ cd/m}^2$ | — | — | ⚪ Static | Photometric luminance |
| **L535** | VR stimuli | Floor width / position | 5 cm wide, 2.5 cm below | — | — | ⚪ Static | Virtual geometry |
| **L535** | VR stimuli | Number of spheres per trial | 24 black spheres | — | — | ⚪ Static | Stimulus density parameter |
| **L535** | VR stimuli | Sphere longitudinal spacing | $0.15 \times d$ (19 ahead, 5 behind) | — | — | ⚪ Static | Spatial arrangement formula |
| **L535** | VR stimuli | Cylindrical azimuth range | $[-40^\circ, 40^\circ]$ or $[140^\circ, 220^\circ]$ | — | — | ⚪ Static | Elevation/azimuth bounds |
| **L535** | VR stimuli | Fade-in distance / duration | $2.7 \times d$ ahead over 0.3 s | — | — | ⚪ Static | Temporal stimulus envelope |
| **L535** | VR stimuli | Sphere radius formula / angle | $\tan 5^\circ \times d$ ($10^\circ$ closest) | — | — | ⚪ Static | Size normalization formula |
| **L537** | VR stimuli | Mixed-depth stimulus spheres | 8 depths, $0.7 \times d$, 32 spheres | — | — | ⚪ Static | Multi-depth stimulus layout |
| **L539** | VR stimuli | Trial distance & ITI duration | 6 m corridor, 10 s ITI | — | — | ⚪ Static | Trial structure parameters |
| **L539** | VR stimuli | Reward volume, timing, rate | $\sim 8\ \mu\text{l}$, 2 s into ITI, 60–80% | — | — | ⚪ Static | Reinforcement schedule |
| **L539** | VR stimuli | Min trials per depth | at least 10 trials | — | — | ⚪ Static | Session inclusion rule |
| **L541** | Motorized wheel | Linear acceleration / deceleration | 7.7 cm/s$^2$ / 15.4 cm/s$^2$ | — | — | ⚪ Static | Kinematic motor profile |
| **L541** | Motorized wheel | Target speeds & spacing | 3.8 to 61 cm/s ($\log_2$ steps) | — | — | ⚪ Static | Motor speeds sampled |
| **L541** | Motorized wheel | Optic flow speeds & spacing | 1 to 1024 deg/s ($2\log_2$ steps) | — | — | ⚪ Static | Optic flow speeds sampled |
| **L541** | Motorized wheel | Total stimulus grid combinations | 25 combinations | — | — | ⚪ Static | Grid size |
| **L545** | Sync & lag | Nominal monitor refresh rate | 144 Hz | — | — | ⚪ Static | Hardware specification |
| **L545** | Sync & lag | Photodiode square size & rate | 3 cm $\times$ 3 cm, 1 kHz | — | — | ⚪ Static | Sensor specs |
| **L545** | Sync & lag | Cohort 1 mice & sessions | 2 mice (25 sessions) | `\statFigOneFiveDepthSessions` | 25 | 🟢 Match | Cohort sample size |
| **L545** | Sync & lag | Frame delivery rate at 144 Hz | $91.5 \pm 0.9\%$ | `\statSuppFrameTargetPct` | $\$91.5 \pm 0.9\%\$$ | 🟢 Match | Exact match |
| **L545** | Sync & lag | Mean effective refresh rate | above $127 \pm 2\text{ Hz}$ | `\statSuppFrameRate` | $\$127 \pm 2~\mathrm{Hz}\$$ | 🟢 Match | Exact match |
| **L545** | Sync & lag | Cohort 2 mice & sessions | 5 mice (60 sessions) | `\statFigOneEightDepthSessions` | 59 sessions | 🔴 Outdated | 25 + 59 = 84 total (paper says 60) |
| **L546** | Sync & lag | Mean measured display lag | $26.6 \pm 1.4\text{ ms}$ | `\statSuppDisplayLag` | $\$26.4 \pm 0.1~\mathrm{ms}\$$ | 🔴 Outdated | Refined photodiode analysis |
| **L547** | Drifting gratings | Session & mouse sample size | $N = 6\text{ sessions}$, 2 mice | `\statRevGratSessions`<br>`\statRevGratMice` | 6<br>2 | 🟢 Match | Exact match |
| **L547** | Drifting gratings | Directions, SFs, TFs | 8 directions, 6 SFs, 6 TFs | — | — | ⚪ Static | Stimulus parameter set |
| **L547** | Drifting gratings | Total conditions & duration | 288 conditions, 4 s, 5 reps | — | — | ⚪ Static | Stimulus protocol |
| **L550** | Training | Training duration & trials | 2–4 weeks, 3 of 5–15 min, $\sim$2/min | — | — | ⚪ Static | Training regimen |
| **L554** | 2P imaging | Cortical depth, objective, NA | 80–300 $\mu$m, 16x, NA 0.8 | — | — | ⚪ Static | Optical hardware parameters |
| **L554** | 2P imaging | Laser wavelength & power | 930 nm at 70–100 mW | — | — | ⚪ Static | Laser parameters |
| **L554** | 2P imaging | Emission filter, pixels, FOV | 525/50, $1024 \times 1024$, $\sim 661\ \mu\text{m}$ | — | — | ⚪ Static | Imaging FOV parameters |
| **L557** | Widefield | Objective, LEDs, filters | 85 mm, 470 nm / 405 nm, 525/50, 60 Hz | — | — | ⚪ Static | Widefield hardware parameters |
| **L559** | Widefield | Patch size, SF, TF, reps | 20 deg, 0.2 cpd, 2 Hz, 50 reps | — | — | ⚪ Static | Widefield stimulus parameters |
| **L567** | Eye tracking | Sessions and mice analyzed | $N = 16\text{ sessions}$, 2 mice | `\statSuppEyeSessions`<br>`\statSuppEyeMice` | 16<br>2 | 🟢 Match | Exact match |
| **L567** | Eye tracking | Key points & outlier criteria | 12 points, $<3$ SD, DLC $<0.88$ | — | — | ⚪ Static | Tracking quality criteria |
| **L571** | Eye tracking | Velocity filter & saccade threshold | 3 frames window, 5 frames, 75 deg/s | — | — | ⚪ Static | Saccade detection parameters |
| **L574** | Preprocessing | Detrending & neuropil formula | 20th pct (900 frames), factor 0.7 | — | — | ⚪ Static | Suite2p / dF/F parameters |
| **L583** | Depth selectivity | Parameter bounds & CV | $\ln 2$ to $\ln 2000$, $\sigma^2 > 0.5$, 5-fold CV | — | — | ⚪ Static | Model fitting constraints |
| **L585** | Depth selectivity | Significance threshold | one-sided Spearman $p < 0.05$ | — | — | ⚪ Static | Significance criterion |
| **L587** | Depth selectivity | Near / far edge neuron cutoff | $<\ln 2 + 10^{-4}$, $>\ln 2000 - 10^{-4}$ | — | — | ⚪ Static | Classification boundary |
| **L590** | RS/OF tuning | Smoothing kernel & speed cutoff | 1 bin width (10 cm/s, 0.691), $>1\text{ cm/s}$ | — | — | ⚪ Static | Smoothing parameters |
| **L626** | RS/OF tuning | Parameter bounds (5 models) | OF 0.03–3000 deg/s, RS 0.5–500 cm/s | — | — | ⚪ Static | Model fitting constraints |
| **L634** | RS/OF tuning | Model comparison cell count | 21,318 of 27,094 (84 sess, 7 mice) | `\statFigTwoModelNeurons`<br>`\statFigOneDepthNeurons` | 21,457 of 27,127 | 🔴 Outdated | Re-run updated population |
| **L636** | RS/OF tuning | Bootstrap iterations | 20,000 resamples | — | — | ⚪ Static | Bootstrap parameter |
| **L638** | RS/OF tuning | Conjunctive model null threshold | $R^2 > 0.002$ | `\statFigTwoConjunctiveThreshold` | 0.0016 | 🔴 Outdated | Newly added exact threshold |
| **L638** | RS/OF tuning | Closed-loop conjunctive fits | 17,953 of 27,094 | `\statFigTwoClosedLoopConjunctiveNeurons` | 18,049 | 🔴 Outdated | Re-run updated population |
| **L641** | RS/OF tuning | Open-loop conjunctive fits | 3,948 of 9,429 (34 sess, 7 mice) | `\statFigTwoOpenLoopNeurons`<br>`\statFigTwoOpenLoopTotalDepthNeurons` | 3,961 of 9,462 | 🔴 Outdated | Re-run updated population |
| **L647** | Motorized wheel | Plateau speed tolerance | within 30% of target in 0.5 s window | — | — | ⚪ Static | Analysis filter criteria |
| **L649** | Motorized wheel | Fit parameter bounds | RS 2.3–101 cm/s, OF 0.61–1688 deg/s | — | — | ⚪ Static | Fit bounds ($e^{0.5}$ pad) |
| **L649** | Motorized wheel | Significant conjunctive fits | $n = 285\text{ neurons}$ (4 sess, 4 mice) | `\statFigThreePolarNeurons` | 285 | 🟢 Match | Exact match |
| **L649** | Motorized wheel | Elongation cutoff fraction | $\sigma_1/\sigma_2 > 1.4$ (266/285 neurons) | `\statFigThreeElongationCutoffRatio` | 266/285 | 🟢 Match | Exact match |
| **L649** | Motorized wheel | Von Mises components & BIC | $k = 3$; $3^\circ$ (14%), $90^\circ$ (11%), $43^\circ$ (76%) | `\statFigThreeVonMisesK`<br>`\statFigThreeVonMisesComponents` | 3<br>Components string | 🟢 Match | Exact match |
| **L651** | Motorized simulation | Indicator kernel decay constants | $\tau_\mathrm{rise} = 0.15\text{ s}$, $\tau_\mathrm{decay} = 2.0\text{ s}$ | — | — | ⚪ Static | Simulation model constants |
| **L651** | Motorized simulation | Free locomotion synthetic elongation | 297/316 (94% with elongation $>1.4$) | `\statSuppSimulFreeElongatedRatio`<br>`\statSuppSimulFreeElongatedPct` | 297/316<br>94% | 🟢 Match | Exact match |
| **L651** | Motorized simulation | Motorized synthetic elongation | 20/295 (7% with elongation $>1.4$) | `\statSuppSimulMotorElongatedRatio`<br>`\statSuppSimulMotorElongatedPct` | 20/295<br>7% | 🟢 Match | Exact match |
| **L651** | Motorized simulation | Real data motorized elongation | 266/285 (93% with elongation $>1.4$) | `\statFigThreeElongationCutoffRatio`<br>`\statFigThreeElongationCutoffPct` | 266/285<br>93% | 🟢 Match | Exact match |
| **L659** | Receptive fields | Deconvolution tau constants | $\tau = 1.1\text{ s}$ (6s), $0.25\text{ s}$ (6f) | — | — | ⚪ Static | Deconvolution parameter |
| **L659** | Receptive fields | Time shift optimization | 2 imaging frames (131 ms), sweep 0–10 | — | — | ⚪ Static | Lag optimization constants |
| **L707** | Receptive fields | Regularization grid search | $13 \times 13$ grid, 2.5 to 10,240, 5-fold CV | — | — | ⚪ Static | Hyperparameter search |
| **L709** | Receptive fields | Significance threshold | 6 standard deviations above null (~5%) | — | — | ⚪ Static | Null rejection threshold |
| **L725** | Cortical location | Spatial Gaussian kernel | $113\ \mu\text{m}$ | — | — | ⚪ Static | Smoothing filter width |
| **L741** | Cortical location | Mahalanobis Chi-squared DoF | 2 degrees of freedom | — | — | ⚪ Static | Theoretical distribution DoF |
| **L755** | Gratings analysis | Direction selectivity index cutoff | $\mathrm{DSI} > 0.3$ ($R_\mathrm{pref}/R_\mathrm{null} \ge 1.86$) | — | — | ⚪ Static | Classification threshold |
| **L762** | Gratings analysis | Tuning CV & significance | 5-fold CV (4 train, 1 test), $p < 0.05$ | — | — | ⚪ Static | Model cross-validation |
| **L764** | Gratings analysis | Direction permutation count | 20,000 permutations, 8 direction bins | — | — | ⚪ Static | Permutation test parameter |
| **L768** | Gratings analysis | Grating correlation bootstrap | 95% CI, 10,000 resamples, 5 pairs | — | — | ⚪ Static | Bootstrap parameter |
| **L770** | Gratings analysis | Running speed modulation bins | $<2.8\text{ cm/s}$ and 5 running bins | — | — | ⚪ Static | Speed bin boundaries |
| **L770** | Gratings analysis | Modulated cell counts (Methods) | TF: $N = 218\text{ cells}$, SF: $N = 342\text{ cells}$ | `\statRevGratSpeedCurvesTfN`<br>`\statRevGratSpeedCurvesSfN` | 218<br>342 | 🟢 Match | Exact match in Methods text |
| **L773** | Decoding | Data split fractions | 64% train, 16% validation, 20% test | — | — | ⚪ Static | Machine learning split |

---

### Section 6: Supplementary Figures (Figure S1 to Figure S10)

| Line | Figure | Caption Text / Snippet | Value in Paper | Macro in Stats File | Value in Stats File | Status | Notes |
| :---: | :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **L839** | Fig S2A-B | Corridor distance & cohort sessions | 5 ($N = 25\text{ sess}$) or 8 ($N = 59\text{ sess}$), 3 m | `\statFigOneFiveDepthSessions`<br>`\statFigOneEightDepthSessions` | 25<br>59 | 🟢 Match | Exact match |
| **L845** | Fig S2K-L | Eye tracking sessions & mice | $N = 16\text{ sessions}$, 2 mice | `\statSuppEyeSessions`<br>`\statSuppEyeMice` | 16<br>2 | 🟢 Match | Exact match |
| **L862** | Fig S3A,D,G | ISI distance | 3 m | — | — | ⚪ Static | Corridor boundary |
| **L864** | Fig S3C,F,I | Running speed tuning thresholds | $>5\text{ cm/s}$, $<5\text{ cm/s}$ (preceding 1 s) | — | — | ⚪ Static | Behavioral classification criterion |
| **L865** | Fig S3J | GCaMP6f cohort sessions & mice | $N = 70\text{ sessions}$, 6 mice | `\statSuppDepthPopSixfSessions`<br>`\statSuppDepthPopSixfMice` | 70<br>6 | 🟢 Match | Exact match |
| **L865** | Fig S3J | GCaMP6f median % depth-tuned | 24.6% | `\statSuppDepthPopSixfMedianPctDepthTuned` | 24.6% | 🟢 Match | Exact match |
| **L865** | Fig S3J | GCaMP6s cohort sessions & mice | $N = 14\text{ sessions}$, 1 mouse | `\statSuppDepthPopSixsSessions`<br>`\statSuppDepthPopSixsMice` | 14<br>1 | 🟢 Match | Exact match |
| **L865** | Fig S3J | GCaMP6s median % depth-tuned | 80.3% | `\statSuppDepthPopSixsMedianPctDepthTuned` | 80.3% | 🟢 Match | Exact match |
| **L866** | Fig S3K | 5-depth sessions, mice, neurons | $N = 25\text{ sess}$, 2 mice, 19,762 cells | `\statSuppDepthPopFiveDepthSessions`<br>`\statSuppDepthPopFiveDepthMice`<br>`\statSuppDepthPopFiveDepthNeurons` | 25<br>2<br>19,762 | 🟢 Match | Exact match |
| **L866** | Fig S3L | 8-depth sessions, mice, neurons | $N = 59\text{ sess}$, 5 mice, **7,332 cells** | `\statSuppDepthPopEightDepthSessions`<br>`\statSuppDepthPopEightDepthMice`<br>`\statSuppDepthPopEightDepthNeurons` | 59<br>5<br>**7,365** | 🔴 Outdated | 7,332 vs 7,365 (+33 cells) |
| **L877** | Fig S4C | Size control neurons, sessions, mice | 314 neurons from 3 sessions in 2 mice | `\statSuppSizeControlNeurons`<br>`\statSuppSizeControlSessions`<br>`\statSuppSizeControlMice` | 314<br>3<br>2 | 🟢 Match | Exact match |
| **L877** | Fig S4C | 5 vs 10 deg: ratio, p, r, p_corr | ratio = 1.05, $p = 0.588$, $r = 0.769$, $p < 0.0001$ | `\statSuppSizeControlRatioFiveTen`<br>`\statSuppSizeControlRatioPvalFiveTen`<br>`\statSuppSizeControlCorrRFiveTen`<br>`\statSuppSizeControlCorrPvalFiveTen` | 1.05<br>$\$p = 0.588\$<br>$\$r = 0.769\$<br>$\$p < 0.0001\$$ | 🟢 Match | Exact match |
| **L877** | Fig S4C | 5 vs 20 deg: ratio, p, r, p_corr | ratio = 1.00, $p = 0.812$, $r = 0.754$, $p < 0.0001$ | `\statSuppSizeControlRatioFiveTwenty`<br>`\statSuppSizeControlRatioPvalFiveTwenty`<br>`\statSuppSizeControlCorrRFiveTwenty`<br>`\statSuppSizeControlCorrPvalFiveTwenty` | 1.00<br>$\$p = 0.812\$<br>$\$r = 0.754\$<br>$\$p < 0.0001\$$ | 🟢 Match | Exact match |
| **L877** | Fig S4C | 10 vs 20 deg: ratio, p, r, p_corr | ratio = 1.00, $p = 0.951$, $r = 0.766$, $p < 0.0001$ | `\statSuppSizeControlRatioTenTwenty`<br>`\statSuppSizeControlRatioPvalTenTwenty`<br>`\statSuppSizeControlCorrRTenTwenty`<br>`\statSuppSizeControlCorrPvalTenTwenty` | 1.00<br>$\$p = 0.951\$<br>$\$r = 0.766\$<br>$\$p < 0.0001\$$ | 🟢 Match | Exact match |
| **L918** | Fig S7C | Free simulation neurons & elongation | $n = 316\text{ neurons}$, $297/316$ (94%) | `\statSuppSimulFreeNeurons`<br>`\statSuppSimulFreeElongatedRatio`<br>`\statSuppSimulFreeElongatedPct` | 316<br>297/316<br>94% | 🟢 Match | Exact match |
| **L918** | Fig S7D | Motorized simulation neurons & elongation | $n = 295\text{ neurons}$, $20/295$ (7%) | `\statSuppSimulMotorNeurons`<br>`\statSuppSimulMotorElongatedRatio`<br>`\statSuppSimulMotorElongatedPct` | 295<br>20/295<br>7% | 🟢 Match | Exact match |
| **L918** | Fig S7C-D| Simulation sessions and mice | 4 sessions in 4 mice | `\statSuppSimulSessions`<br>`\statSuppSimulMice` | 4<br>4 | 🟢 Match | Exact match |
| **L936** | Fig S8A-C | Virtual depths displayed | 8 depths (5 to 640 cm) | — | — | ⚪ Static | Stimulus levels list |
| **L937** | Fig S8D | RF correlation comparison sample size | **$N = 895\text{ neurons}$** (5 sess, 4 mice) | `\statSuppEightRFCorrNeurons`<br>`\statSuppEightRFCorrSessions`<br>`\statSuppEightRFCorrMice` | **1,009 neurons**<br>5 sess, 4 mice | 🔴 Outdated | Spike RF update expanded sample to 1,009 |
| **L937** | Fig S8D | Contra median correlation | **$r = 0.286$** | `\statSuppEightRFCorrMedianContra` | **$\$r = 0.334\$$** | 🔴 Outdated | Spike RF update: 0.334 |
| **L937** | Fig S8D | Ipsi median correlation | **$r = 0.042$** | `\statSuppEightRFCorrMedianIpsi` | **$\$r = 0.031\$$** | 🔴 Outdated | Spike RF update: 0.031 |
| **L937** | Fig S8D | Difference p-value | **$p = 2.18 \times 10^{-124}$** | `\statSuppEightRFCorrPval` | **$\$p < 0.0001\$$** | 🔴 Outdated | Hierarchical bootstrap over mice/sessions: $p < 0.0001$ |
| **L938** | Fig S8E | Median % significant RFs across sessions | 46.2%, $N = 89\text{ sessions}$ | `\statSuppEightSigRFMedianTotal`<br>`\statSuppEightSigRFSessionsTotal` | 46.2%<br>89 | 🟢 Match | Exact match |
| **L939** | Fig S8F | Multi vs single peak depth neurons | **$N = 603\text{ neurons}$** | `\statSuppEightPeakDepthNeurons` | **1,009 neurons** | 🔴 Outdated | Spike RF update expanded sample to 1,009 |
| **L939** | Fig S8F | Multi vs single peak depth Spearman r | **$r = 0.587$** | `\statSuppEightPeakDepthR` | **$\$r = 0.561\$$** | 🔴 Outdated | Spike RF update: 0.561 |
| **L939** | Fig S8F | Multi vs single peak depth p-value | **$p = 4.05 \times 10^{-57}$** | `\statSuppEightPeakDepthPval` | **$\$p < 0.0001\$$** | 🔴 Outdated | Hierarchical bootstrap over mice/sessions: $p < 0.0001$ |
| **L939** | Fig S8F | Multi vs single peak depth median ratio | **0.865** | `\statSuppEightPeakDepthMedianRatio` | **0.939** | 🔴 Outdated | Spike RF update: 0.939 |
| **L939** | Fig S8F | Multi vs single peak depth ratio IQR | **0.631--1.290** | `\statSuppEightPeakDepthRatioIQR` | **0.669--1.506** | 🔴 Outdated | Spike RF update: 0.669--1.506 |
| **L954** | Fig S9A | Retinotopic patch locations & scale | 30 and 90 deg azimuth, 1 mm scale | — | — | ⚪ Static | Visual stimulus geometry |
| **L955** | Fig S9B-D | Spatial RF distribution neurons & sess | **$N = 14,173\text{ neurons}$** (84 sess) | `\statSuppNineRFNeuronsTotal`<br>`\statSuppNineRFSessionsTotal` | **16,974 neurons**<br>84 sessions | 🔴 Outdated | Spike RF update expanded sample to 16,974 |
| **L956** | Fig S9E | Median corrected depth neurons | **$N = 14,173\text{ neurons}$** (min 20) | `\statSuppNineRFNeuronsTotal` | **16,974 neurons** | 🔴 Outdated | Spike RF update: 16,974 |
| **L957** | Fig S9F | Corrected Near depth neurons ($<20\text{ cm}$) | **$N = 2,476\text{ neurons}$** | `\statSuppNineNearCorrected` | **3,315 neurons** | 🔴 Outdated | Spike RF update: 3,315 |
| **L957** | Fig S9F | Corrected Mid depth neurons ($20\text{--}100\text{ cm}$) | **$N = 5,832\text{ neurons}$** | `\statSuppNineMidCorrected` | **7,288 neurons** | 🔴 Outdated | Spike RF update: 7,288 |
| **L957** | Fig S9F | Corrected Far depth neurons ($>100\text{ cm}$) | **$N = 5,865\text{ neurons}$** | `\statSuppNineFarCorrected` | **6,371 neurons** | 🔴 Outdated | Spike RF update: 6,371 |
| **L957** | Fig S9F | Grid resolution & plotting jitter | 5 degree resolution, 2 degrees jitter | — | — | ⚪ Static | Plotting parameters |
| **L958** | Fig S9G-H| Uncorrected Near depth neurons | **$N = 3,078\text{ neurons}$** | `\statSuppNineNearUncorrected` | **4,132 neurons** | 🔴 Outdated | Spike RF update: 4,132 |
| **L958** | Fig S9G-H| Uncorrected Mid depth neurons | **$N = 6,744\text{ neurons}$** | `\statSuppNineMidUncorrected` | **8,100 neurons** | 🔴 Outdated | Spike RF update: 8,100 |
| **L958** | Fig S9G-H| Uncorrected Far depth neurons | **$N = 4,351\text{ neurons}$** | `\statSuppNineFarUncorrected` | **4,742 neurons** | 🔴 Outdated | Spike RF update: 4,742 |
| **L977** | Fig S10D | Grating sessions and mice | 6 sessions in 2 mice | `\statRevGratSessions`<br>`\statRevGratMice` | 6<br>2 | 🟢 Match | Exact match |
| **L977** | Fig S10D | Observed vs expected overlap | 390 observed vs 345.2 expected | `\statRevGratBothN`<br>`\statRevGratBothExpectedN` | 390<br>345 (raw 345.16) | 🟢 Match | Exact match |
| **L977** | Fig S10D | Mantel-Haenszel overlap p-value | $p < 0.0001$ | `\statRevGratOverlapP` | $\$p < 0.0001\$$ | 🟢 Match | Exact match |
| **L978** | Fig S10E | Depth-selective DSI median & $N$ | median = 0.14, $N = 390$ | `\statRevGratDsiMedianDepth`<br>`\statRevGratBothN` | 0.14<br>390 | 🟢 Match | Exact match |
| **L978** | Fig S10E | Other SF/TF-tuned DSI median & $N$ | median = 0.19, $N = 1,052$ | `\statRevGratDsiMedianOther`<br>`\statRevGratOtherSftfNeurons` | 0.19<br>1,052 | 🟢 Match | Exact match |
| **L978** | Fig S10E | DSI permutation test p-value | $p = 0.73$ | `\statRevGratDsiP` | $\$p = 0.725\$$ | 🟢 Match | $p = 0.725$ rounds to 0.73 |
| **L979** | Fig S10F | Direction-selective cells (DSI $>0.3$) | Depth: $N = 130$, Other: $N = 393$ | `\statRevGratDirTunedN`<br>`\statRevGratDirTunedOtherN` | 130<br>393 | 🟢 Match | Exact match |
| **L979** | Fig S10F | Direction chi-square p-value | $p = 0.056$ | `\statRevGratDirP` | $\$p = 0.0561\$$ | 🟢 Match | $p = 0.0561$ rounds to 0.056 |
| **L981** | Fig S10G | Preferred grating speed vs depth | $N = 390$, $\rho = -0.16$, $p_{\mathrm{Holm}} = 0.0276$ | `\statRevGratSpeedDepthN`<br>`\statRevGratSpeedDepthRho`<br>`\statRevGratSpeedDepthPHolm` | 390<br>-0.16<br>$\$p = 0.0276\$$ | 🟢 Match | Exact match |
| **L982** | Fig S10H | Preferred TF vs depth | $N = 390$, $\rho = 0.06$, $p_{\mathrm{Holm}} = 1.0$ | `\statRevGratTfDepthN`<br>`\statRevGratTfDepthRho`<br>`\statRevGratTfDepthPHolm` | 390<br>0.06<br>$\$p = 1\$$ | 🟢 Match | Exact match |
| **L983** | Fig S10I | Preferred SF vs depth | $N = 390$, $\rho = 0.36$, $p_{\mathrm{Holm}} = 0.0291$ | `\statRevGratSfDepthN`<br>`\statRevGratSfDepthRho`<br>`\statRevGratSfDepthPHolm` | 390<br>0.36<br>$\$p = 0.0291\$$ | 🟢 Match | Exact match |
| **L984** | Fig S10J | Preferred SF vs optic flow | **$N = 333$**, $\rho = -0.41$, **$p_{\mathrm{Holm}} = 0.0060$** | `\statRevGratSfOfN`<br>`\statRevGratSfOfRho`<br>`\statRevGratSfOfPHolm` | **330**<br>-0.41<br>**$\$p = 0.0025\$$** | 🔴 Outdated | Re-run updated $N=330$ and $p=0.0025$ |
| **L987** | Fig S10M | Speed-modulated TF curves population | **$N = 142\text{ neurons}$** | `\statRevGratSpeedCurvesTfN` | **218 neurons** | 🔴 Outdated | Caption has 142; Methods & stats have 218! |
| **L989** | Fig S10O | Speed-modulated SF curves population | **$N = 176\text{ neurons}$** | `\statRevGratSpeedCurvesSfN` | **342 neurons** | 🔴 Outdated | Caption has 176; Methods & stats have 342! |

---

## 5. Numbers in the Stats Manifest Not Cited in the Manuscript

The stats manifests compile a total of **62 metrics** that are calculated by figure notebooks but are currently **not directly cited in the text or captions** of `v1_depth_map.tex`. These metrics are cataloged below by theme and notebook, along with the scientific reason they are not currently in prose:

### 5.1. Figure 2: Pairwise RS/OF Model Comparison Tests (10 metrics)
*Source Notebook: `figure2_rsof_integration.ipynb`*

In the main text (line 202), the manuscript summarizes the model comparison as a single consolidated claim:
> *"instead both the additive and conjunctive model outperformed the three others (Figure 2E; all $p < 0.0001$)"* (cited via `\statFigTwoModelComparisonPval`).

The 10 underlying individual pairwise hierarchical bootstrap tests are computed and tracked in the stats manifest:

| Macro Name | Formatted Value | Raw Value | Statistical Test / Description | Why Not Cited in Manuscript |
| :--- | :---: | :---: | :--- | :--- |
| `\statFigTwoPvalRunningSpeedVsOpticFlow` | `$p < 0.0001$` | 0.00005 | Running speed vs Optic flow: median diff -0.058, 95% CI [-0.095, -0.028] | Subsumed by parent claim |
| `\statFigTwoPvalRunningSpeedVsRsOfRatio` | `$p = 0.0001$` | 0.0001 | Running speed vs RS/OF: median diff -0.071, 95% CI [-0.122, -0.039] | Subsumed by parent claim |
| `\statFigTwoPvalRunningSpeedVsAdditive` | `$p < 0.0001$` | 0.00005 | Running speed vs Additive: median diff -0.318, 95% CI [-0.368, -0.284] | Subsumed by parent claim |
| `\statFigTwoPvalRunningSpeedVsConjunctive` | `$p < 0.0001$` | 0.00005 | Running speed vs Conjunctive: median diff -0.364, 95% CI [-0.432, -0.327] | Subsumed by parent claim |
| `\statFigTwoPvalOpticFlowVsRsOfRatio` | `$p = 0.438$` | 0.4381 | Optic flow vs RS/OF: median diff -0.012, 95% CI [-0.053, 0.017] | Non-significant pair; not singled out in text |
| `\statFigTwoPvalOpticFlowVsAdditive` | `$p < 0.0001$` | 0.00005 | Optic flow vs Additive: median diff -0.261, 95% CI [-0.326, -0.199] | Subsumed by parent claim |
| `\statFigTwoPvalOpticFlowVsConjunctive` | `$p < 0.0001$` | 0.00005 | Optic flow vs Conjunctive: median diff -0.302, 95% CI [-0.392, -0.264] | Subsumed by parent claim |
| `\statFigTwoPvalRsOfRatioVsAdditive` | `$p < 0.0001$` | 0.00005 | RS/OF vs Additive: median diff -0.247, 95% CI [-0.316, -0.177] | Subsumed by parent claim |
| `\statFigTwoPvalRsOfRatioVsConjunctive` | `$p < 0.0001$` | 0.00005 | RS/OF vs Conjunctive: median diff -0.292, 95% CI [-0.363, -0.239] | Subsumed by parent claim |
| `\statFigTwoPvalAdditiveVsConjunctive` | `$p = 0.206$` | 0.2057 | Additive vs Conjunctive: median diff -0.046, 95% CI [-0.120, 0.027] | Additive vs Conjunctive comparison not detailed in text |

---

### 5.2. Figure 2 & Figure 4: Auxiliary Behavioral & Anatomical Metrics (4 metrics)
*Source Notebooks: `figure2_openloop.ipynb`, `figure4_receptive_fields.ipynb`*

| Macro Name | Formatted Value | Raw Value | Description | Why Not Cited in Manuscript |
| :--- | :---: | :---: | :--- | :--- |
| `\statFigTwoRSRatioMedian` | `1.0454` | 1.04537 | Hierarchical bootstrap median of preferred RS closed/open ratio | Manuscript highlights correlation ($r=0.470$) and amp ratio ($0.98$) |
| `\statFigTwoRSRatioPval` | `$p = 0.209$` | 0.2087 | Hierarchical bootstrap p-value of preferred RS closed/open ratio | Supplementary to correlation test |
| `\statFigTwoDecoderFiveDepthSessions` | `7` | 7 | Sessions with 5 depths in SVM decoder comparison | Manuscript cites total sessions ($34$) and 8-depth matrices ($27$) |
| `\statFigFourRFDistPairs` | `6,536,536` | 6536536 | Neuron pairs $>10\ \mu\text{m}$ apart in pairwise distance binning | Caption cites cell count ($16,974$) and session count ($84$), not pairs |

---

### 5.3. Figure S3: Preferred Virtual Depth Distributions & Cohort Differences (6 metrics)
*Source Notebook: `figsupp3_depth_pop.ipynb`*

These metrics characterize the distribution of preferred depths shown in the histograms of Fig S3K (5 depths) and Fig S3L (8 depths):

| Macro Name | Formatted Value | Raw Value | Description | Why Not Cited in Manuscript |
| :--- | :---: | :---: | :--- | :--- |
| `\statSuppDepthPopFiveDepthMedianPrefDepth` | `40.9` | 40.91 | Median preferred depth (cm), 5-depth sessions (Fig S3K) | Distribution is shown visually in histogram; median not in prose |
| `\statSuppDepthPopFiveDepthPrefDepthIQR` | `18.7--99.6` | 18.7–99.6 | Interquartile range of preferred depth (cm), 5-depth sessions | Displayed visually in histogram; IQR not in prose |
| `\statSuppDepthPopEightDepthMedianPrefDepth` | `51.1` | 51.09 | Median preferred depth (cm), 8-depth sessions (Fig S3L) | Distribution is shown visually in histogram; median not in prose |
| `\statSuppDepthPopEightDepthPrefDepthIQR` | `11.8--239.5` | 11.8–239.5 | Interquartile range of preferred depth (cm), 8-depth sessions | Displayed visually in histogram; IQR not in prose |
| `\statSuppDepthPopKsStat` | `$D = 0.160$` | 0.1598 | KS statistic, log preferred depth of 5- vs 8-depth sessions | Statistical test comparing the two cohorts; not written in text |
| `\statSuppDepthPopKsPval` | `$p < 0.0001$` | $3.6 \times 10^{-120}$ | KS test p-value comparing 5- vs 8-depth cohorts | Statistical test comparing the two cohorts; not written in text |

---

### 5.4. Figure S5: Optic Flow & Running Speed Tuning Across Speed Bins (10 metrics)
*Source Notebook: `figsupp5_rsof.ipynb`*

> [!NOTE]
> Panels S5I and S5J were present in an earlier draft of `v1_depth_map.tex` but are currently **commented out** (lines 894–895). If restored, these macros are fully compiled and ready.

| Macro Name | Formatted Value | Raw Value | Description | Why Not Cited in Manuscript |
| :--- | :---: | :---: | :--- | :--- |
| `\statSuppRSOFCorrOFDepthR` | `$r = -0.595$` | -0.5948 | Correlation between preferred depth and optic flow speed (Fig S5I) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrOFDepthPval` | `$p < 0.0001$` | 0.0 | P-value of correlation between preferred depth and optic flow speed | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthAllR` | `$r = -0.056$` | -0.0556 | Correlation between preferred depth and running speed (all neurons, Fig S5J) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthAllPval` | `$p = 0.557$` | 0.5573 | P-value of correlation between preferred depth and running speed (all) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthBinOneR` | `$r = 0.506$` | 0.5058 | Correlation between preferred depth and running speed (OF 1–10 deg/s) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthBinOnePval` | `$p < 0.0001$` | 0.0 | P-value of correlation (OF 1–10 deg/s) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthBinTwoR` | `$r = 0.396$` | 0.3955 | Correlation between preferred depth and running speed (OF 10–100 deg/s) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthBinTwoPval` | `$p < 0.0001$` | 0.0 | P-value of correlation (OF 10–100 deg/s) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthBinThreeR` | `$r = 0.361$` | 0.3606 | Correlation between preferred depth and running speed (OF 100–1000 deg/s) | Caption lines 894-895 commented out in tex |
| `\statSuppRSOFCorrRSDepthBinThreePval` | `$p < 0.0001$` | 0.0 | P-value of correlation (OF 100–1000 deg/s) | Caption lines 894-895 commented out in tex |

---

### 5.5. Figure S8: Single vs Multi-Depth RF Sub-Proportions & Cell Counts (7 metrics)
*Source Notebook: `figsupp8_multidepth_receptive_fields.ipynb`*

| Macro Name | Formatted Value | Raw Value | Description | Why Not Cited in Manuscript |
| :--- | :---: | :---: | :--- | :--- |
| `\statSuppEightSigRFSessionsSingle` | `84` | 84 | Number of single-depth sessions in RF analysis (Fig S8E) | Caption S8E cites combined total ($89$ sessions) |
| `\statSuppEightSigRFMedianSingle` | `46.4\%` | 0.4644 | Median proportion of depth-tuned neurons with significant RFs (single-depth) | Caption S8E cites combined median ($46.2\%$) |
| `\statSuppEightSigRFSessionsMulti` | `5` | 5 | Number of multi-depth sessions in RF analysis (Fig S8E) | Caption S8E cites combined total ($89$ sessions) |
| `\statSuppEightSigRFMedianMulti` | `44.9\%` | 0.4492 | Median proportion of depth-tuned neurons with significant RFs (multi-depth) | Caption S8E cites combined median ($46.2\%$) |
| `\statSuppEightSigRFMiceTotal` | `11` | 11 | Number of mice, single- and multi-depth sessions combined | Available for expanded methods text |
| `\statSuppEightSigRFNeuronsTotal` | `28,999` | 28999 | Total depth-tuned neurons across all 89 RF sessions | Available for expanded methods text |
| `\statSuppEightSigRFSigNeuronsTotal` | `17,983` | 17983 | Total depth-tuned neurons with significant RFs across all 89 RF sessions | Available for expanded methods text |

---

### 5.6. Figure S9: 3D RF Visual Space Gradient Tests (2 metrics)
*Source Notebook: `figsupp9_v1_depth_map.ipynb`*

| Macro Name | Formatted Value | Raw Value | Description | Why Not Cited in Manuscript |
| :--- | :---: | :---: | :--- | :--- |
| `\statSuppNineGradientPvalCorrected` | `$p = 4.90e-09$` | $4.90 \times 10^{-9}$ | 2D Z-test p-value of depth gradient across visual space (viewing-angle corrected) | In earlier draft (line 278, $p=5.67\times 10^{-5}$), currently commented out |
| `\statSuppNineGradientPvalUncorrected` | `$p = 0.0406$` | 0.04056 | 2D Z-test p-value of depth gradient across visual space (uncorrected) | In earlier draft (line 278, $p=0.00701$), currently commented out |

---

### 5.7. Figure S10: Drifting Gratings Marginal Counts, Uncorrected Tests & Control Pair (23 metrics)
*Source Notebook: `figsupp10_gratings.ipynb`*

| Macro Name | Formatted Value | Raw Value | Description | Why Not Cited in Manuscript |
| :--- | :---: | :---: | :--- | :--- |
| `\statRevGratDepthNeurons` | `693` | 693 | Depth-selective neurons in the 6 grating sessions | Caption S10D emphasizes overlap count ($390$) |
| `\statRevGratPctDepth` | `23\%` | 23.01 | Percentage of neurons depth selective in grating sessions | Marginal fraction; not written in text |
| `\statRevGratSftfNeurons` | `1,442` | 1442 | SF/TF-tuned neurons (cross-validated) | Caption S10D emphasizes overlap count ($390$) |
| `\statRevGratPctSftf` | `48\%` | 47.88 | Percentage of neurons SF/TF-tuned in grating sessions | Marginal fraction; not written in text |
| `\statRevGratRsofNeurons` | `495` | 495 | RS/OF-tuned neurons in grating sessions | Auxiliary characterization |
| `\statRevGratRsofThreshold` | `0.0016` | 0.00158 | RS/OF held-out $R^2$ threshold (95th percentile null) | Used for classification thresholding |
| `\statRevGratPctBoth` | `13\%` | 12.95 | Percentage of neurons depth-selective and SF/TF-tuned | Overlap count ($390$) cited rather than % |
| `\statRevGratPctBothExpected` | `11\%` | 11.46 | Expected percentage under independence | Expected count ($345.2$) cited rather than % |
| `\statRevGratOverlapOR` | `1.43` | 1.432 | Mantel-Haenszel pooled odds ratio over sessions | Test p-value ($p < 0.0001$) cited in caption |
| `\statRevGratSpeedDepthP` | `$p = 0.0069$` | 0.0069 | Raw bootstrap p-value: preferred speed vs depth | Caption cites Holm-corrected p-value ($p = 0.0276$) |
| `\statRevGratTfDepthP` | `$p = 0.566$` | 0.5664 | Raw bootstrap p-value: preferred TF vs depth | Caption cites Holm-corrected p-value ($p = 1.0$) |
| `\statRevGratSfDepthP` | `$p = 0.0097$` | 0.0097 | Raw bootstrap p-value: preferred SF vs depth | Caption cites Holm-corrected p-value ($p = 0.0291$) |
| `\statRevGratSfOfP` | `$p = 0.0005$` | 0.0005 | Raw bootstrap p-value: preferred SF vs optic flow | Caption cites Holm-corrected p-value ($p = 0.0025$) |
| `\statRevGratSfRsN` | `330` | 330 | Neurons, preferred SF vs preferred running speed | Control 5th pair; noted as "(not shown)" |
| `\statRevGratSfRsRho` | `-0.14` | -0.144 | Spearman rho, preferred SF vs preferred running speed | Control 5th pair; noted as "(not shown)" |
| `\statRevGratSfRsP` | `$p = 0.535$` | 0.5347 | Raw bootstrap p-value: preferred SF vs running speed | Control 5th pair; noted as "(not shown)" |
| `\statRevGratSfRsPHolm` | `$p = 1$` | 1.0 | Holm-corrected p-value: preferred SF vs running speed | Control 5th pair; noted as "(not shown)" |
| `\statRevGratDsiDiff` | `0.008` | 0.00829 | Within-session DSI difference (depth selective minus other) | Caption cites medians ($0.14$ vs $0.19$) and $p = 0.73$ |
| `\statRevGratSpeedCurvesTfSessions` | `6` | 6 | Sessions, population TF speed curves (M, O) | Caption cites neuron count ($N = 218$) |
| `\statRevGratSpeedCurvesTfMice` | `2` | 2 | Mice, population TF speed curves (M, O) | Caption cites neuron count ($N = 218$) |
| `\statRevGratSpeedCurvesSfSessions` | `6` | 6 | Sessions, population SF speed curves (N, P) | Caption cites neuron count ($N = 342$) |
| `\statRevGratSpeedCurvesSfMice` | `2` | 2 | Mice, population SF speed curves (N, P) | Caption cites neuron count ($N = 342$) |

---

## 6. Action Items & Step-by-Step Resolution Guide

When ready to update the manuscript, follow this ordered workflow:

### Step 1: Resolve Text Placeholders in Main Text (Line 282)
In `v1_depth_map.tex` around line 282, replace literal placeholders with macros:
```latex
% Line 282 replacement:
Preferred depth was negatively correlated with preferred grating speed (\statRevGratSpeedDepthRho{}, \statRevGratSpeedDepthPHolm{}, Figure \ref{sup:gratings}G).
Surprisingly this was not driven by a negative correlation with preferred temporal frequency (\statRevGratTfDepthRho{}, \statRevGratTfDepthPHolm{}, Figure \ref{sup:gratings}H)
but by a positive correlation with preferred spatial frequency (\statRevGratSfDepthRho{}, \statRevGratSfDepthPHolm{}, Figure \ref{sup:gratings}I),
itself arising from an anticorrelation with preferred optic flow speed (\statRevGratSfOfRho{}, \statRevGratSfOfPHolm{}, Figure \ref{sup:gratings}J).
```

### Step 2: Update Outdated Numbers in Overleaf Manuscript
Review the 17 flagged discrepancies in §3 above. Once verified, update the literal numbers in `v1_depth_map.tex` (or substitute the corresponding `\stat...{}` macros from `manuscript_stats.tex`).

### Step 3: Copy Updated Macro Definitions to Overleaf
Per Design Rule 5 in `notes/05_manuscript_statistics.md`, compile and copy `manuscript_stats.tex` across to Overleaf once verified:
```bash
cp /Users/blota/Documents/code/v1_depth_map/v1_depth_map/stats/manuscript_stats.tex \
   "/Users/blota/Library/CloudStorage/Dropbox-TheFrancisCrick/Antonin Blot/Apps/Overleaf/v1_depth_map Science re-submission/manuscript_stats.tex"
```
And add `\input{manuscript_stats.tex}` to the custom commands section of `v1_depth_map.tex`.
