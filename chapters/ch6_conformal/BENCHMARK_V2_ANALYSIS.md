# Chapter 7, Experiment 2: probabilistic benchmark analysis (v2, superseded by v4)

Analysis of the first aligned probabilistic-benchmark run, before the partial rerun of the QR, FPQR, BNN and Laplace heads. Analysed with `shared/ch7_analyse_results.py` (decisive metrics: per-fold MAE for point accuracy, per-fold Winkler score (WS) for intervals; DM test with Benjamini-Hochberg adjustment and Model Confidence Set (MCS) at alpha = 0.05; CWE, PICP, NMPI diagnostics).

## Runs

| dataset | run | commit | conditions |
| --- | --- | --- | --- |
| MLVS-PT | `20261002_072839…_f25312` | `f25312` | mlpgam_crc, mlpgam_qr, mlpgam_fpqr, mlpgam_bnn, mlpgam_laplace (all succeeded) |
| SPS-UK (`MLVS-UK`) | `20261002_091324…_717efa` | `717efa` | same five (all succeeded) |

Configuration: 10 folds, 4-month test windows, 1-month stride, 2-month validation, 3-month calibration, daily forecast origins (stride 48), 100 epochs, shared MLPGAM backbone per dataset, single seed.

## Problems found in this run

1. **Gate asymmetry.** Only `mlpgam_crc` trained with the feature gate. The probabilistic MLPGAM wrappers (QR, FPQR, BNN, Laplace and the other parametric heads) dropped the gate fields when building the network, so their `use_gate=True` had no effect. Fixed in `twiga.models.nn.mlpgam_model.mlpgam_gate_kwargs` (commit `c6214c3`). The architectural study measured the gate's effect on curated inputs at about 2% WS and not significant, so the bias is likely small, but the comparison was not like-for-like.
2. **FPQR failure in data-scarce folds (MLVS-PT).** FPQR's MAE fell from 15.2 in fold 1 to 5.5 in fold 5 and its PICP rose from 0.11 to 0.82 as the expanding training window grew; folds 6 to 10 were healthy (WS 21.5 to 23.5, PICP 0.92 to 0.97). Its head search had drawn a Kumaraswamy proposal with a spline QVN and no gradient clipping. FPQR now pins `proposal_type="softmax"`, `qvn_type="standard"`, `proposal_eps=1e-4`, `gradient_clip_val=1.0` (commit `689cca5`). SPS-UK FPQR was stable.
3. **Analysis fixes.** The analysis crashed headless (Agg backend now set), and the Kupiec test counted one row per metric type, inflating n two- to threefold; it now counts one interval row per window (n = 1,215 and 1,219).

These problems are being corrected by rerunning the four non-CRC heads (README, "Partial rerun"). The numbers below are from this run and will change for those heads.

## Results by hypothesis

### H1: the proposed framework matches or exceeds the alternatives on Winkler score — supported

| | MLVS-PT | SPS-UK |
| --- | --- | --- |
| mean WS | **CRC 25.90**, QR 26.34, Laplace 30.28, BNN 34.47, FPQR 59.49 | **CRC 1.691**, Laplace 1.751, FPQR 2.247, QR 2.656, BNN 3.229 |
| CRC significantly better than | Laplace (p = 0.003), BNN (p < 1e-6) | Laplace (p = 0.029), QR, FPQR, BNN (p < 1e-6) |
| not significantly different | QR (p = 0.63), FPQR (p = 0.07) | none |
| MCS | CRC, QR | CRC, Laplace |
| folds in which CRC ranks first | 6 of 10 (second in the rest) | 7 of 10 (second in the rest) |
| mean WS, folds 6 to 10 | CRC 21.05, Laplace 22.22, FPQR 22.46, QR 23.64, BNN 31.48 | CRC 1.725, Laplace 1.777, FPQR 2.219, QR 2.643, BNN 3.679 |

On MLVS-PT the defensible claim is a tie with QR: their 1.7% gap is within the 1 to 1.5% retraining noise measured by the architectural study, and this benchmark is single-seed.

### H2: the framework keeps competitive point accuracy — partly supported

- MLVS-PT: CRC MAE 4.18 against Laplace 4.03 (+3.8%, p = 1.8e-5) and QR 4.06 (p = 0.022); MCS {Laplace, QR}. CRC is significantly better than BNN (4.98) and FPQR (6.54).
- SPS-UK: MCS {CRC, Laplace}; CRC 0.298 against Laplace 0.291 (p = 0.039); CRC significantly better than QR, FPQR and BNN.
- Competitive (within 4% of the best, in the MCS on SPS-UK) but not the most accurate. Part of the gap is structural: CRC withholds the three-month calibration window from training, which cost up to 2.7% MAE in the architectural study.

### H3: horizon-wise calibration delivers near-nominal coverage — mostly supported

| miscoverage (nominal 10%) | MLVS-PT | SPS-UK |
| --- | --- | --- |
| CRC | 11.7% (Kupiec p = 0.056, not rejected) | 11.8% (p = 0.040, borderline rejection) |
| BNN | 8.7% (p = 0.13) | 3.6% (over-covers) |
| QR | 4.9% (over-covers) | 1.8% (over-covers) |
| FPQR | 28.3% (failed folds) | 4.5% (over-covers) |
| Laplace | 33.4% (under-covers) | 22.8% (under-covers) |

- CRC is closest to nominal on SPS-UK and second to BNN on MLVS-PT; it is the only head near nominal on both.
- CRC's shortfall is concentrated in folds whose test windows span November to April: MLVS-PT folds 1 and 2 (PICP 0.68, 0.74) and SPS-UK folds 6 and 7 (0.83, 0.84). This matches the calibration-to-test shift seen in the architectural study (H3) and points to seasonal exchangeability, not to horizon-wise calibration.
- Laplace is the most accurate point forecaster but badly under-covers (misspecified scale). QR and BNN buy coverage with much wider intervals (SPS-UK NMPI 0.34 and 0.41 against CRC 0.12).
- CWE ranks QR first (0.94 and 0.97), rewarding over-coverage, while WS ranks it fourth on SPS-UK; CWE stays a diagnostic.

### Proper scores

CRPS is not available for CRC (the engine scores conformal conditions on point and interval metrics only), so WS is the only proper score common to all five heads. Among the other four: MLVS-PT QR 2.83, Laplace 3.05, BNN 3.78, FPQR 4.90; SPS-UK Laplace 0.211, QR 0.246, FPQR 0.267, BNN 0.354.

## Implications for the chapter

1. Headline: the complete framework gives the best interval quality on both datasets (tied with QR on MLVS-PT) and the most consistently near-nominal coverage across them, while staying within 4% of the most accurate point forecaster.
2. State the single-seed limitation and, until the rerun is in, the gate asymmetry and FPQR's data-scarce-fold failure.
3. CRC cannot be ranked on CRPS; WS is the only proper score covering all heads.

## v4: reported results (FPQR excluded)

`probabilistic_benchmark_v4`, runs `20261002_201924…_689cca` (MLVS-PT) and `20261002_204451…_689cca` (SPS-UK), commit `689cca` (gate forwarded to every head; FPQR pins). These are the results reported in `Chapter-7.tex` (`chap7:result-benchmark`). Reproduce with:

```bash
uv run python chapters/ch6_conformal/shared/ch7_analyse_results.py --experiment probabilistic_benchmark \
  --dataset all --results-root <folder whose probabilistic_benchmark entry points at v4> --thesis-figures --copy-to-images
```

FPQR is run but excluded from the analysis (`EXPERIMENTS["probabilistic_benchmark"]["excluded"]`): both runs used one HPO trial, which resumed the persisted Optuna study holding trials from the unpinned configuration, so FPQR trained the pinned architecture (softmax proposal, standard QVN) with hyperparameters tuned for the Kumaraswamy/spline one. A clean re-tune needs the FPQR head cache and its Optuna journal deleted and 50 trials.

| | MLVS-PT | SPS-UK |
| --- | --- | --- |
| mean WS | **CRC 25.54**, QR 26.16, Laplace 29.55, BNN 34.62 | **CRC 1.665**, Laplace 1.750, QR 2.662, BNN 3.212 |
| WS MCS | CRC, QR (CRC vs QR p = 0.52) | CRC alone (vs Laplace p = 1.8e-5) |
| CRC ranks first | 6 of 10 folds (second in the 4 Nov to Mar folds) | 9 of 10 folds |
| MAE | Laplace 3.99, QR 4.04, CRC 4.09 (+2.4% vs Laplace, p = 0.030), BNN 5.03; MCS {Laplace, QR} | Laplace 0.286, CRC 0.290 (p = 0.086), QR 0.303, BNN 0.482; MCS {Laplace} |
| miscoverage (Kupiec p) | CRC 11.2% (0.17), BNN 9.0% (0.22), QR 4.9%, Laplace 33.3% | CRC 10.1% (0.92), BNN 3.9%, QR 1.6%, Laplace 22.4% |

All three hypotheses supported: best interval quality (significant on SPS-UK, tied with QR on MLVS-PT, where CRC leads over the April-onwards folds 20.64 vs Laplace 22.22 and QR 23.68 but trails QR over the winter folds 30.44 vs 28.63); competitive point accuracy (within 2.4% of the best); the only head whose coverage the Kupiec test accepts on both datasets. CRC under-covers in folds whose test windows begin in November/December (MLVS-PT PICP 0.682, 0.740; SPS-UK 0.849), a calibration-to-test seasonal shift. Enabling the gate on QR, BNN and Laplace changed their WS by at most 2.4% against v2.
