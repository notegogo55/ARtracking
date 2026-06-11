# Phase 5 evaluation report

Generated 2026-06-11 14:21 UTC. All splits chronological; thresholds frozen on validation tails before any test evaluation; seeds fixed (see configs/default.yaml).

## 1. Held-out time block: train SWAN-SF P3 -> test P4 (single evaluation)

| model        |    tss |    hss |     bss |   brier |   precision |   recall |   threshold |
|:-------------|-------:|-------:|--------:|--------:|------------:|---------:|------------:|
| lstm         | 0.8727 | 0.2515 | -2.1635 |  0.076  |      0.1648 |        1 |      0.5    |
| ensemble     | 0.8724 | 0.2509 | -0.1815 |  0.0284 |      0.1644 |        1 |      0.26   |
| holt_winters | 0.7832 | 0.1504 |  0.3041 |  0.0167 |      0.1038 |        1 |      0.0186 |
| climatology  | 0      | 0      |  0      |  0.024  |      0.0245 |        1 |      0      |

Figures: `figures/roc.png`, `figures/reliability.png`, per-model training curves under `outputs/forecast/holdout_p3p4/curves/`.

**Comparison vs DeepFlareNet**: DeFN reports TSS ~= 0.80 for >=M within 24 h on its own chronological split of 2010-2015 SHARP data. Our numbers sit in the same band but are NOT directly comparable: different sample frame (SWAN-SF instances vs DeFN's per-AR-day), different periods/partitions, different class-balance handling. The honest claim: this pipeline reaches the published TSS band on a public benchmark with strictly time-blocked validation.

## 2. Cross-validated context (Phase 4)

SWAN-SF P3, 3-fold time-blocked CV (mean +/- std):

| model        |   tss_mean |   tss_std |   hss_mean |   hss_std |   bss_mean |   bss_std |
|:-------------|-----------:|----------:|-----------:|----------:|-----------:|----------:|
| lstm         |     0.7704 |    0.0295 |     0.2025 |    0.0985 |    -7.0744 |    5.7805 |
| ensemble     |     0.769  |    0.0245 |     0.2019 |    0.0967 |    -1.6227 |    1.6222 |
| holt_winters |     0.7575 |    0.0152 |     0.1796 |    0.0564 |     0.1176 |    0.0871 |
| climatology  |     0      |    0      |     0      |    0      |     0      |    0      |

P4 replication (independent partition):

| model        |   tss_mean |   tss_std |   hss_mean |   hss_std |
|:-------------|-----------:|----------:|-----------:|----------:|
| holt_winters |     0.855  |    0.0534 |     0.389  |    0.1431 |
| lstm         |     0.7954 |    0.1314 |     0.4551 |    0.2383 |
| ensemble     |     0.7906 |    0.128  |     0.4199 |    0.2046 |
| climatology  |     0      |    0      |     0      |    0      |

Lookback sweep (LSTM, P3):

|   lookback_steps |   tss_mean |   tss_std |   hss_mean |
|-----------------:|-----------:|----------:|-----------:|
|               15 |     0.7609 |    0.0319 |     0.1958 |
|               30 |     0.7656 |    0.0256 |     0.201  |
|               60 |     0.7704 |    0.0295 |     0.2025 |

## 3. Ablation - SWAN-SF (statistical)

Grouped permutation importance of the P3-trained LSTM, evaluated on held-out P4 (delta TSS when the group is shuffled across samples; gradients bundled with their base feature):

| group   |   n_columns |   tss_base |   delta_tss_mean |   delta_tss_std |
|:--------|------------:|-----------:|-----------------:|----------------:|
| MEANGAM |           1 |     0.8727 |           0.0147 |          0.0034 |
| TOTUSJH |           1 |     0.8727 |           0.0135 |          0.0011 |
| SHRGT45 |           1 |     0.8727 |           0.0119 |          0.0005 |
| MEANSHR |           1 |     0.8727 |           0.0113 |          0.0028 |
| ABSNJZH |           1 |     0.8727 |           0.01   |          0.001  |
| TOTBSQ  |           1 |     0.8727 |           0.0098 |          0.0007 |
| SAVNCPP |           1 |     0.8727 |           0.0091 |          0.0012 |
| MEANGBT |           1 |     0.8727 |           0.005  |          0.0015 |
| R_VALUE |           1 |     0.8727 |           0.0034 |          0.0069 |
| TOTUSJZ |           1 |     0.8727 |           0.0026 |          0.0032 |
| MEANGBZ |           1 |     0.8727 |           0.0022 |          0.004  |
| TOTPOT  |           1 |     0.8727 |           0      |          0      |
| USFLUX  |           1 |     0.8727 |           0      |          0      |
| MEANPOT |           1 |     0.8727 |           0      |          0      |

Figure: `figures/permutation_importance.png`.

## 4. Ablation - MVP AIA channels (anecdotal, n=14)

The same harness on the single-AR seq_v1 dataset, focus channels per the confined-vs-eruptive question (AIA 131/304; magnetogram enters via its derived groups flux_total/b_peak):

| group       |   n_columns |   tss_base |   delta_tss_mean |   delta_tss_std |
|:------------|------------:|-----------:|-----------------:|----------------:|
| area_px     |           2 |          0 |                0 |               0 |
| flux_total  |           2 |          0 |                0 |               0 |
| signed_flux |           2 |          0 |                0 |               0 |
| b_peak      |           2 |          0 |                0 |               0 |
| aia_0094    |           2 |          0 |                0 |               0 |
| aia_0131    |           2 |          0 |                0 |               0 |
| aia_0171    |           2 |          0 |                0 |               0 |
| aia_0193    |           2 |          0 |                0 |               0 |
| aia_0211    |           2 |          0 |                0 |               0 |
| aia_0304    |           2 |          0 |                0 |               0 |
| aia_1600    |           2 |          0 |                0 |               0 |
| aia_1700    |           2 |          0 |                0 |               0 |

Drop-one retrains:

| group        |   tss |   delta_tss |
|:-------------|------:|------------:|
| <full model> |     0 |           0 |
| aia_0131     |     0 |           0 |
| aia_0304     |     0 |           0 |
| flux_total   |     0 |           0 |
| b_peak       |     0 |           0 |

With 14 sequences (3 positive) every delta is 0: the harness runs, but channel attribution on our own data requires fetching more AR windows (the builder is multi-AR-ready). Reported as-is; no conclusion drawn.

## 5. End-to-end reproducibility (Gate G5)

`solarflare run-all -w ar11158_feb2011` executed twice: stages A-C served from cache, D-E rebuilt deterministically; the reproducibility keys (dataset stats + evaluation metrics) compared EQUAL.

```json
{
  "dataset": {
    "status": "build",
    "seconds": 0.05,
    "n_sequences": 14,
    "n_positive": 3,
    "positive_rate": 0.21428571428571427,
    "missing_fraction_overall": 0.0018601190476190475
  },
  "evaluate": {
    "status": "build",
    "seconds": 0.27,
    "models": [
      "climatology",
      "holt_winters"
    ],
    "tss_mean": {
      "climatology": 0.0,
      "holt_winters": 0.0
    }
  }
}
```

## 6. Known limitations / next steps

- LSTM probabilities are miscalibrated (pos-weight inflation; negative BSS) - add Platt/isotonic calibration before operational use; TSS/ROC unaffected.
- Our own labeled dataset is one AR / 36 h; every per-AR conclusion is anecdotal until more windows are fetched (Phase 1 fetch is the only bottleneck, ~3.5 h per AR window).
- Streamlit dashboard: deferred (optional in the phase spec); the per-AR probability view can be assembled from the cached sample + fitted models when needed.

## Figures

![roc.png](figures/roc.png)
![reliability.png](figures/reliability.png)
![permutation_importance.png](figures/permutation_importance.png)
![lstm_curves.png](figures/lstm_curves.png)