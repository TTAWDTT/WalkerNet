# CNOP plotting entry points

This directory is the index for the plotting scripts in `scripts/cnop/`. The
scripts remain in their historical locations because several experiment and
audit modules import them directly; the table below is the intended routing.

## Paper figures

| Product | Entry point | Status |
|---|---|---|
| Response evolution (TOS + wind / narrow ZOS) | `../plot_cnop_paper_response.py` | **Canonical** |
| Ten-case lead-12 overview | `../plot_cnop_ten_case_lead12.py` | Canonical overview |
| Initial perturbation overview | `../plot_cnop_initial_delta_overview.py` | Canonical perturbation map |
| Portable overview | `../plot_portable_cnop_overview.py` | Compatibility renderer |
| Portable response evolution | `../plot_portable_cnop_response_evolution.py` | Compatibility renderer; do not use for paper delivery |
| Pacific delayed candidate panels | `../plot_pacific_delayed_onset_candidates.py` | Specialized candidate product |

## Diagnostics and supporting products

- `plot_cnop_diagnostics.py`: numerical diagnostic panels and summary plots.
- `plot_cnop_monthly_response.py`: shared model replay, map, and smoothing
  helpers used by the paper renderers.
- `plot_climatology_cnop_overview.py`: climatology-driven overview.
- `plot_cnop_cluster_composites.py`: cluster composites.
- `plot_global_constraint_calibration.py`: calibration summary plots.

## Fixed display convention

For cross-basin response-evolution figures, use the canonical renderer with:

- `--tos-vmax 0.8`
- `--zos-vmax 0.03`

These are display-only settings. They do not alter CNOP optimization, saved
perturbations, or evaluation metrics. The obsolete remote renderer with a
full-height left panel and automatic per-case scales is archival only.

The Pacific delayed candidate generator now defaults to the fixed convention
and accepts `--tos-vmax 0.8 --zos-vmax 0.03`. Passing `0` to either option
explicitly restores the legacy percentile-floor behavior for provenance
reproduction.
