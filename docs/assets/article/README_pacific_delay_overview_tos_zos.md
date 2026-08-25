# Pacific delayed overview with per-case TOS/ZOS perturbations

`pacific_delay_overview_tos_zos.png` keeps the original truth/baseline/perturbed columns and replaces the left initial-delta column with a two-layer pseudo-3D mini-map for every case:

- upper skewed surface of each row: rank-1 TOS perturbation;
- lower skewed surface of each row: rank-1 ZOS perturbation;
- shallow side walls and displaced trapezoid surfaces provide the pseudo-3D geometry;
- both fields use high-density cubic display interpolation, Gaussian smoothing, and dense contour levels;
- the original right-side columns, labels, and colorbars are preserved from `pacific_delay_overview.png`.

The ten delayed rank-1 NPZ files were used as the source; no CNOP or forecast values were recomputed. A copy is also in `C:\Users\zhen.luo\Desktop\article\pacific_delay_overview_tos_zos.png`.
