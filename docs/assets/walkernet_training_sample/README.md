# WalkerNet training sample: global pseudo-3D fields

This figure uses one CESM2 historical training sample (`time index 0`) and displays the four model input variables:

- TOS
- ZOS
- TAUX
- TAUY

Each field is first represented on its native rectangular `180×360` latitude–longitude grid. The display then applies NaN-aware Gaussian smoothing, 4× cubic interpolation, dense contour filling, and a bilinear pseudo-3D trapezoid transform. The physical field values are not modified for analysis.

Files:

- `walker_net_training_sample_global_pseudo3d.png`
- `walker_net_training_sample_global_pseudo3d.pdf`
- `walker_net_training_sample_global_pseudo3d.provenance.json`

The vertically stacked version is:

- `walker_net_training_sample_global_pseudo3d_stack.png`
- `walker_net_training_sample_global_pseudo3d_stack.pdf`
