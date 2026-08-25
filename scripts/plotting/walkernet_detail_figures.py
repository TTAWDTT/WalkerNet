"""Generate standalone PlotNeuralNet detail figures for WalkerNet.

The three figures correspond to the implementation in ``src/model.py``:
input patch embedding/local fusion, target-month TMoE routing, and decoder plus
residual autoregressive rollout.  They intentionally complement, rather than
replace, the compact whole-network overview.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


LOCAL_PLOT_NEURAL_NET = Path(__file__).resolve().parent
PLOT_NEURAL_NET = (
    LOCAL_PLOT_NEURAL_NET
    if (LOCAL_PLOT_NEURAL_NET / "pycore" / "tikzeng.py").exists()
    else Path(r"C:\Users\zhen.luo\AppData\Local\Temp\PlotNeuralNet_WalkerNet")
)
sys.path.insert(0, str(PLOT_NEURAL_NET))
from pycore.tikzeng import (  # type: ignore  # noqa: E402
    to_Conv,
    to_ConvConvRelu,
    to_Pool,
    to_SoftMax,
    to_Sum,
    to_connection,
    to_cor,
    to_end,
    to_generate,
    to_head,
    to_begin,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
LAYER_ROOT = OUT / "plotneuralnet_layers"


def _head() -> str:
    return to_head(str(LAYER_ROOT).replace("\\", "/").rstrip("/") + "/../") + r"\usetikzlibrary{calc}"


def _finish(arch: list[str], name: str) -> Path:
    output = OUT / f"{name}.tex"
    to_generate(arch, str(output))
    tex = output.read_text(encoding="utf-8")
    wrong = str((LAYER_ROOT / ".." / "layers").as_posix()) + "/"
    tex = tex.replace(wrong, "plotneuralnet_layers/")
    # Keep long dimensions in the explicit note strip, not on narrow faces.
    tex = re.sub(r"^\s*xlabel=.*\n", "", tex, flags=re.MULTILINE)
    tex = re.sub(r"^\s*zlabel=.*\n", "", tex, flags=re.MULTILINE)
    output.write_text(tex, encoding="utf-8")
    return output


def patch_embedding() -> Path:
    arch = [
        _head(), to_cor(), to_begin(),
        to_Conv("input", s_filer="12 x 4 x 180 x 360", n_filer=1, width=1.8, height=12, depth=12, caption="Input"),
        to_ConvConvRelu("proj", s_filer="45 x 90", n_filer=(1, 256), offset="(1.4,0,0)", to="(input-east)", width=(1.2, 1.0), height=10, depth=10, caption="Proj."),
        to_connection("input", "proj"),
        to_Conv("tokens", s_filer="45 x 90", n_filer=256, offset="(1.5,0,0)", to="(proj-east)", width=1.0, height=9, depth=9, caption="Tokens"),
        to_connection("proj", "tokens"),
        to_ConvConvRelu("local", s_filer="49 tokens", n_filer=(256, 256), offset="(1.5,0,0)", to="(tokens-east)", width=(1.4, 1.0), height=10, depth=10, caption="Fusion"),
        to_connection("tokens", "local"),
        to_Conv("spatial", s_filer="4050", n_filer=256, offset="(1.5,0,0)", to="(local-east)", width=1.6, height=11, depth=11, caption="Output"),
        to_connection("local", "spatial"),
        r"\node[font=\Large\bfseries] at (7.5,4.0) {Patch Embedding and Local Fusion};",
        r"\node[font=\large] at (7.5,3.3) {Variable-aware patch projection followed by per-patch time-variable attention};",
        r"\tikzset{note/.style={font=\scriptsize,align=center,text width=3.3cm,minimum height=0.85cm,rounded corners=2pt,draw=black!25,fill=black!3,inner sep=3pt}}",
        r"\node[note] at (1.4,-4.0) {\textbf{Input}\\B x 12 x 4 x 180 x 360};",
        r"\node[note] at (4.5,-4.0) {\textbf{Four projections}\\one 4x4 Conv2d per variable};",
        r"\node[note] at (7.6,-4.0) {\textbf{Patch grid}\\45 x 90 = 4050 spatial locations};",
        r"\node[note] at (10.7,-4.0) {\textbf{Local sequence}\\48 field-time tokens + fusion token};",
        r"\node[note] at (13.8,-4.0) {\textbf{Output}\\B x 4050 x 256 spatial tokens};",
        to_end(),
    ]
    return _finish(arch, "walkernet_patch_embedding_local_fusion")


def tmoe() -> Path:
    arch = [
        _head(), to_cor(), to_begin(),
        to_Conv("input", s_filer="4050 tokens", n_filer=256, width=1.6, height=10, depth=10, caption="Tokens"),
        to_Pool("norm", offset="(1.35,0,0)", to="(input-east)", width=0.75, height=8, depth=8, caption="LN"),
        to_connection("input", "norm"),
        to_SoftMax("gate", s_filer="12", offset="(1.35,1.6,0)", to="(norm-east)", width=1.4, height=3.5, depth=7, caption="Month gate"),
        to_connection("norm", "gate"),
        to_Conv("e1", s_filer="FFN", n_filer=256, offset="(1.5,2.2,0)", to="(gate-east)", width=0.9, height=5, depth=5, caption="E1"),
        to_Conv("e2", s_filer="FFN", n_filer=256, offset="(1.5,-2.2,0)", to="(gate-east)", width=0.9, height=5, depth=5, caption="E2"),
        to_connection("gate", "e1"),
        to_connection("gate", "e2"),
        to_Sum("merge", offset="(1.6,0,0)", to="(e1-east)", radius=1.45, opacity=0.85),
        to_connection("e1", "merge"),
        r"\draw [connection] (e2-east) -- (merge-west);",
        r"\draw [copyconnection,-Stealth] (gate-north) .. controls +(0,2.0) and +(0,2.0) .. (merge-north);",
        to_Conv("output", s_filer="4050 tokens", n_filer=256, offset="(1.45,0,0)", to="(merge-east)", width=1.6, height=10, depth=10, caption="Output"),
        to_connection("merge", "output"),
        r"\node[font=\Large\bfseries] at (8.0,6.3) {Target-month Temporal Mixture-of-Experts};",
        r"\node[font=\large] at (8.0,5.6) {Soft top-k routing is shared across all spatial tokens in a sample};",
        r"\tikzset{note/.style={font=\scriptsize,align=center,text width=2.55cm,minimum height=0.85cm,rounded corners=2pt,draw=black!25,fill=black!3,inner sep=3pt}}",
        r"\node[note] at (1.4,-4.0) {\textbf{Input}\\B x 4050 x 256};",
        r"\node[note] at (4.2,-4.0) {\textbf{Gate input}\\target month embedding};",
        r"\node[note] at (7.0,-4.0) {\textbf{Routing}\\12 logits; retain top-k=2};",
        r"\node[note] at (9.8,-4.0) {\textbf{Experts}\\two weighted FFN branches};",
        r"\node[note] at (12.6,-4.0) {\textbf{Merge}\\softmax weights; residual output};",
        to_end(),
    ]
    return _finish(arch, "walkernet_temporal_moe")


def decoder_rollout() -> Path:
    arch = [
        _head(), to_cor(), to_begin(),
        to_Conv("tokens", s_filer="4050 x 256", n_filer=1, width=1.4, height=9, depth=9, caption="Tokens"),
        to_Pool("map", offset="(1.3,0,0)", to="(tokens-east)", width=0.9, height=7, depth=7, caption="Map"),
        to_connection("tokens", "map"),
        to_Pool("up1", offset="(1.25,0,0)", to="(map-east)", width=1.0, height=9, depth=9, caption="Up x2"),
        to_connection("map", "up1"),
        to_Pool("up2", offset="(1.25,0,0)", to="(up1-east)", width=1.0, height=11, depth=11, caption="Up x2"),
        to_connection("up1", "up2"),
        to_Conv("head", s_filer="180 x 360", n_filer=4, offset="(1.35,0,0)", to="(up2-east)", width=1.2, height=12, depth=12, caption="Head"),
        to_connection("up2", "head"),
        to_Sum("add", offset="(1.3,-1.2,0)", to="(head-east)", radius=1.5, opacity=0.85),
        to_connection("head", "add"),
        to_Conv("output", s_filer="180 x 360", n_filer=4, offset="(1.35,0,0)", to="(add-east)", width=1.7, height=12, depth=12, caption=" "),
        to_connection("add", "output"),
        # The residual baseline is the last observed high-resolution state,
        # not the low-resolution token tensor.  It is rendered as a small
        # PlotNeuralNet block so the residual branch matches the main flow.
        to_Conv("xlast", s_filer="180 x 360", n_filer=4, offset="(0,-1.8,0)", to="(tokens-west)", width=0.8, height=3.0, depth=3.0, caption=" "),
        r"\draw [copyconnection,-Stealth] (xlast-east) .. controls +(3.0,-2.0) and +(-1.0,-2.0) .. (add-south);",
        r"\node[font=\Large\bfseries] at (8.0,6.7) {Decoder and Residual Autoregressive Rollout};",
        r"\node[font=\large] at (8.0,6.0) {Decode $\Delta y$ at full resolution, then feed the new state back into the input window};",
        r"\node[font=\small] at (8.0,5.35) {$\hat{x}_{t+1}=x_{last}+\Delta y_{t+1}$};",
        # Explicit rollout timeline: each prediction updates the next 12-month
        # input window before the following WalkerNet call.
        r"\tikzset{rollbox/.style={draw=black!35,fill=blue!4,rounded corners=2pt,align=center,font=\scriptsize,minimum width=2.05cm,minimum height=0.65cm,inner sep=3pt}}",
        r"\node[font=\small\bfseries,text=black!60] at (8.0,-4.10) {Autoregressive rollout};",
        r"\node[rollbox] (w0) at (1.5,-5.20) {$x_{t-11:t}$};",
        r"\node[rollbox] (m1) at (4.3,-5.20) {$f_\theta$};",
        r"\node[rollbox] (p1) at (7.1,-5.20) {$\hat{x}_{t+1}$};",
        r"\node[rollbox] (w1) at (9.9,-5.20) {shift};",
        r"\node[rollbox] (m2) at (12.7,-5.20) {$f_\theta$};",
        r"\node[rollbox] (p2) at (15.5,-5.20) {$\hat{x}_{t+2}$};",
        r"\node[font=\Large\bfseries] (more) at (18.0,-5.20) {$\cdots$};",
        r"\draw[-Stealth,thick,draw=black!60] (w0.east) -- (m1.west);",
        r"\draw[-Stealth,thick,draw=black!60] (m1.east) -- (p1.west);",
        r"\draw[-Stealth,thick,draw=black!60] (p1.east) -- node[above,font=\scriptsize] {update} (w1.west);",
        r"\draw[-Stealth,thick,draw=black!60] (w1.east) -- (m2.west);",
        r"\draw[-Stealth,thick,draw=black!60] (m2.east) -- (p2.west);",
        r"\draw[-Stealth,thick,draw=black!60] (p2.east) -- (more.west);",
        to_end(),
    ]
    return _finish(arch, "walkernet_decoder_residual_rollout")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for generator in (patch_embedding, tmoe, decoder_rollout):
        print(generator())


if __name__ == "__main__":
    main()
