"""Generate a PlotNeuralNet-style diagram of WalkerNet's SpatialAttentionBlock."""

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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arch = [
        to_head(str(LAYER_ROOT).replace("\\", "/").rstrip("/") + "/../"),
        to_cor(),
        to_begin(),
        # WalkerNet uses B x N x d_model spatial tokens, with N=45x90=4050.
        to_Conv(
            "input", s_filer="4050 tokens", n_filer=256, offset="(0,0,0)", to="(0,0,0)",
            width=1.8, height=10, depth=10, caption="Input",
        ),
        to_Pool(
            "norm1", offset="(1.35,0,0)", to="(input-east)",
            width=0.75, height=8.0, depth=8.0, caption="LN",
        ),
        to_connection("input", "norm1"),
        # Multi-head self-attention: Q=K=V=x, eight heads.
        to_ConvConvRelu(
            "mha", s_filer="4050", n_filer=(256, 256), offset="(1.35,0,0)", to="(norm1-east)",
            width=(1.25, 1.0), height=8.5, depth=8.5, caption="MHA",
        ),
        to_connection("norm1", "mha"),
        to_Sum("add1", offset="(1.15,0,0)", to="(mha-east)", radius=1.35, opacity=0.85),
        to_connection("mha", "add1"),
        # First residual path: x + MHA(LN(x)).
        r"\draw[copyconnection,-Stealth] (input-north) .. controls +(0,2.0) and +(0,2.0) .. (add1-north);",
        to_Pool(
            "norm2", offset="(1.25,0,0)", to="(add1-east)",
            width=0.75, height=8.0, depth=8.0, caption="LN",
        ),
        to_connection("add1", "norm2"),
        # FFN: Linear(256,1024) -> GELU -> Linear(1024,256).
        to_ConvConvRelu(
            "ffn", s_filer="256-1024-256", n_filer=(256, 1024), offset="(1.35,0,0)", to="(norm2-east)",
            width=(1.35, 1.0), height=8.5, depth=8.5, caption="FFN",
        ),
        to_connection("norm2", "ffn"),
        to_Sum("add2", offset="(1.15,0,0)", to="(ffn-east)", radius=1.35, opacity=0.85),
        to_connection("ffn", "add2"),
        # Second residual path: x' + FFN(LN(x')).
        r"\draw[copyconnection,-Stealth] (add1-north) .. controls +(0,2.0) and +(0,2.0) .. (add2-north);",
        to_Conv(
            "output", s_filer="4050 tokens", n_filer=256, offset="(1.25,0,0)", to="(add2-east)",
            width=1.8, height=10, depth=10, caption="Output",
        ),
        to_connection("add2", "output"),
        r"\node[font=\Large\bfseries] at (8.0,4.0) {WalkerNet Spatial Attention Block};",
        r"\node[font=\large] at (8.0,3.3) {Pre-norm attention + FFN with two residual paths};",
        r"\node[font=\small] at (8.0,2.65) {$x' = x + \mathrm{MHA}(\mathrm{LN}(x))$\qquad $y = x' + \mathrm{FFN}(\mathrm{LN}(x'))$};",
        r"\tikzset{note/.style={font=\scriptsize,align=center,text width=3.0cm,minimum height=0.75cm,rounded corners=2pt,draw=black!25,fill=black!3,inner sep=3pt}}",
        r"\node[note] at (1.0,-4.0) {\textbf{Token shape}\\B x 4050 x 256};",
        r"\node[note] at (4.2,-4.0) {\textbf{Attention}\\8 heads; Q=K=V};",
        r"\node[note] at (7.4,-4.0) {\textbf{FFN}\\256 to 1024 to 256; GELU};",
        r"\node[note] at (10.6,-4.0) {\textbf{Residual 1}\\attention update};",
        r"\node[note] at (13.8,-4.0) {\textbf{Residual 2}\\FFN update};",
        r"\node[note] at (17.0,-4.0) {\textbf{Output}\\same token shape};",
        to_end(),
    ]

    output = OUT / "walkernet_spatial_attention_block.tex"
    to_generate(arch, str(output))
    tex = output.read_text(encoding="utf-8")
    wrong = str((LAYER_ROOT / ".." / "layers").as_posix()) + "/"
    tex = tex.replace(wrong, "plotneuralnet_layers/")
    # Long dimension labels are retained in the note strip instead of being
    # rendered on the narrow 3-D faces where they would overlap.
    tex = re.sub(r"^\s*xlabel=.*\n", "", tex, flags=re.MULTILINE)
    tex = re.sub(r"^\s*zlabel=.*\n", "", tex, flags=re.MULTILINE)
    output.write_text(tex, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
