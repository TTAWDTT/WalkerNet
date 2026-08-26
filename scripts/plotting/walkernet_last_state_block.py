"""Generate a standalone PlotNeuralNet block for WalkerNet's x_last state."""

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
        to_Conv(
            "xlast",
            s_filer="180 x 360",
            n_filer=4,
            width=1.8,
            height=13,
            depth=13,
            caption=r"\Huge $x_{last}$",
        ),
        r"\node[font=\Huge\bfseries] at (3.0,4.0) {$x_{last}$: last observed state};",
        r"\node[font=\large] at (3.0,3.25) {high-resolution baseline for residual prediction};",
        r"\node[font=\small] at (3.0,2.55) {$x_{last}\in\mathrm{R}^{4\times180\times360}$};",
        r"\tikzset{note/.style={font=\scriptsize,align=center,text width=3.4cm,minimum height=0.8cm,rounded corners=2pt,draw=black!25,fill=black!3,inner sep=3pt}}",
        r"\node[note] at (1.3,-4.0) {TOS\\sea-surface temperature};",
        r"\node[note] at (4.6,-4.0) {ZOS\\sea-surface height};",
        r"\node[note] at (7.9,-4.0) {TAUU / TAUV\\surface wind stress};",
        r"\node[font=\small] at (5.0,-5.2) {$\hat{x}_{t+1}=x_{last}+\Delta y_{t+1}$};",
        to_end(),
    ]
    output = OUT / "walkernet_last_state_block.tex"
    to_generate(arch, str(output))
    tex = output.read_text(encoding="utf-8")
    wrong = str((LAYER_ROOT / ".." / "layers").as_posix()) + "/"
    tex = tex.replace(wrong, "plotneuralnet_layers/")
    tex = re.sub(r"^\s*xlabel=.*\n", "", tex, flags=re.MULTILINE)
    tex = re.sub(r"^\s*zlabel=.*\n", "", tex, flags=re.MULTILINE)
    output.write_text(tex, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
