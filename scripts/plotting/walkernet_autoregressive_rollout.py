"""Generate a standalone diagram of WalkerNet's autoregressive rollout."""

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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arch = [
        to_head(str(LAYER_ROOT).replace("\\", "/").rstrip("/") + "/../"),
        to_cor(),
        to_begin(),
        # One-step WalkerNet call: the model predicts a residual increment.
        to_Conv(
            "history0", s_filer="12 x 4 x 180 x 360", n_filer=1,
            width=1.6, height=10, depth=10, caption="History k",
        ),
        to_ConvConvRelu(
            "model", s_filer="4050 x 256", n_filer=(256, 256),
            offset="(1.5,0,0)", to="(history0-east)",
            width=(1.7, 1.1), height=11, depth=11, caption="WalkerNet",
        ),
        to_connection("history0", "model"),
        to_SoftMax(
            "condition", s_filer="month + step", offset="(0,2.25,0)",
            to="(model-north)", width=1.35, height=2.6, depth=4.5, caption="Condition",
        ),
        r"\draw [connection] (condition-south) -- (model-north);",
        to_Conv(
            "delta", s_filer="180 x 360", n_filer=4,
            offset="(1.6,0,0)", to="(model-east)",
            width=1.15, height=9, depth=9, caption="Delta y",
        ),
        to_connection("model", "delta"),
        # The last observed high-resolution state is the residual baseline.
        to_Conv(
            "xlast", s_filer="180 x 360", n_filer=4,
            offset="(0,-2.55,0)", to="(delta-east)",
            width=0.95, height=4.5, depth=4.5, caption="x-last",
        ),
        to_Sum("add", offset="(1.45,0,0)", to="(delta-east)", radius=1.5, opacity=0.85),
        to_connection("delta", "add"),
        r"\draw [copyconnection,-Stealth] (xlast-east) .. controls +(0.9,0) and +(-0.5,-1.0) .. (add-south);",
        to_Conv(
            "pred", s_filer="180 x 360", n_filer=4,
            offset="(1.35,0,0)", to="(add-east)",
            width=1.65, height=11, depth=11, caption="x-hat k+1",
        ),
        to_connection("add", "pred"),
        # Window update is shown below the one-step path to make the loop
        # explicit instead of implying that the network is unrolled inside fθ.
        to_Pool(
            "shift", offset="(0,-3.0,0)", to="(pred-south)",
            width=1.25, height=3.6, depth=5.4, caption="Shift",
        ),
        r"\draw [connection] (pred-south) -- (shift-north);",
        to_Conv(
            "history1", s_filer="12 x 4 x 180 x 360", n_filer=1,
            offset="(-7.0,0,0)", to="(shift-west)",
            width=1.6, height=8.5, depth=8.5, caption="History k+1",
        ),
        r"\draw [connection] (shift-west) -- node[above,font=\scriptsize] {append / drop} (history1-east);",
        # Re-enter the same model with the shifted window; this is the rollout
        # loop.  The curved path stays below the main blocks and labels the step.
        r"\draw [copyconnection,-Stealth] (history1-north) .. controls +(0,2.0) and +(-1.5,-2.0) .. (model-south);",
        r"\node[font=\scriptsize,text=black!60] at (4.1,-2.2) {repeat for k+2, k+3, ...};",
        r"\node[font=\Large\bfseries] at (8.0,5.0) {Autoregressive Rollout};",
        r"\node[font=\large] at (8.0,4.35) {one-step residual prediction followed by a deterministic 12-month window shift};",
        r"\node[font=\small] at (8.0,3.72) {$\hat{x}_{k+1}=x_{last}+\Delta y_{k+1}$};",
        r"\tikzset{note/.style={font=\scriptsize,align=center,text width=3.0cm,minimum height=0.75cm,rounded corners=2pt,draw=black!25,fill=black!3,inner sep=3pt}}",
        r"\node[note] at (7.1,-4.65) {\textbf{Window update}\\append $\hat{x}_{k+1}$ and drop the oldest month};",
        r"\node[note] at (12.0,-4.65) {\textbf{Next call}\\same $f_\theta$, updated month and rollout step};",
        to_end(),
    ]

    output = OUT / "walkernet_autoregressive_rollout.tex"
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
