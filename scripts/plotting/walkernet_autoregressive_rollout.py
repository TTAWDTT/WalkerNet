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
        # One-step transition: X_k -> f_theta -> xhat_(k+1) -> window update.
        to_Conv("history0", s_filer="12 x 4 x 180 x 360", n_filer=1,
                to="(0,0,0)", width=1.45, height=8.0, depth=8.0, caption=r"$X_k$"),
        to_ConvConvRelu("model", s_filer="4050 x 256", n_filer=(256, 256),
                        offset="(3.4,0,0)", to="(0,0,0)", width=(1.8, 1.1),
                        height=8.5, depth=8.5, caption="WalkerNet"),
        to_connection("history0", "model"),
        to_Conv("pred", s_filer="180 x 360", n_filer=4,
                offset="(7.2,0,0)", to="(0,0,0)", width=1.45, height=8.0,
                depth=8.0, caption=r"$\hat{x}_{k+1}$"),
        to_connection("model", "pred"),
        r"\tikzset{updatebox/.style={draw=black!45,fill=blue!4,rounded corners=3pt,align=center,font=\small,text width=5.3cm,minimum height=2.7cm,inner sep=6pt}}",
        r"\node[updatebox] (update) at (12.2,0) {\textbf{Update the 12-month window}\\[-1pt] $X_k=[x_{k-11},\ldots,x_k]$\\[-1pt] $\downarrow\;$ drop $x_{k-11}$; append $\hat{x}_{k+1}$\\[-1pt] $X_{k+1}=[x_{k-10},\ldots,x_k,\hat{x}_{k+1}]$\\[-1pt] {\scriptsize advance target month and rollout step}};",
        r"\draw [connection] (pred-east) -- node {\midarrow} (update.west);",
        # The same f_theta is called again with the updated window.
        r"\draw [copyconnection,-Stealth] (update.south) .. controls +(0,-2.0) and +(2.0,-2.0) .. (model-south);",
        r"\node[font=\Large\bfseries] at (7.5,5.25) {Autoregressive Rollout};",
        r"\node[font=\large] at (7.5,4.65) {predict one month, update the window, and call the same model again};",
        r"\node[font=\small] at (7.5,4.05) {$\hat{x}_{k+1}=f_\theta(X_k),\qquad X_{k+1}=\mathrm{shift}(X_k,\hat{x}_{k+1})$};",
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
