"""Generate a WalkerNet architecture figure with PlotNeuralNet components."""

from __future__ import annotations

import sys
import re
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
        # ``to_head`` appends ``layers/`` for the upstream repository layout.
        # The vendored styles live directly in ``plotneuralnet_layers`` here,
        # so the generated header is normalized immediately after emission.
        to_head(str(LAYER_ROOT).replace("\\", "/").rstrip("/") + "/../"),
        to_cor(),
        to_begin(),
        # Input: 12 historical months × 4 coupled fields.
        to_Conv(
            "input", s_filer="12x4", n_filer=180, offset="(0,0,0)", to="(0,0,0)",
            width=2.2, height=15, depth=15, caption="Input",
        ),
        # Four variable-specific projections with shared weights over history.
        to_ConvConvRelu(
            "patch", s_filer="45x90", n_filer=(4, 256), offset="(1.6,0,0)", to="(input-east)",
            width=(1.4, 1.0), height=12, depth=12, caption="Patch",
        ),
        to_connection("input", "patch"),
        # Local fusion: 48 tokens + one fusion token.
        to_ConvConvRelu(
            "fusion", s_filer="49 tokens", n_filer=(4, 256), offset="(1.6,0,0)", to="(patch-east)",
            width=(1.5, 1.1), height=10, depth=10, caption="Fusion",
        ),
        to_connection("patch", "fusion"),
        # Global spatial attention: representative repeated stack.
        to_Conv(
            "spatial1", s_filer="4050", n_filer="256", offset="(1.7,0.8,0)", to="(fusion-east)",
            width=1.0, height=8.5, depth=8.5, caption="S1",
        ),
        to_Conv(
            "spatial6", s_filer="4050", n_filer="256", offset="(1.15,0,0)", to="(spatial1-east)",
            width=1.0, height=8.5, depth=8.5, caption="S6",
        ),
        to_connection("fusion", "spatial1"),
        to_connection("spatial1", "spatial6"),
        # Month gate and top-2 routing.
        to_SoftMax("gate", s_filer="month", offset="(1.7,-0.9,0)", to="(spatial6-east)", width=1.5, height=3.0, depth=7.0, caption="Gate"),
        to_connection("spatial6", "gate"),
        to_Conv("expert_a", s_filer="FFN", n_filer=1, offset="(1.3,1.6,0)", to="(gate-east)", width=0.8, height=4.5, depth=4.5, caption="E1"),
        to_Conv("expert_b", s_filer="FFN", n_filer=1, offset="(1.3,-2.3,0)", to="(gate-east)", width=0.8, height=4.5, depth=4.5, caption="E2"),
        to_connection("gate", "expert_a"),
        to_connection("gate", "expert_b"),
        # Coupled decoder: low-res map → two x2 upsampling stages.
        to_Pool("lowres", offset="(1.4,1.6,0)", to="(expert_b-east)", width=1.0, height=7.5, depth=7.5, caption="Map"),
        to_connection("expert_a", "lowres"),
        to_connection("expert_b", "lowres"),
        to_Pool("up2", offset="(1.2,0,0)", to="(lowres-east)", width=1.0, height=10.0, depth=10.0, caption="Up2"),
        to_connection("lowres", "up2"),
        to_Pool("up4", offset="(1.2,0,0)", to="(up2-east)", width=1.0, height=13.0, depth=13.0, caption="Up2"),
        to_connection("up2", "up4"),
        to_Conv("delta", s_filer="180x360", n_filer="4", offset="(1.4,0,0)", to="(up4-east)", width=1.2, height=14.0, depth=14.0, caption="Head"),
        to_connection("up4", "delta"),
        to_Sum("add", offset="(1.2,-1.0,0)", to="(delta-east)", radius=2.1, opacity=0.85),
        to_connection("delta", "add"),
        to_Conv("output", s_filer="180x360", n_filer="4", offset="(1.1,0,0)", to="(add-east)", width=1.5, height=14.0, depth=14.0, caption="Output"),
        to_connection("add", "output"),
        # A compact annotation strip keeps the implementation details legible
        # without letting long captions collide with the 3-D blocks.
        r"\tikzset{stageNote/.style={font=\scriptsize,align=center,text width=3.0cm,minimum height=0.85cm,rounded corners=2pt,draw=black!25,fill=black!3,inner sep=3pt}}",
        r"\node[font=\scriptsize\bfseries,text=black!55] at (11,-4.05) {Implementation details};",
        r"\node[stageNote] at (1.25,-5.0) {\textbf{Input}\\12-month state; TOS / ZOS / TAUU / TAUV};",
        r"\node[stageNote] at (4.9,-5.0) {\textbf{Patch embedding}\\4x4 Conv2d to 45x90 tokens};",
        r"\node[stageNote] at (8.55,-5.0) {\textbf{Local fusion}\\48 field-time tokens\\+ fusion token};",
        r"\node[stageNote] at (12.2,-5.0) {\textbf{Global space}\\6 pre-norm SABs; 8 heads};",
        r"\node[stageNote] at (15.85,-5.0) {\textbf{Month-gated TMoE}\\12 experts; top-k=2};",
        r"\node[stageNote] at (19.5,-5.0) {\textbf{Decoder / rollout}\\PixelShuffle x2 x2; 1x1 head; residual};",
        r"\node[font=\Large\bfseries] at (11,3.8) {WalkerNet};",
        r"\node[font=\large] at (11,3.1) {local fusion $\rightarrow$ global space $\rightarrow$ month-gated TMoE $\rightarrow$ residual rollout};",
        r"\node[font=\small] at (11,2.45) {B x 12 x 4 x 180 x 360  |  4050 spatial tokens  |  12 experts, top-k=2  |  residual output};",
        to_end(),
    ]
    output = OUT / "walkernet_plotneuralnet.tex"
    to_generate(arch, str(output))
    # Normalize the header to the vendored PlotNeuralNet layer directory.
    tex = output.read_text(encoding="utf-8")
    wrong = str((LAYER_ROOT / ".." / "layers").as_posix()) + "/"
    # The generated .tex sits in ``figures/``; keep this reference portable
    # instead of embedding the author's absolute checkout path.
    right = "plotneuralnet_layers/"
    tex = tex.replace(wrong, right)
    # PlotNeuralNet's default dimension labels are useful for small textbook
    # diagrams but collide in this wide, multi-stage architecture.  Keep the
    # dimensions in the compact annotation above and let the 3-D blocks read
    # cleanly at publication scale.
    tex = re.sub(r"^\s*xlabel=.*\n", "", tex, flags=re.MULTILINE)
    tex = re.sub(r"^\s*zlabel=.*\n", "", tex, flags=re.MULTILINE)
    output.write_text(tex, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
