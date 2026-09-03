"""Generate the standalone residual-baseline arrow used by WalkerNet figures."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"


def main() -> None:
    tex = r"""\documentclass[border=3pt,tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\begin{document}
\begin{tikzpicture}
\draw[-Stealth,line width=1.15mm,draw={rgb:blue,4;red,1;green,1;black,3}]
  (0,0) .. controls (1.1,-1.15) and (3.0,-1.15) .. (4.3,0.25);
\end{tikzpicture}
\end{document}
"""
    output = OUT / "walkernet_residual_arrow.tex"
    output.write_text(tex, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
