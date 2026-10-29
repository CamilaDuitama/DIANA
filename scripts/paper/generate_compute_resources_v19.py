"""Runtime and memory of diana-predict end to end with the released DIANA (v19 benchmark).

Reads results/benchmark_v19net/benchmark.tsv (scripts/inference/run_benchmark_v19.sbatch, collected by
scripts/inference/collect_benchmark_v19.py): 30 held-out runs from their Logan unitigs and 10 from raw reads, 8 CPUs each,
/usr/bin/time -v (wall clock; maximum resident set size of the largest process). Writes one plotly figure with two panels
(colours from config.py) to results/paper_v19/sup_compute.png: (a) wall time and (b) peak memory against input size, and
prints the numbers quoted in the manuscript.

    ./env/bin/python scripts/paper/generate_compute_resources_v19.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/paper_v19"
V = PLOT_CONFIG["colors"]["palette"]
XT = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
ROUTES = {"logan": ("Logan unitigs", V[0]), "reads": ("raw reads", V[2])}   # not model colours


def panels(b: pd.DataFrame, out: Path) -> None:
    """(a) wall time and (b) peak memory against compressed input size, one point per run, log axes."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["a  Wall time", "b  Peak memory"], horizontal_spacing=0.12)
    for col, (y, ytitle, yticks) in enumerate([("minutes", "wall time (min)", [0.25, 0.5, 1, 2, 5, 10, 20, 30]),
                                               ("memory_gb", "peak memory (GB)", [2, 5, 10, 20, 40])], start=1):
        for route, (name, colour) in ROUTES.items():
            d = b[b.route == route]
            fig.add_scatter(x=d.input_gb, y=d[y], mode="markers", name=f"{name} ({len(d)} runs)", showlegend=col == 1,
                            marker=dict(color=colour, size=9, line=dict(color="white", width=1)), row=1, col=col)
        fig.update_yaxes(title_text=ytitle, type="log", tickvals=yticks, ticktext=[f"{v:g}" for v in yticks], row=1, col=col)
    fig.update_xaxes(title_text="input size (GB, compressed)", type="log", tickvals=XT, ticktext=[f"{v:g}" for v in XT])
    fig.update_annotations(xanchor="left", x=None, font=dict(size=16))
    for a, x in zip(fig.layout.annotations, (0, 0.56)):
        a.x = x
    fig.update_layout(template=PLOT_CONFIG["template"], width=1150, height=470, font=dict(size=14, color="#333333"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0), margin=dict(l=70, r=20, t=80, b=60))
    fig.write_image(out, scale=2)


def main() -> None:
    b = pd.read_csv(ROOT / "results/benchmark_v19net/benchmark.tsv", sep="\t")
    if (b.exit_status != 0).any():
        raise ValueError("a benchmark task failed")
    b = b.assign(input_gb=b.input_bytes / 1e9, minutes=b.wall_clock_s / 60, memory_gb=b.max_rss_kb / 1024 ** 2)
    FIG.mkdir(parents=True, exist_ok=True)
    panels(b, FIG / "sup_compute.png")
    for route, g in b.groupby("route"):
        print(f"{route}: {len(g)} runs, input {g.input_gb.min():.4f} to {g.input_gb.max():.1f} GB; wall time median {g.wall_clock_s.median():.0f} s, "
              f"max {g.minutes.max():.1f} min; peak memory median {g.memory_gb.median():.1f} GB, max {g.memory_gb.max():.1f} GB; "
              f"max difference from the paper's probabilities {g.max_abs_diff_paper.max():.1e}")


if __name__ == "__main__":
    main()
