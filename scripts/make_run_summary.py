"""Render the DAG of the last run (status + duration per task) from reports/run_log.jsonl -> reports/orchestration_run.png."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["ingest", "validate", "prepare", "transform", "features", "train", "evaluate", "report"]
COLORS = {"COMPLETED": "#c8e6c9", "FAILED": "#ffcdd2", "NOT RUN": "#eeeeee"}


def make():
    rows = {}
    for line in (ROOT / "reports" / "run_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["stage"]] = r
    total = sum(r["seconds"] for r in rows.values())
    ok = all(rows.get(s, {}).get("status") == "COMPLETED" for s in ORDER)
    dot = ['digraph G { rankdir=LR; bgcolor="white"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.18,0.1"]; edge [color="#555555"];',
           f'labelloc=t; fontname="Helvetica-Bold"; fontsize=14; label="Flow run: recommendation-pipeline  |  {"Completed" if ok else "FAILED"}  |  total {total:.1f}s";']
    for s in ORDER:
        r = rows.get(s)
        st = r["status"] if r else "NOT RUN"
        dot.append(f'{s} [label="{s}\\n{st}\\n{r["seconds"]:.1f}s" , fillcolor="{COLORS[st]}"];' if r else
                   f'{s} [label="{s}\\nNOT RUN", fillcolor="{COLORS[st]}"];')
    dot.append(" -> ".join(ORDER) + ";")
    dot.append("}")
    src = ROOT / "reports" / "orchestration_run.dot"
    src.write_text("\n".join(dot))
    try:
        subprocess.run(["dot", "-Tpng", "-Gdpi=170", str(src), "-o", str(ROOT / "reports" / "orchestration_run.png")], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):      # no graphviz binary (e.g. Kaggle) -> matplotlib
        _matplotlib_fallback(rows, ok, total)
    src.unlink()


def _matplotlib_fallback(rows, ok, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(14, 2.2))
    ax.set_xlim(0, len(ORDER) * 1.5); ax.set_ylim(0, 1.6); ax.axis("off")
    ax.set_title(f"Flow run: recommendation-pipeline | {'Completed' if ok else 'FAILED'} | total {total:.1f}s", fontweight="bold")
    for i, s in enumerate(ORDER):
        r = rows.get(s)
        st = r["status"] if r else "NOT RUN"
        x = i * 1.5 + 0.1
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.15, 0.8, boxstyle="round,pad=0.02,rounding_size=0.12", fc=COLORS[st], ec="#555"))
        ax.text(x + 0.575, 0.75, f"{s}\n{st}" + (f"\n{r['seconds']:.1f}s" if r else ""), ha="center", va="center", fontsize=9)
        if i < len(ORDER) - 1:
            ax.annotate("", xy=(x + 1.45, 0.75), xytext=(x + 1.17, 0.75), arrowprops=dict(arrowstyle="->", color="#555"))
    fig.savefig(ROOT / "reports" / "orchestration_run.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make()
