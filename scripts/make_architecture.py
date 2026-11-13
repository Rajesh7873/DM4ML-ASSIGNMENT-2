"""Generate reports/architecture.png (Graphviz) - run:  python scripts/make_architecture.py"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOT = r'''
digraph A {
  rankdir=TB; bgcolor="white"; nodesep=0.32; ranksep=0.55; newrank=true;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fillcolor="#e8f0fe", color="#5b7db1", margin="0.12,0.07"];
  edge [color="#555555", fontname="Helvetica", fontsize=8];
  labelloc=t; fontname="Helvetica-Bold"; fontsize=14; label="End-to-End Recommendation Data Pipeline  (all stages orchestrated by Prefect, cron 02:00 daily)";

  src  [label="Sources\nCSV / synthetic\nusers | items | interactions", shape=cylinder, fillcolor="#fff3cd", color="#b8962e"];
  ing  [label="1. Ingestion\nload as strings\nrow counts + MD5 log"];
  val  [label="2. Validation\nnulls, dtypes, duplicates,\nrating range, timestamps,\nreferential integrity, quality gate"];
  prep [label="3. Preparation\nimpute, standardise,\nde-duplicate,\ntime-based split"];
  tr   [label="4. Transformation\nuser activity, item popularity,\nuser-item sparse matrix,\nitem-genre matrix"];
  feat [label="5. Feature management\nregistry + offline store\nuser | item | pair features"];
  train[label="6. Training\npopularity | ALS MF | hybrid\n(params, metrics, artifacts\n-> MLflow)"];
  ev   [label="7. Evaluation\nP@K R@K NDCG@K\ncoverage, diversity,\nRMSE / MAE"];
  err  [label="data/errors/\nrejected rows + reasons\nreports/validation_report", shape=note, fillcolor="#fde2e2", color="#c05050"];
  fs   [label="data/features/\nParquet + feature catalog", shape=cylinder, fillcolor="#e6f4ea", color="#4a8f5c"];
  mod  [label="models/*.pkl", shape=cylinder, fillcolor="#e6f4ea", color="#4a8f5c"];
  ml   [label="MLflow (SQLite)\nruns, params, metrics", shape=component, fillcolor="#f3e8ff", color="#8a5bb5"];
  out  [label="reports/\nresults table + plots", shape=note, fillcolor="#e6f4ea", color="#4a8f5c"];
  srv  [label="recommend.py\nprediction CLI\n(same FeatureStore)", fillcolor="#fff3cd", color="#b8962e"];

  { rank=same; src -> ing -> val -> prep; }
  { rank=same; ev -> train -> feat -> tr [dir=back]; }
  { rank=same; err; ml; fs; }
  { rank=same; out; mod; srv; }
  prep -> tr;
  val -> err [style=dashed];
  feat -> fs [style=dashed]; fs -> train [style=dashed, label="reused"];
  train -> ml [style=dashed]; ev -> ml [style=dashed];
  train -> mod [style=dashed]; mod -> ev [style=dashed]; mod -> srv [style=dashed]; fs -> srv [style=dashed, label="reused"];
  ev -> out [style=dashed];
}
'''
(ROOT / "reports" / "architecture.dot").write_text(DOT)
subprocess.run(["dot", "-Tpng", "-Gdpi=170", str(ROOT / "reports" / "architecture.dot"), "-o", str(ROOT / "reports" / "architecture.png")], check=True)
(ROOT / "reports" / "architecture.dot").unlink()
print("wrote reports/architecture.png")
