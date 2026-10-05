"""Generate publication figures from completed Module 05 logs; never run a solve.

python -m experiments.archive.paper_development.plot_05_policy
"""
from __future__ import annotations

from experiments.paper_final.common.artifacts import ROOT as REPOSITORY_ROOT

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np


from experiments.archive.paper_development.module05_figures.data import load_dataset, sha256
from experiments.archive.paper_development.module05_figures.style import Exporter, apply_style
from experiments.archive.paper_development.module05_figures import panels


ROOT = REPOSITORY_ROOT
DEFAULT_RESULTS = ROOT / "results/paper_final/05_policy/20260927_diffusion60_6seeds_100cases"


def write_gallery(output, entries, provenance):
    cards = []
    for item in entries:
        files = item["files"]
        links = " · ".join(f'<a href="{html.escape(files[e])}">{e.upper()}</a>' for e in ("pdf", "svg", "png"))
        cards.append(f'''<article data-category="{item['category']}">
<a href="{html.escape(files['png'])}" target="_blank"><img loading="lazy" src="{html.escape(files['preview'])}" alt="{html.escape(item['title'])}"></a>
<div class="text"><h2>{html.escape(item['title'])}</h2><p class="takeaway">{html.escape(item['takeaway'])}</p>
<p class="links">{links}</p><details><summary>Scientific caption and selection rule</summary><p>{html.escape(item['caption'])}</p></details></div></article>''')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Module 05 — academic figure gallery</title><style>
:root{color-scheme:light}body{font-family:system-ui,sans-serif;color:#24313d;background:#f7f8fa;margin:0;line-height:1.55}
header,main{max-width:1260px;margin:auto;padding:28px}header{padding-bottom:10px}h1{font-size:29px;margin:0 0 10px}h2{font-size:17px;margin:0 0 8px}
p{margin:8px 0}header p{max-width:950px}a{color:#155d9a}nav{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}
button{font:inherit;border:1px solid #c8d2dc;border-radius:5px;background:white;padding:7px 14px;cursor:pointer}button.active{background:#234f77;color:white}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{background:white;border:1px solid #dbe1e6;border-radius:5px;overflow:hidden}
article img{width:100%;display:block;background:white}.text{padding:18px}.takeaway{font-size:14px}.links{font-size:13px;font-weight:600}details{font-size:13px;color:#465565}
.note{font-size:14px;color:#465565}.hidden{display:none}footer{font-size:12px;margin:26px 0;color:#697884}@media(max-width:800px){.grid{grid-template-columns:1fr}header,main{padding:18px}}
</style><header><h1>Module 05 · Academic figure gallery</h1>
<p>60³ diffusion · six frozen checkpoints · 100 shared fresh test problems · both hierarchy sources.</p>
<p>Start with the aggregate savings, matched action/residual heatmaps, and convergence examples. The supplementary views include every checkpoint; no best-seed selection is needed.</p>
<p class="note">No-overhead solve costs are the primary view. Trajectory heatmaps follow Module 04: problem rank horizontally, AMG cycle increasing upward. Gray cells mean the solve has finished. The dashed prefix–tail boundary marks the switch to a constant tail weight. Hindsight constants are marked with an asterisk. Individual-case gains and cumulative time reductions are different quantities.</p>
<p><a href="main_figure_atlas.pdf">Main figure atlas (PDF)</a> · <a href="figure_pack.zip">Download the complete figure pack</a> · <a href="CAPTIONS.md">All captions</a> · <a href="provenance.json">Data and code provenance</a></p>
<nav><button class="active" data-filter="main">Main figures</button><button data-filter="all-seeds">All-seed variants</button><button data-filter="all">Everything</button></nav></header>
<main><div class="grid">''' + "\n".join(cards) + '''</div><footer>PDF and SVG retain editable vector text and plot geometry; heatmap cells and dense scatter marks are rasterized where appropriate. PNG exports are 600 dpi by default. Captions describe the exact accounting and statistical units.</footer></main>
<script>function show(filter){document.querySelectorAll('article').forEach(x=>x.classList.toggle('hidden',filter!=='all'&&x.dataset.category!==filter));document.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x.dataset.filter===filter));}document.querySelectorAll('button').forEach(x=>x.addEventListener('click',()=>show(x.dataset.filter)));show('main');</script></html>'''
    (output / "index.html").write_text(page)
    text = ["# Module 05 figure captions", "", "The figure gallery is `index.html`; the main pages are combined in `main_figure_atlas.pdf`.", ""]
    for item in entries:
        text.extend([f"## {item['name']} — {item['title']}", "", item["caption"], "", "Interpretation: " + item["takeaway"], ""])
    (output / "CAPTIONS.md").write_text("\n".join(text))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def generate(result_dir, output=None, *, dpi=600, all_seeds=True):
    apply_style()
    output = Path(output or result_dir / "analysis/paper_figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Validating raw data and reconstructing paired measurements...", flush=True)
    data = load_dataset(result_dir)
    print(f"Validated {len(data.seeds)} checkpoints × {len(data.cases)} shared test cases; maximum displayed trajectory {data.maximum_cycles()} cycles.", flush=True)
    (output / "figure_data.json").write_text(json.dumps(data.data_export(), indent=2) + "\n")
    arrays = {"case_order_0_based": data.order}
    width = data.maximum_cycles()
    for source in panels.SOURCES:
        for seed in data.seeds:
            for method in panels.METHODS:
                for field in ("actions", "residuals", "times"):
                    arrays[f"{source}__seed{seed}__{method}__{field}"] = data.heatmap(seed, source, method, field, width)
        arrays[f"{source}__fixed_native_costs"] = np.array([[[data.cells[seed,source,case,f"fixed_{1+i*.05:.2f}"]["native"]
            for i in range(41)] for case in data.cases] for seed in data.seeds])
    np.savez_compressed(output / "figure_arrays.npz", **arrays)
    exporter = Exporter(output, dpi)
    jobs = [
        ("Aggregate effects", lambda: panels.overview(data, exporter)),
        ("Paired RL advantages", lambda: panels.paired_advantage(data, exporter)),
        ("Matched action heatmaps", lambda: panels.matched_heatmaps(data, exporter, all_seeds=all_seeds)),
        ("Matched residual heatmaps", lambda: panels.matched_heatmaps(data, exporter, residual=True, all_seeds=all_seeds)),
        ("RL heatmaps across every seed", lambda: panels.rl_seed_atlas(data, exporter)),
        ("Per-case gain heatmaps", lambda: panels.gain_heatmaps(data, exporter)),
        ("Paired-cost scatter plots", lambda: panels.paired_scatter(data, exporter)),
        ("Exhaustive fixed-weight response", lambda: panels.weight_scan(data, exporter)),
        ("Convergence and action examples", lambda: panels.convergence_examples(data, exporter, all_seeds=all_seeds)),
        ("Controller overhead", lambda: panels.overhead(data, exporter)),
        ("Native references with setup cost", lambda: panels.overview(data, exporter, full_cost=True)),
        ("Empirical distributions of gains", lambda: panels.ecdf(data, exporter)),
    ]
    try:
        for title, function in jobs:
            print(title, flush=True)
            function()
    finally:
        exporter.close()
    source_paths = [Path(__file__), *sorted(Path(__file__).with_name("module05_figures").glob("*.py")),
                    Path(__file__).with_name("test_05_figures.py"), Path(__file__).parent / "05_policy/FIGURES.md"]
    code = {}
    for path in source_paths:
        if path.exists():
            relative = path.relative_to(ROOT)
            target = output / "reproduction" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            code[str(relative)] = sha256(path)
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "result_directory": str(data.root),
                  "plot_code_sha256": code, "validation": data.provenance,
                  "dpi": dpi, "all_seed_variants": all_seeds, "figure_count": len(exporter.entries),
                  "main_figure_count": sum(r["category"] == "main" for r in exporter.entries),
                  "reproduce": "python -m experiments.archive.paper_development.plot_05_policy --result-dir PATH_TO_COMPLETED_RESULTS"}
    write_gallery(output, exporter.entries, provenance)
    # Package only this figure deliverable; original logs and checkpoints stay in place.
    archive = output / "figure_pack.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != archive:
                handle.write(path, path.relative_to(output))
    print(json.dumps({"output": str(output), "figures": len(exporter.entries), "archive_bytes": archive.stat().st_size}), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--main-only", action="store_true", help="Render the seed-1 matched examples plus all main summaries")
    args = parser.parse_args()
    generate(args.result_dir.resolve(), args.output, dpi=args.dpi, all_seeds=not args.main_only)


if __name__ == "__main__":
    main()
