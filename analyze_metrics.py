import argparse
from collections import defaultdict
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def _float(x):
    try:
        return float(x)
    except Exception:
        return None


def _int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "content": r.get("content"),
                    "style": r.get("style"),
                    "size": _int(r.get("size")),
                    "steps": _int(r.get("steps")),
                    "alpha": _float(r.get("alpha")),
                    "beta": _float(r.get("beta")),
                    "gamma": _float(r.get("gamma")),
                    "lr": _float(r.get("lr")),
                    "seed": _int(r.get("seed")),
                    "final_total_loss": _float(r.get("final_total_loss")),
                    "wall_time_s": _float(r.get("wall_time_s")),
                    "ssim": _float(r.get("ssim_vs_content")),
                    "final_image": r.get("final_image"),
                }
            )
    return rows


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Analysis] Saved: {out_path}")


def plot_ssim_vs_beta_by_size(rows, out_dir: Path):
    ssim_by = defaultdict(list)
    for r in rows:
        if r["ssim"] is not None and r["beta"] is not None and r["size"] is not None:
            ssim_by[(r["size"], r["beta"])].append(r["ssim"])
    if not ssim_by:
        return
    sizes = sorted(set(k[0] for k in ssim_by.keys()))
    fig = plt.figure(figsize=(7, 5))
    for sz in sizes:
        betas = sorted(b for (s, b) in ssim_by.keys() if s == sz)
        means = [np.mean(ssim_by[(sz, b)]) for b in betas]
        plt.plot(betas, means, marker="o", label=f"{sz}px")
    plt.xlabel("beta (style weight)")
    plt.ylabel("SSIM vs content (avg)")
    plt.title("SSIM vs beta (by size)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    save_plot(fig, out_dir / "ssim_vs_beta_by_size.png")


def plot_runtime_vs_size(rows, out_dir: Path):
    rt_by = defaultdict(list)
    for r in rows:
        if r["wall_time_s"] is not None and r["size"] is not None:
            rt_by[r["size"]].append(r["wall_time_s"])
    if not rt_by:
        return
    sizes = sorted(rt_by.keys())
    means = [np.mean(rt_by[s]) for s in sizes]
    fig = plt.figure(figsize=(6, 4))
    plt.bar([str(s) for s in sizes], means)
    plt.xlabel("image size (px)")
    plt.ylabel("runtime (s, avg)")
    plt.title("Runtime vs size")
    save_plot(fig, out_dir / "runtime_vs_size.png")


def plot_runtime_vs_beta(rows, out_dir: Path):
    rt_by = defaultdict(list)
    for r in rows:
        if r["wall_time_s"] is not None and r["beta"] is not None:
            rt_by[r["beta"]].append(r["wall_time_s"])
    if not rt_by:
        return
    betas = sorted(rt_by.keys())
    means = [np.mean(rt_by[b]) for b in betas]
    fig = plt.figure(figsize=(6, 4))
    plt.bar([str(b) for b in betas], means)
    plt.xlabel("beta (style weight)")
    plt.ylabel("runtime (s, avg)")
    plt.title("Runtime vs beta")
    save_plot(fig, out_dir / "runtime_vs_beta.png")


def plot_runtime_vs_beta_by_size(rows, out_dir: Path):
    rt_by = defaultdict(list)
    sizes_all = set()
    betas_all = set()
    for r in rows:
        if r["wall_time_s"] is not None and r["beta"] is not None and r["size"] is not None:
            rt_by[(r["beta"], r["size"])].append(r["wall_time_s"])
            sizes_all.add(r["size"])
            betas_all.add(r["beta"])
    if not rt_by:
        return
    sizes = sorted(sizes_all)
    betas = sorted(betas_all)
    x = np.arange(len(betas))
    width = 0.8 / max(len(sizes), 1)
    fig = plt.figure(figsize=(8, 4.5))
    for i, sz in enumerate(sizes):
        means = [np.mean(rt_by.get((b, sz), [0.0])) for b in betas]
        plt.bar(x + (i - len(sizes) / 2) * width + width / 2, means, width=width, label=f"{sz}px")
    plt.xticks(x, [str(b) for b in betas])
    plt.xlabel("beta (style weight)")
    plt.ylabel("runtime (s, avg)")
    plt.title("Runtime vs beta (grouped by size)")
    plt.legend()
    save_plot(fig, out_dir / "runtime_vs_beta_by_size.png")


def plot_runtime_per_step(rows, out_dir: Path):
    per_size = defaultdict(list)
    per_beta = defaultdict(list)
    for r in rows:
        if r["wall_time_s"] is None or not r["steps"]:
            continue
        per = r["wall_time_s"] / max(int(r["steps"]), 1)
        if r["size"] is not None:
            per_size[r["size"]].append(per)
        if r["beta"] is not None:
            per_beta[r["beta"]].append(per)

    if per_size:
        sizes = sorted(per_size.keys())
        means = [np.mean(per_size[s]) for s in sizes]
        fig = plt.figure(figsize=(6, 4))
        plt.bar([str(s) for s in sizes], means)
        plt.xlabel("image size (px)")
        plt.ylabel("runtime per step (s, avg)")
        plt.title("Runtime per step vs size")
        save_plot(fig, out_dir / "runtime_per_step_vs_size.png")

    if per_beta:
        betas = sorted(per_beta.keys())
        means = [np.mean(per_beta[b]) for b in betas]
        fig = plt.figure(figsize=(6, 4))
        plt.bar([str(b) for b in betas], means)
        plt.xlabel("beta (style weight)")
        plt.ylabel("runtime per step (s, avg)")
        plt.title("Runtime per step vs beta")
        save_plot(fig, out_dir / "runtime_per_step_vs_beta.png")


def plot_ssim_box_by_size(rows, out_dir: Path):
    ssim_vals = defaultdict(list)  # key: (size, beta)
    sizes_all, betas_all = set(), set()
    for r in rows:
        if r["ssim"] is None or r["size"] is None or r["beta"] is None:
            continue
        ssim_vals[(r["size"], r["beta"])].append(r["ssim"]) 
        sizes_all.add(r["size"]); betas_all.add(r["beta"])

    if not ssim_vals:
        return

    betas = sorted(betas_all)
    for sz in sorted(sizes_all):
        data = [ssim_vals.get((sz, b), []) for b in betas]
        if not any(len(d) for d in data):
            continue
        fig = plt.figure(figsize=(8, 4.5))
        plt.boxplot([d if d else [None] for d in data], labels=[str(b) for b in betas], showmeans=True)
        plt.xlabel("beta (style weight)")
        plt.ylabel("SSIM vs content")
        plt.title(f"SSIM distribution by beta — {sz}px")
        save_plot(fig, out_dir / f"ssim_box_by_beta_{sz}px.png")


def plot_ssim_scatter(rows, out_dir: Path):
    sizes = sorted(set(r["size"] for r in rows if r["size"] is not None))
    if not sizes:
        return
    cmap = {s: cm.tab10(i % 10) for i, s in enumerate(sizes)}
    fig = plt.figure(figsize=(7, 5))
    for r in rows:
        if r["ssim"] is None or r["beta"] is None or r["size"] is None:
            continue
        jitter = (np.random.rand() - 0.5) * 0.05
        plt.scatter(r["beta"] + jitter, r["ssim"], s=18, alpha=0.6, color=cmap[r["size"]])
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=f"{s}px", markerfacecolor=cmap[s], markersize=6) for s in sizes]
    plt.legend(handles=handles, title="size")
    plt.xlabel("beta (style weight)")
    plt.ylabel("SSIM vs content")
    plt.title("SSIM vs beta (scatter across pairs)")
    save_plot(fig, out_dir / "ssim_vs_beta_scatter_by_pair.png")


def write_top_summary(rows, out_csv: Path, k: int = 20):
    ranked = sorted([r for r in rows if r["ssim"] is not None], key=lambda x: x["ssim"], reverse=True)[:k]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "content", "style", "size", "beta", "alpha", "gamma", "lr", "steps", "ssim", "runtime_s", "final_image"])
        for i, r in enumerate(ranked, 1):
            w.writerow([
                i,
                r["content"], r["style"], r["size"], r["beta"], r["alpha"], r["gamma"], r["lr"], r["steps"],
                f"{r['ssim']:.6f}" if r["ssim"] is not None else "",
                f"{(r['wall_time_s'] or 0.0):.3f}",
                r.get("final_image") or "",
            ])
    print(f"[Analysis] Saved summary: {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Generate analysis plots from metrics.csv")
    ap.add_argument("--csv", default=str(Path("metrics") / "metrics.csv"), help="Path to metrics CSV")
    ap.add_argument("--out", default=str(Path("outputs") / "plots"), help="Directory to write plots")
    ap.add_argument("--top-k", type=int, default=20, help="Top-k for SSIM summary CSV")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return 1

    rows = load_rows(csv_path)
    if not rows:
        print("No rows found in CSV; nothing to analyze.")
        return 0

    plot_ssim_vs_beta_by_size(rows, out_dir)
    plot_runtime_vs_size(rows, out_dir)
    plot_runtime_vs_beta(rows, out_dir)
    plot_runtime_vs_beta_by_size(rows, out_dir)
    plot_runtime_per_step(rows, out_dir)
    plot_ssim_box_by_size(rows, out_dir)
    plot_ssim_scatter(rows, out_dir)

    # Summary CSV next to the metrics file
    write_top_summary(rows, csv_path.parent / "summary_top_by_ssim.csv", k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

