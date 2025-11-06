# 保存为 tools/viz_results.py 后运行：
#   python tools/viz_results.py
#
# 该脚本会从以下默认路径读取 eval 输出的 JSON：
#   runs/exp_tcn/test_metrics.json
#   runs/trans_raw/test_metrics.json
#   runs/trans_kine/test_metrics.json
# 你也可以在脚本顶部修改 default_paths 指向新的 JSON 文件。

import os, json, math, textwrap
from pathlib import Path
import matplotlib.pyplot as plt

# ====== 你可以修改这里来指定不同实验的JSON路径 ======
default_paths = {
    "TCN+Gating+ATP": r"F:\ZMX\Chapter4\Encoder_exp\runs\exp_tcn\test_metrics.json",
    "Transformer-Raw": r"F:\ZMX\Chapter4\Encoder_exp\runs\trans_raw\test_metrics.json",
    "Transformer-Kine": r"F:\ZMX\Chapter4\Encoder_exp\runs\trans_kine\test_metrics.json",
    "TCN++": r"F:\ZMX\Chapter4\Encoder_exp\runs\exp++\test_metrics.json",
    "Transformer-Kinepro": r"F:\ZMX\Chapter4\Encoder_exp\runs\trans_kine_80epoch\test_metrics.json"
}
# ==================================================

out_dir = Path("figures_2")
out_dir.mkdir(parents=True, exist_ok=True)

def load_results(path_map):
    results, missing = {}, {}
    for name, p in path_map.items():
        pth = Path(p)
        if pth.exists():
            with open(pth, "r", encoding="utf-8") as f:
                try:
                    results[name] = json.load(f)
                except Exception as e:
                    missing[name] = f"JSON parse error: {e}"
        else:
            missing[name] = "file not found"
    return results, missing

def gather_metric_matrix(results, keys):
    models = list(results.keys())
    matrix = {k: [] for k in keys}
    for m in models:
        res = results[m]
        for k in keys:
            matrix[k].append(float(res.get(k, float("nan"))))
    return models, matrix

def bar_with_values(ax, x, heights, labels):
    bars = ax.bar(x, heights)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    for rect, h in zip(bars, heights):
        ax.annotate(f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

def save_fig(fig, fname):
    fpath = out_dir / fname
    fig.tight_layout()
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)

def write_table_csv(models, matrix, fname="results_table.csv"):
    headers = ["Model"] + list(matrix.keys())
    lines = [",".join(headers)]
    for i, m in enumerate(models):
        row = [m] + [f"{matrix[k][i]:.4f}" for k in matrix.keys()]
        lines.append(",".join(row))
    f = out_dir / fname
    f.write_text("\n".join(lines), encoding="utf-8")
    return str(f)

def write_table_latex(models, matrix, fname="results_table.tex"):
    metrics = list(matrix.keys())
    header = " & ".join(["Model"] + metrics) + " \\\\"
    lines = [
        "\\begin{tabular}{l" + "c"*len(metrics) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for i, m in enumerate(models):
        row = " & ".join([m] + [f"{matrix[k][i]:.3f}" for k in metrics]) + " \\\\"
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    f = out_dir / fname
    f.write_text("\n".join(lines), encoding="utf-8")
    return str(f)

def summarize_conclusions(models, matrix, primary_metric="top1"):
    best_idx = max(range(len(models)), key=lambda i: matrix[primary_metric][i])
    best_model = models[best_idx]
    best_val = matrix[primary_metric][best_idx]
    try:
        base_idx = models.index("Transformer-Raw")
    except ValueError:
        base_idx = 0
    base_model = models[base_idx]
    base_val = matrix[primary_metric][base_idx]
    imp = (best_val - base_val) / (base_val + 1e-12) * 100.0

    lines = []
    lines.append(f"• 主结论：在 {primary_metric.upper()} 指标上，**{best_model}** 取得最高分 {best_val:.3f}。")
    if best_idx != base_idx:
        lines.append(f"• 相比基线 **{base_model}**（{base_val:.3f}），提升 **{imp:.2f}%**。")
    for k in matrix.keys():
        if k == primary_metric:
            continue
        best_k_idx = max(range(len(models)), key=lambda i: matrix[k][i])
        if best_k_idx == best_idx:
            lines.append(f"• 在 {k} 上同样领先，数值为 {matrix[k][best_k_idx]:.3f}。")
        else:
            lines.append(f"• 在 {k} 上最优为 **{models[best_k_idx]}**（{matrix[k][best_k_idx]:.3f}），但 **{best_model}** 仍保持竞争力。")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

def main():
    results, missing = load_results(default_paths)
    if not results:
        msg = "未找到任何评估结果文件。请先运行 eval 生成 JSON，或修改 default_paths。\n" + \
              "\n".join([f"- {name}: {default_paths[name]} -> {why}" for name, why in missing.items()])
        (out_dir / "README_VIZ.txt").write_text(msg, encoding="utf-8")
        print(msg)
        return

    metric_keys = ["top1", "top5", "auc", "hungarian_acc", "mofa"]
    models, matrix = gather_metric_matrix(results, metric_keys)

    # 单图单指标（论文清晰展示）
    fig1 = plt.figure(); ax1 = fig1.add_subplot(111)
    bar_with_values(ax1, range(len(models)), matrix["top1"], models)
    ax1.set_ylabel("Top-1 Accuracy"); ax1.set_title("Retrieval Top-1")
    save_fig(fig1, "fig_top1.png")

    fig2 = plt.figure(); ax2 = fig2.add_subplot(111)
    bar_with_values(ax2, range(len(models)), matrix["top5"], models)
    ax2.set_ylabel("Top-5 Accuracy"); ax2.set_title("Retrieval Top-5")
    save_fig(fig2, "fig_top5.png")

    fig3 = plt.figure(); ax3 = fig3.add_subplot(111)
    bar_with_values(ax3, range(len(models)), matrix["auc"], models)
    ax3.set_ylabel("ROC-AUC"); ax3.set_title("Binary Discrimination (AUC)")
    save_fig(fig3, "fig_auc.png")

    fig4 = plt.figure(); ax4 = fig4.add_subplot(111)
    bar_with_values(ax4, range(len(models)), matrix["hungarian_acc"], models)
    ax4.set_ylabel("Hungarian Assignment Accuracy"); ax4.set_title("Global 1-1 Assignment")
    save_fig(fig4, "fig_hungarian.png")

    fig5 = plt.figure(); ax5 = fig5.add_subplot(111)
    bar_with_values(ax5, range(len(models)), matrix["mofa"], models)
    ax5.set_ylabel("MOFA (Fusion Accuracy)"); ax5.set_title("Multi-Object Fusion Accuracy (Proxy)")
    save_fig(fig5, "fig_mofa.png")

    # 总览对比（单图，无子图）：按指标维度分组
    metrics_order = metric_keys
    fig6 = plt.figure(); ax6 = fig6.add_subplot(111)
    M = len(models); X = list(range(len(metrics_order))); width = 0.8 / max(1, M)
    for j, m in enumerate(models):
        vals = [matrix[k][j] for k in metrics_order]
        xj = [x + (j - (M-1)/2)*width for x in X]
        bars = ax6.bar(xj, vals, width=width, label=m)
        for rect, h in zip(bars, vals):
            ax6.annotate(f"{h:.3f}", xy=(rect.get_x()+rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    ax6.set_xticks(X); ax6.set_xticklabels(metrics_order)
    ax6.set_ylabel("Score"); ax6.set_title("Overall Comparison across Metrics")
    ax6.legend()
    save_fig(fig6, "fig_overall_grouped.png")

    # 表格
    write_table_csv(models, matrix, fname="results_table.csv")
    write_table_latex(models, matrix, fname="results_table.tex")

    # 自动摘要（中文）
    summarize_conclusions(models, matrix, primary_metric="top1")
    print("完成，可在 figures/ 目录查看图表与表格。")

if __name__ == "__main__":
    main()
