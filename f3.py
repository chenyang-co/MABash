import numpy as np
import matplotlib.pyplot as plt

# ================= 数据 =================
methods = [
    'Single-LLM',
    'w/o Lexical Agent',
    'w/o Syntax Agent',
    'w/o Semantic Agent',
    'MABash (Full)'
]

metrics = {
    'BLEU-1': [54.85, 57.02, 55.83, 56.41, 59.39],
    'BLEU-4': [27.89, 31.46, 30.12, 30.87, 34.32],
    'ROUGE-L': [51.43, 54.31, 53.02, 53.78, 56.29],
    'METEOR': [29.75, 31.24, 30.21, 30.88, 32.91],
}

baseline_idx = 0
x = np.arange(len(methods))

# ================= 颜色（学术蓝灰） =================
colors = [
    '#C7D6E0',  # ablations
    '#C7D6E0',
    '#C7D6E0',
    '#C7D6E0',
    '#4A6FA5'   # Ours
]

# ================= 绘图 =================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (metric, values) in zip(axes, metrics.items()):
    baseline = values[baseline_idx]

    # 画基线
    ax.axhline(
        baseline,
        linestyle='--',
        linewidth=1.5,
        color='#9AA5AE',
        alpha=0.9,
        label='Single-LLM'
    )

    # 画点
    for i, v in enumerate(values):
        ax.scatter(
            i, v,
            s=120 if i == len(values) - 1 else 90,
            color=colors[i],
            edgecolor='black',
            linewidth=0.8,
            zorder=3
        )

        # 标注提升
        if i != baseline_idx:
            delta = v - baseline
            ax.text(
                i, v + 0.15,
                f'+{delta:.2f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='#2c3e50'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.25)

# ================= 全局设置 =================
fig.suptitle(
    'Ablation Study of Multi-Agent Components',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figure_ablation_dotplot.png', dpi=300, bbox_inches='tight')
plt.show()
