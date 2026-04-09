import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# =========================
# 数据准备
# =========================
methods = [
    'LSI', 'VSM', 'BM25', 'NNGen', 'CCGIR',
    'Transformer', 'CODE-NN', 'Hybrid-DeepCom', 'Rencos', 'SG-Trans',
    'CODESCRIBE', 'CoSS', 'CodeBERT', 'PLBART', 'UniXcoder',
    'BASHEXPLAINER', 'Bash2Com', 'HBCom',
    'MABash'
]

bleu4_scores = [
    9.40, 15.25, 19.24, 27.85, 29.43,
    19.97, 24.17, 22.75, 24.39, 28.27,
    29.19, 30.74, 24.83, 27.55, 27.25,
    29.13, 31.68, 32.76,
    34.32
]

categories = ['IR-Based', 'DL-Based', 'Pre-trained', 'Bash-Specific', 'MABash']
category_indices = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17],
    [18]
]

# =========================
# 构建热力图矩阵
# =========================
heatmap_data = np.full((5, 6), np.nan)

for i, indices in enumerate(category_indices):
    scores = [bleu4_scores[j] for j in indices]
    for k, s in enumerate(scores[:5]):
        heatmap_data[i, k] = s
    heatmap_data[i, 5] = np.mean(scores)

# =========================
# 论文级淡色 GnBu colormap（稳定）
# =========================
base = plt.cm.GnBu(np.linspace(0.05, 0.65, 256))
cmap = mcolors.LinearSegmentedColormap.from_list("GnBu_ultralight", base)
cmap.set_bad("white")

# =========================
# 绘图
# =========================
fig, ax = plt.subplots(figsize=(14, 8))

mask = np.isnan(heatmap_data)

sns.heatmap(
    heatmap_data,
    mask=mask,
    cmap=cmap,
    annot=True,
    fmt=".1f",
    linewidths=1.2,
    linecolor="#B0B0B0",
    square=True,
    vmin=20, vmax=35,
    cbar_kws={
        "label": "BLEU-4 Score",
        "orientation": "horizontal",
        "pad": 0.12
    },
    ax=ax
)

# =========================
# 坐标轴与标签
# =========================
ax.set_xticklabels(
    ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5', 'Avg'],
    fontsize=11,
    fontweight='bold'
)
ax.set_yticklabels(categories, rotation=0,fontsize=12, fontweight='bold')
# ax.set_xlabel("Methods within Each Category", fontsize=12, fontweight='bold')
# ax.set_ylabel("Method Categories", fontsize=12, fontweight='bold')

# 类别分割线
for i in range(1, 5):
    ax.axhline(i, color='black', linewidth=2)

# =========================
# 方法缩写标注
# =========================
abbreviations = [
    ['LSI', 'VSM', 'BM25', 'NNGen', 'CCGIR'],
    ['Trans.', 'CODE-NN', 'Hybrid', 'Rencos', 'SG-Trans'],
    ['CODESCR.', 'CoSS', 'CodeBERT', 'PLBART', 'UniXcoder'],
    ['BASHEX', 'Bash2Com', 'HBCom', '', ''],
    ['MABash', '', '', '', '']
]

for i in range(5):
    for j in range(5):
        if not np.isnan(heatmap_data[i, j]):
            ax.text(
                j + 0.5, i + 0.78,
                abbreviations[i][j],
                ha='center', va='center',
                fontsize=9, fontweight='bold', color='black'
            )

# =========================
# 高亮 Ours
# =========================
ax.add_patch(
    plt.Rectangle(
        (0, 4), 6, 1,
        fill=False,
        edgecolor="#D55E00",
        linewidth=3,
        linestyle='--'
    )
)

plt.tight_layout()
plt.savefig(
    "f51.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
