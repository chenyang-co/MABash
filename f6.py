import matplotlib.pyplot as plt

# 数据
categories = ["Simple", "Medium", "Complex"]

hbcom = [34.7972,37.2873, 38.1892]
bash2com = [27.2011, 28.4094, 27.6870]
ours = [37.5055,38.1227 , 41.5914]

# 画图
plt.figure(figsize=(7, 4.5))

plt.plot(categories, hbcom, marker='o', linewidth=2.2, markersize=7,
         markeredgewidth=1.2, label="HBCom")
plt.plot(categories, bash2com, marker='s', linewidth=2.2, markersize=7,
         markeredgewidth=1.2, label="Bash2Com")
plt.plot(categories, ours, marker='^', linewidth=2.2, markersize=7,
         markeredgewidth=1.2, label="MABash")

# 数值标注函数
def annotate_values(x, y):
    for xi, yi in zip(x, y):
        plt.annotate(
            f"{yi:.2f}",
            xy=(xi, yi),
            xytext=(0, 6),  # 向上偏移
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

# 添加数值标注
annotate_values(categories, hbcom)
annotate_values(categories, bash2com)
annotate_values(categories, ours)

# 坐标轴与标签
plt.ylabel("S-BLEU (%)", fontsize=12)
plt.xlabel("Command Complexity", fontsize=12)
plt.ylim(20, 45)

# 网格（浅色，论文常见）
plt.grid(True, linestyle="--", alpha=0.5)

# 图例
plt.legend(frameon=True, fontsize=10)


# 紧凑布局
plt.tight_layout()

# 保存（矢量友好）
plt.savefig("f6.png", dpi=300, bbox_inches="tight")

plt.show()
