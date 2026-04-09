import matplotlib.pyplot as plt
import numpy as np

# 命令复杂度
categories = ["Simple", "Medium", "Complex"]
x = np.arange(len(categories))
width = 0.35

# S-BLEU 数据
ours = np.array([38.1227, 36.2873, 41.5914])
hbcom = np.array([21.9829, 23.7177, 24.7490])
bash2com = np.array([27.2011, 28.4094, 27.6870])

# 提升幅度
gain_hbcom = ours - hbcom
gain_bash2com = ours - bash2com

# 画图
plt.figure(figsize=(7, 4.5))

plt.bar(x - width/2, gain_hbcom, width, label="Ours − HBCom")
plt.bar(x + width/2, gain_bash2com, width, label="Ours − Bash2Com")

# 坐标轴
plt.xticks(x, categories)
plt.ylabel("S-BLEU Improvement (%)", fontsize=12)
plt.xlabel("Command Complexity", fontsize=12)

# 网格
plt.grid(axis="y", linestyle="--", alpha=0.6)

# 图例
plt.legend(frameon=True)

# 标题（可选）
plt.title("S-BLEU Improvement over Baselines across Command Complexity", fontsize=13)

# 紧凑布局
plt.tight_layout()

# 保存

plt.savefig("f7.png", dpi=300)

plt.show()
