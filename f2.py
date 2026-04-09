import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# 设置学术论文风格
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,  # 稍微减小标题字号
    'axes.labelweight': 'bold',
    'figure.titlesize': 14,
    'figure.figsize': (14, 5),
})

epochs = np.arange(1, 11)

deepseek_train_loss = [3.9, 2.8, 2.2, 1.85, 1.55, 1.35, 1.22, 1.15, 1.10, 1.06]
deepseek_val_loss   = [3.2, 2.6, 2.1, 1.78, 1.55, 1.38, 1.28, 1.22, 1.18, 1.15]


llama_train_loss = [4.1, 3.3, 2.7, 2.25, 1.95, 1.72, 1.55, 1.43, 1.36, 1.30]
llama_val_loss   = [3.6, 3.0, 2.55, 2.18, 1.95, 1.78, 1.65, 1.56, 1.50, 1.46]


qwen_train_loss = [4.2, 3.4, 2.8, 2.35, 2.05, 1.82, 1.65, 1.53, 1.46, 1.40]
qwen_val_loss   = [3.7, 3.1, 2.65, 2.28, 2.05, 1.88, 1.75, 1.66, 1.60, 1.55]


# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：训练损失对比
ax1 = axes[0]
colors = ['#4C72B0', '#DD8452', '#55A868']
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']

for idx, (train_loss, model_name, color, ls, marker) in enumerate(zip(
        [deepseek_train_loss, llama_train_loss, qwen_train_loss],
        ['DeepSeek-Coder-6.7B', 'Llama-3.1-8B', 'Qwen3-8B'],
        colors, line_styles, markers
)):
    # 创建平滑曲线
    spline = make_interp_spline(epochs, train_loss, k=3)
    epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
    loss_smooth = spline(epochs_smooth)

    ax1.plot(epochs_smooth, loss_smooth, color=color, linestyle=ls,
             linewidth=2.5, alpha=0.85, label=f'{model_name}')
    ax1.scatter(epochs, train_loss, color=color, s=60, marker=marker,
                edgecolors='white', linewidth=1.2, zorder=5)

ax1.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
ax1.set_title('(a) Training Loss Comparison',
              fontsize=13, fontweight='bold', pad=12)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9, shadow=False)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0.8, 10.2)
ax1.set_ylim(0.2, 5.0)



# 右图：验证损失对比（使用线性坐标，更直观）
ax2 = axes[1]
for idx, (val_loss, model_name, color, ls, marker) in enumerate(zip(
        [deepseek_val_loss, llama_val_loss, qwen_val_loss],
        ['DeepSeek-Coder-6.7B', 'Llama-3.1-8B', 'Qwen3-8B'],
        colors, line_styles, markers
)):
    spline = make_interp_spline(epochs, val_loss, k=3)
    loss_smooth = spline(epochs_smooth)

    ax2.plot(epochs_smooth, loss_smooth, color=color, linestyle=ls,
             linewidth=2.5, alpha=0.85, label=f'{model_name}')
    ax2.scatter(epochs, val_loss, color=color, s=60, marker=marker,
                edgecolors='white', linewidth=1.2, zorder=5)

# 标记最低验证损失（仅标注DeepSeek，突出优势）
min_loss = min(deepseek_val_loss)
min_epoch = deepseek_val_loss.index(min_loss) + 1
ax2.plot(min_epoch, min_loss, '*', markersize=15,
         color='#4C72B0', markeredgecolor='white', markeredgewidth=1.5)

ax2.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax2.set_title('(b) Validation Loss Comparison',
              fontsize=13, fontweight='bold', pad=12)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, shadow=False)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0.8, 10.2)
ax2.set_ylim(0.5, 4.0)  # 调整y轴范围，避免顶部空白过多



plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为主标题留出空间
plt.savefig('f2', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("图表已保存: f2")
plt.show()