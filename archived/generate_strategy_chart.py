#!/usr/bin/env python3
"""
生成二维策略坐标系图
X轴: 估值百分位 (L1极度低估 → L9极度高估)
Y轴: 趋势强度 (T1极度熊市 → T9极度牛市)
色块/曲线: 目标仓位%
"""

import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "Hiragino Sans GB"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# 9x9仓位矩阵 (行=L1-L9估值, 列=T1-T9趋势)
ALLOCATION_MATRIX = [
    [0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.55, 0.45, 0.35],  # L1 极度低估
    [0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30],  # L2 低估
    [0.85, 0.80, 0.75, 0.70, 0.65, 0.55, 0.45, 0.35, 0.25],  # L3 偏低估
    [0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20],  # L4 中性偏低
    [0.70, 0.65, 0.60, 0.55, 0.50, 0.40, 0.30, 0.25, 0.15],  # L5 中性
    [0.60, 0.55, 0.50, 0.45, 0.40, 0.30, 0.25, 0.20, 0.10],  # L6 中性偏高
    [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10],  # L7 偏高估
    [0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.12, 0.08, 0.05],  # L8 高估 (fixed: col5 was 0.35, should be monotonic)
    [0.30, 0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.03],  # L9 极度高估
]

matrix = np.array(ALLOCATION_MATRIX) * 100  # 转为百分比

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#0d1117')

# ==============================
# 左图: 热力图
# ==============================
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

# 自定义颜色: 红(低仓) → 黄 → 绿(高仓)
colors = ['#d32f2f', '#f57c00', '#fbc02d', '#aed581', '#43a047', '#1b5e20']
cmap = LinearSegmentedColormap.from_list('position', colors, N=256)

im = ax1.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100,
                origin='upper', alpha=0.9)

# 在每个格子里写数字
for i in range(9):
    for j in range(9):
        val = matrix[i, j]
        color = 'white' if val < 40 or val > 75 else '#1a1a2e'
        ax1.text(j, i, f'{val:.0f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

# 标签
trend_labels = ['T1\n极熊', 'T2\n深熊', 'T3\n弱熊', 'T4\n偏熊', 'T5\n猴市',
                'T6\n偏牛', 'T7\n弱牛', 'T8\n强牛', 'T9\n极牛']
val_labels = ['L1\n极低估\n<10%', 'L2\n低估\n10-20%', 'L3\n偏低估\n20-35%',
              'L4\n中性偏低\n35-45%', 'L5\n中性\n45-55%', 'L6\n中性偏高\n55-65%',
              'L7\n偏高估\n65-75%', 'L8\n高估\n75-88%', 'L9\n极高估\n>88%']

ax1.set_xticks(range(9))
ax1.set_xticklabels(trend_labels, fontsize=8, color='#e0e0e0')
ax1.set_yticks(range(9))
ax1.set_yticklabels(val_labels, fontsize=7.5, color='#e0e0e0')

ax1.set_xlabel('趋势强度 (T轴) →  熊市 ←→ 牛市', fontsize=12,
               color='#90caf9', fontweight='bold', labelpad=10)
ax1.set_ylabel('← 低估      估值百分位 (L轴)      高估 →', fontsize=12,
               color='#ffcc80', fontweight='bold', labelpad=10)
ax1.set_title('ETF双维度量化策略 — 目标仓位矩阵', fontsize=14,
              color='white', fontweight='bold', pad=15)

# 颜色条
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('目标仓位 (%)', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.ax.set_facecolor('#0d1117')

# 添加当前状态标记 (今日: L8 x T5 = 35%)
ax1.add_patch(plt.Rectangle((4.5-0.5, 7.5-0.5), 1, 1,
              fill=False, edgecolor='#ff6b6b', linewidth=3, linestyle='--'))
ax1.text(4, 7, '▶ 今日\nL8×T5\n35%', ha='center', va='center',
        fontsize=8, color='#ff6b6b', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', alpha=0.8))

# ==============================
# 右图: 仓位曲线图
# ==============================
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.spines['bottom'].set_color('#555')
ax2.spines['top'].set_color('#555')
ax2.spines['left'].set_color('#555')
ax2.spines['right'].set_color('#555')
ax2.tick_params(colors='#e0e0e0')

x = np.arange(1, 10)  # L1~L9估值等级
trend_curves = {
    'T9 极牛市': (matrix[:, 8], '#ef5350', '--'),
    'T7 弱牛市': (matrix[:, 6], '#ff8a65', '-.'),
    'T5 猴市':   (matrix[:, 4], '#ffd54f', '-'),
    'T3 弱熊市': (matrix[:, 2], '#81c784', '-.'),
    'T1 极熊市': (matrix[:, 0], '#4fc3f7', '--'),
}

for label, (y_vals, color, ls) in trend_curves.items():
    ax2.plot(x, y_vals, color=color, linewidth=2.5, linestyle=ls,
             marker='o', markersize=5, label=label, alpha=0.9)

# 区域填充
ax2.fill_between(x, matrix[:, 8], matrix[:, 0],
                alpha=0.08, color='#90caf9', label='_nolegend_')

# 参考线
ax2.axhline(y=80, color='#ef9a9a', linewidth=1, linestyle=':', alpha=0.6)
ax2.axhline(y=50, color='#ffe082', linewidth=1, linestyle=':', alpha=0.6)
ax2.axhline(y=20, color='#a5d6a7', linewidth=1, linestyle=':', alpha=0.6)
ax2.text(9.1, 80, '高仓区\n80%', color='#ef9a9a', fontsize=8, va='center')
ax2.text(9.1, 50, '均衡\n50%', color='#ffe082', fontsize=8, va='center')
ax2.text(9.1, 20, '低仓区\n20%', color='#a5d6a7', fontsize=8, va='center')

# 三个市场区域标注
ax2.axvspan(0.5, 3.5, alpha=0.06, color='#4fc3f7', label='_nolegend_')
ax2.axvspan(3.5, 6.5, alpha=0.06, color='#ffd54f', label='_nolegend_')
ax2.axvspan(6.5, 9.5, alpha=0.06, color='#ef5350', label='_nolegend_')
ax2.text(2, 97, '低估区', color='#4fc3f7', fontsize=10, ha='center', alpha=0.8)
ax2.text(5, 97, '中性区', color='#ffd54f', fontsize=10, ha='center', alpha=0.8)
ax2.text(8, 97, '高估区', color='#ef9a9a', fontsize=10, ha='center', alpha=0.8)

# 今日状态标注
ax2.scatter([8], [35], s=200, color='#ff6b6b', zorder=10, marker='*')
ax2.annotate('今日状态\nL8×T5=35%', xy=(8, 35), xytext=(6.5, 25),
            color='#ff6b6b', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.9))

ax2.set_xlim(0.5, 9.5)
ax2.set_ylim(0, 102)
ax2.set_xticks(range(1, 10))
ax2.set_xticklabels([f'L{i}' for i in range(1, 10)], color='#e0e0e0', fontsize=10)
ax2.set_yticks(range(0, 101, 10))
ax2.set_yticklabels([f'{i}%' for i in range(0, 101, 10)], color='#e0e0e0', fontsize=9)
ax2.set_xlabel('估值百分位等级 (L轴)   低估 ←————→ 高估', fontsize=12,
               color='#ffcc80', fontweight='bold', labelpad=10)
ax2.set_ylabel('目标仓位 (%)', fontsize=12, color='#90caf9',
               fontweight='bold', labelpad=10)
ax2.set_title('不同市场趋势下的仓位曲线', fontsize=14,
              color='white', fontweight='bold', pad=15)

legend = ax2.legend(loc='upper right', fontsize=10, framealpha=0.3,
                    facecolor='#1a1a2e', edgecolor='#555',
                    labelcolor='white')

ax2.grid(True, alpha=0.15, color='#555')

# 总标题
fig.suptitle('A股ETF双维度量化波段策略 · 仓位决策系统', 
             fontsize=16, color='white', fontweight='bold', y=1.01)

plt.tight_layout(pad=2)
plt.savefig('/tmp/etf-trader/strategy_matrix.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("✅ 策略图已生成: /tmp/etf-trader/strategy_matrix.png")
