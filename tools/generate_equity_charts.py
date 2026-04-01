#!/usr/bin/env python3
"""
Generate equity curve charts for ALL strategies.
Outputs PNG files to docs/charts/ for README embedding.
v2.0 - Improved visual quality with aligned start dates.
"""
import subprocess
import sys
import os
import json
import csv
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib', 'numpy'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

# CJK font setup
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STSong', 'PingFang SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(ROOT, 'docs', 'charts')
DATA_DIR = os.path.join(ROOT, 'strategies', 'data')
os.makedirs(CHART_DIR, exist_ok=True)

# Strategy metadata
STRATEGIES = {
    'A': {'name': '债券分散持有', 'cagr': 4.1, 'mdd': 2.1, 'sharpe': 1.54, 'calmar': 1.95},
    'B': {'name': '因子ETF轮动', 'cagr': 15.6, 'mdd': 18.3, 'sharpe': 1.157, 'calmar': 0.85},
    'C': {'name': '跨境ETF纯动量', 'cagr': 17.0, 'mdd': 23.3, 'sharpe': 0.61, 'calmar': 0.73},
    'D': {'name': 'CSI300低波动', 'cagr': 13.9, 'mdd': 8.5, 'sharpe': 0.76, 'calmar': 1.64},
    'E': {'name': '逆向价值+质量', 'cagr': 9.7, 'mdd': 17.9, 'sharpe': 0.49, 'calmar': 0.54},
    'F': {'name': '美股QDII动量', 'cagr': 13.0, 'mdd': 18.9, 'sharpe': 0.77, 'calmar': 0.69},
    'G': {'name': 'CSI500低波价值', 'cagr': 4.4, 'mdd': 15.8, 'sharpe': 0.19, 'calmar': 0.28},
    'H': {'name': '指数超跌买入', 'cagr': 13.9, 'mdd': 6.6, 'sharpe': 1.04, 'calmar': 2.11},
    'L': {'name': '300成长MA60趋势', 'cagr': 18.9, 'mdd': 31.7, 'sharpe': 0.49, 'calmar': 0.60},
    'M': {'name': 'CSI800低换手+动量', 'cagr': 21.4, 'mdd': 25.4, 'sharpe': 0.97, 'calmar': 0.84},
    'Q': {'name': '红利低波DY利差择时', 'cagr': 17.6, 'mdd': 16.9, 'sharpe': None, 'calmar': 1.04},
}

# Color palette
COLORS = {
    'A': '#EF5350', 'B': '#AB47BC', 'C': '#5C6BC0', 'D': '#29B6F6',
    'E': '#26A69A', 'F': '#66BB6A', 'G': '#FFA726', 'H': '#8D6E63',
    'L': '#42A5F5', 'M': '#EC407A', 'Q': '#7E57C2',
}


def read_equity_csv(letter):
    """Read equity curve from CSV."""
    path = os.path.join(DATA_DIR, f'strategy_{letter.lower()}_equity.csv')
    if not os.path.exists(path):
        return None
    dates, values = [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row['date']
            try:
                dt = datetime.strptime(d, '%Y%m%d')
            except ValueError:
                try:
                    dt = datetime.strptime(d, '%Y-%m-%d')
                except ValueError:
                    continue
            dates.append(dt)
            values.append(float(row['value']))
    return dates, values


def generate_single_chart(letter, dates, values, meta):
    """Generate individual equity curve chart."""
    fig, ax = plt.subplots(figsize=(12, 5))

    v0 = values[0]
    norm = [v / v0 for v in values]

    color = COLORS.get(letter, '#2196F3')
    ax.plot(dates, norm, linewidth=1.5, color=color)
    ax.fill_between(dates, norm, alpha=0.15, color=color)

    # Drawdown overlay
    peak = np.maximum.accumulate(norm)
    dd = [(n / p - 1) * 100 for n, p in zip(norm, peak)]
    ax2 = ax.twinx()
    ax2.fill_between(dates, dd, alpha=0.2, color='#F44336', label='Drawdown')
    ax2.set_ylabel('Drawdown (%)', color='#F44336', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#F44336')
    ax2.set_ylim(min(dd) * 1.5, 5)

    ax.set_title(f"Strategy {letter}: {meta['name']}", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('净值 (起始=1.0)', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(dates[0], dates[-1])

    # Stats box
    final_val = norm[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    stats = (f"终值: {final_val:.2f}x  |  年化: {meta['cagr']}%  |  "
             f"最大回撤: {meta['mdd']}%  |  Calmar: {meta['calmar']:.2f}")
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    path = os.path.join(CHART_DIR, f'strategy_{letter.lower()}_equity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_comparison_chart(all_curves):
    """Generate combined comparison chart - split into 2 panels by start date."""
    # Split into long-history (pre-2015) and short-history groups
    long_curves = [(l, d, v, m) for l, d, v, m in all_curves if d[0].year < 2015]
    short_curves = [(l, d, v, m) for l, d, v, m in all_curves if d[0].year >= 2015]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1]})

    for ax, curves, title in [
        (axes[0], long_curves, '长周期策略 (2005-2026)'),
        (axes[1], short_curves, '中短周期策略 (2015-2026)'),
    ]:
        if not curves:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        for letter, dates, values, meta in curves:
            v0 = values[0]
            norm = [v / v0 for v in values]
            color = COLORS.get(letter, '#666')
            label = f"{letter}: {meta['name']} (年化{meta['cagr']}%)"
            ax.plot(dates, norm, linewidth=1.8, color=color, label=label, alpha=0.9)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('净值 (起始=1.0)', fontsize=11)
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)

        # Y-axis formatting
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}x' if y < 10 else f'{y:.0f}x'))

    plt.suptitle('ETF Trader - 全策略净值走势对比', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(CHART_DIR, 'all_strategies_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved comparison chart: {path}")
    return path


def generate_metrics_chart():
    """Generate bar chart comparing key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    letters = sorted(STRATEGIES.keys())
    names = [f"  {l}: {STRATEGIES[l]['name']}" for l in letters]
    cagrs = [STRATEGIES[l]['cagr'] for l in letters]
    mdds = [STRATEGIES[l]['mdd'] for l in letters]
    calmars = [STRATEGIES[l]['calmar'] for l in letters]

    y_pos = np.arange(len(letters))

    # CAGR
    colors_cagr = ['#4CAF50' if c >= 15 else '#FFC107' if c >= 10 else '#EF5350' for c in cagrs]
    bars1 = axes[0].barh(y_pos, cagrs, color=colors_cagr, edgecolor='white', height=0.7)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_title('年化收益 CAGR (%)', fontweight='bold', fontsize=13)
    axes[0].set_xlim(0, max(cagrs) * 1.25)
    for i, v in enumerate(cagrs):
        axes[0].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    # MDD
    colors_mdd = ['#4CAF50' if m <= 10 else '#FFC107' if m <= 20 else '#EF5350' for m in mdds]
    axes[1].barh(y_pos, mdds, color=colors_mdd, edgecolor='white', height=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(['' for _ in letters])
    axes[1].set_title('最大回撤 MDD (%)', fontweight='bold', fontsize=13)
    axes[1].set_xlim(0, max(mdds) * 1.25)
    for i, v in enumerate(mdds):
        axes[1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Calmar
    colors_cal = ['#4CAF50' if c >= 1.0 else '#FFC107' if c >= 0.5 else '#EF5350' for c in calmars]
    axes[2].barh(y_pos, calmars, color=colors_cal, edgecolor='white', height=0.7)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(['' for _ in letters])
    axes[2].set_title('Calmar Ratio', fontweight='bold', fontsize=13)
    axes[2].set_xlim(0, max(calmars) * 1.25)
    for i, v in enumerate(calmars):
        axes[2].text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9, fontweight='bold')

    plt.suptitle('ETF Trader - 策略核心指标对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'strategy_metrics_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved metrics chart: {path}")
    return path


def main():
    print("=" * 60)
    print("Generating Equity Curve Charts for All Strategies (v2)")
    print("=" * 60)

    all_curves = []

    for letter in sorted(STRATEGIES.keys()):
        meta = STRATEGIES[letter]
        print(f"\nStrategy {letter}: {meta['name']}")
        result = read_equity_csv(letter)
        if result:
            dates, values = result
            generate_single_chart(letter, dates, values, meta)
            all_curves.append((letter, dates, values, meta))
        else:
            print(f"  No equity data, skipping.")

    if all_curves:
        print(f"\nGenerating comparison chart ({len(all_curves)} strategies)...")
        generate_comparison_chart(all_curves)

    print("\nGenerating metrics comparison chart...")
    generate_metrics_chart()

    print(f"\nDone! Charts saved to: {CHART_DIR}")
    print(f"Total: {len(all_curves)} individual + 1 comparison + 1 metrics")


if __name__ == '__main__':
    main()
