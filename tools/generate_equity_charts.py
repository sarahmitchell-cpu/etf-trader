#!/usr/bin/env python3
"""
Generate equity curve charts for ALL strategies.
Outputs PNG files to docs/charts/ for README embedding.
"""
import subprocess
import sys
import os
import json
import csv
import importlib.util
from datetime import datetime

# Ensure matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

# CJK font setup
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STSong', 'PingFang SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(ROOT, 'docs', 'charts')
STRATEGIES_DIR = os.path.join(ROOT, 'strategies')
os.makedirs(CHART_DIR, exist_ok=True)

# Strategy metadata (name, description, has_equity_csv)
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


def run_strategy_and_capture_equity(letter):
    """Run a strategy script and try to capture its equity curve."""
    script_name = None
    for f in os.listdir(STRATEGIES_DIR):
        if f.startswith(f'strategy_{letter.lower()}_') and f.endswith('.py'):
            script_name = f
            break

    if not script_name:
        return None

    # Check for pre-generated CSV
    csv_path = os.path.join(STRATEGIES_DIR, 'data', f'strategy_{letter.lower()}_equity.csv')
    if os.path.exists(csv_path):
        return read_equity_csv(csv_path)

    # Try running the script with --equity-csv flag (won't work for most, but try)
    print(f"  No equity CSV found for Strategy {letter}, skipping chart.")
    return None


def read_equity_csv(path):
    """Read equity curve from CSV (date,value format)."""
    dates = []
    values = []
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
    """Generate a single strategy equity curve chart."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Normalize to 1.0
    v0 = values[0]
    norm_values = [v / v0 for v in values]

    ax.plot(dates, norm_values, linewidth=1.5, color='#2196F3')
    ax.fill_between(dates, norm_values, alpha=0.1, color='#2196F3')

    ax.set_title(f"Strategy {letter}: {meta['name']}", fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Value (start=1.0)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    ax.set_xlim(dates[0], dates[-1])

    # Add stats box
    stats_text = f"CAGR: {meta['cagr']}%  |  MDD: {meta['mdd']}%  |  Calmar: {meta['calmar']:.2f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(CHART_DIR, f'strategy_{letter.lower()}_equity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_comparison_chart(all_curves):
    """Generate a combined comparison chart of all strategies."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5',
              '#2196F3', '#00BCD4', '#009688', '#4CAF50', '#FF9800',
              '#795548', '#607D8B']

    for i, (letter, dates, values, meta) in enumerate(all_curves):
        v0 = values[0]
        norm = [v / v0 for v in values]
        color = colors[i % len(colors)]
        label = f"{letter}: {meta['name']} (CAGR {meta['cagr']}%)"
        ax.plot(dates, norm, linewidth=1.2, color=color, label=label, alpha=0.85)

    ax.set_title('All Strategies - Normalized Equity Curves', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized Value (start=1.0)')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'all_strategies_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved comparison chart: {path}")
    return path


def generate_metrics_chart():
    """Generate a bar chart comparing key metrics across strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    letters = sorted(STRATEGIES.keys())
    names = [f"{l}" for l in letters]
    cagrs = [STRATEGIES[l]['cagr'] for l in letters]
    mdds = [STRATEGIES[l]['mdd'] for l in letters]
    calmars = [STRATEGIES[l]['calmar'] for l in letters]

    # CAGR bars
    colors_cagr = ['#4CAF50' if c >= 15 else '#FFC107' if c >= 10 else '#F44336' for c in cagrs]
    axes[0].barh(names, cagrs, color=colors_cagr, edgecolor='white')
    axes[0].set_title('CAGR (%)', fontweight='bold')
    axes[0].set_xlim(0, max(cagrs) * 1.2)
    for i, v in enumerate(cagrs):
        axes[0].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)

    # MDD bars
    colors_mdd = ['#4CAF50' if m <= 10 else '#FFC107' if m <= 20 else '#F44336' for m in mdds]
    axes[1].barh(names, mdds, color=colors_mdd, edgecolor='white')
    axes[1].set_title('Max Drawdown (%)', fontweight='bold')
    axes[1].set_xlim(0, max(mdds) * 1.2)
    for i, v in enumerate(mdds):
        axes[1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)

    # Calmar bars
    colors_cal = ['#4CAF50' if c >= 1.0 else '#FFC107' if c >= 0.5 else '#F44336' for c in calmars]
    axes[2].barh(names, calmars, color=colors_cal, edgecolor='white')
    axes[2].set_title('Calmar Ratio', fontweight='bold')
    axes[2].set_xlim(0, max(calmars) * 1.2)
    for i, v in enumerate(calmars):
        axes[2].text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)

    plt.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'strategy_metrics_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved metrics chart: {path}")
    return path


def main():
    print("=" * 60)
    print("Generating Equity Curve Charts for All Strategies")
    print("=" * 60)

    all_curves = []

    for letter in sorted(STRATEGIES.keys()):
        meta = STRATEGIES[letter]
        print(f"\nStrategy {letter}: {meta['name']}")
        result = run_strategy_and_capture_equity(letter)
        if result:
            dates, values = result
            generate_single_chart(letter, dates, values, meta)
            all_curves.append((letter, dates, values, meta))
        else:
            print(f"  No equity data available, skipping individual chart.")

    if all_curves:
        print(f"\nGenerating comparison chart ({len(all_curves)} strategies)...")
        generate_comparison_chart(all_curves)

    print("\nGenerating metrics comparison chart...")
    generate_metrics_chart()

    print(f"\nDone! Charts saved to: {CHART_DIR}")
    print(f"Total: {len(all_curves)} equity curves + 1 comparison + 1 metrics")


if __name__ == '__main__':
    main()
