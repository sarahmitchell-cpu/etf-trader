#!/usr/bin/env python3
"""
ETF短线/中短线监控系统
策略: 5万本金, 仅债券ETF + 中国宽基ETF + 行业ETF

资金分配:
- 1万: 债券ETF T+0高频 (511090/511010)
- 2万: 宽基ETF中短线 (510300/510500/588000/159915)
- 2万: 行业ETF超跌反弹 (512480/516110/515070/159869/512170/515000)

Author: Sarah Mitchell (VisionClaw)
"""

import urllib.request
import json
import time
import re
from datetime import datetime, date
import os

# ETF分组
BOND_ETFS = {
    "511090": "30年国债ETF",
    "511010": "国债ETF",
}

BROAD_ETFS = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "588000": "科创50ETF",
    "159915": "创业板ETF",
}

SECTOR_ETFS = {
    "512480": "半导体ETF",
    "516110": "新能源ETF",
    "515070": "新能车ETF",
    "159869": "港股通互联网ETF",
    "512170": "医疗ETF",
    "515000": "房地产ETF",
}

ALL_ETFS = {**BOND_ETFS, **BROAD_ETFS, **SECTOR_ETFS}

# 策略阈值
BOND_BUY_DROP = -0.10   # 债券ETF跌0.10%买入
BOND_SELL_RISE = 0.15   # 债券ETF涨0.15%卖出
BOND_STOP_LOSS = -0.25  # 止损

BROAD_BUY_DROP = -1.5   # 宽基大盘跌1.5%建仓
BROAD_TAKE_PROFIT = 2.0 # 涨2%止盈
BROAD_STOP_LOSS = -3.0  # 止损

SECTOR_BUY_DROP = -3.0  # 行业单日跌3%买入
SECTOR_TAKE_PROFIT = 0.8 # 次日涨0.8%卖出  
SECTOR_STOP_LOSS = -4.0 # 止损

DAILY_LOSS_LIMIT = -500  # 单日亏损超500元停止操作 (元)


def _exchange(code: str) -> str:
    """Return exchange prefix for a stock/ETF code."""
    return "sz" if code.startswith("1") else "sh"


def get_quote(symbols: list) -> dict:
    """Fetch ETF/stock quotes using Tencent Finance API (free, no auth).
    Falls back to Sina Finance if Tencent fails.
    Returns dict: {code: {name, current, prev_close, open, change_pct, volume}}
    """
    # --- Tencent Finance ---
    try:
        codes = ",".join(f"{_exchange(s)}{s}" for s in symbols)
        url = f"https://qt.gtimg.cn/q={codes}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.qq.com"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("gbk", errors="replace")

        result = {}
        for sym in symbols:
            full = f"{_exchange(sym)}{sym}"
            m = re.search(rf'v_{re.escape(full)}="([^"]*)"', raw)
            if not m:
                continue
            fields = m.group(1).split("~")
            if len(fields) < 6:
                continue
            try:
                name = fields[1]
                current = float(fields[3])
                prev_close = float(fields[4])
                open_p = float(fields[5])
                volume = int(fields[6]) if fields[6].isdigit() else 0
                change_pct = (current - prev_close) / prev_close * 100 if prev_close else 0
                result[sym] = {
                    "name": name,
                    "current": current,
                    "prev_close": prev_close,
                    "open": open_p,
                    "change_pct": change_pct,
                    "volume": volume,
                }
            except (ValueError, IndexError):
                pass
        if result:
            return result
    except Exception as e:
        print(f"  [腾讯行情失败] {e}, 切换新浪...")

    # --- Sina Finance fallback ---
    try:
        codes = ",".join(f"{_exchange(s)}{s}" for s in symbols)
        url = f"https://hq.sinajs.cn/list={codes}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("gbk", errors="replace")

        result = {}
        for line in raw.strip().split("\n"):
            if "=" not in line or not line.strip():
                continue
            sym_part = line.split("=")[0].strip().split("_")[-1]
            code = sym_part[2:]
            try:
                data = line.split('"')[1]
                fields = data.split(",")
                if len(fields) < 10:
                    continue
                name = fields[0]
                prev_close = float(fields[2]) if fields[2] else 0
                current = float(fields[3]) if fields[3] else 0
                open_p = float(fields[1]) if fields[1] else 0
                volume = int(fields[8]) if fields[8] else 0
                change_pct = (current - prev_close) / prev_close * 100 if prev_close else 0
                result[code] = {
                    "name": name,
                    "current": current,
                    "prev_close": prev_close,
                    "open": open_p,
                    "change_pct": change_pct,
                    "volume": volume,
                }
            except (ValueError, IndexError):
                pass
        return result
    except Exception as e:
        print(f"  [新浪行情失败] {e}")
        return {}


# Alias for backward compatibility
get_quote_simple = get_quote

def analyze_opportunities(quotes: dict) -> dict:
    """分析交易机会"""
    opportunities = {
        'bond': [],      # 债券ETF机会
        'broad': [],     # 宽基ETF机会
        'sector': [],    # 行业ETF机会
        'watchlist': [], # 关注列表 (接近阈值)
    }
    
    for code, info in quotes.items():
        if not info or 'change_pct' not in info:
            continue
            
        pct = info['change_pct']
        name = info.get('name', code)
        current = info.get('current', 0)
        
        if code in BOND_ETFS:
            if BOND_BUY_DROP <= pct < BOND_BUY_DROP * 0.7:
                opportunities['watchlist'].append({
                    'code': code, 'name': name, 'pct': pct,
                    'action': '接近债券买点', 'current': current
                })
            elif pct <= BOND_BUY_DROP:
                opportunities['bond'].append({
                    'code': code, 'name': name, 'pct': pct,
                    'action': '【债券买点】跌幅触发', 'current': current
                })
                
        elif code in BROAD_ETFS:
            if BROAD_BUY_DROP * 0.7 <= pct < BROAD_BUY_DROP:
                opportunities['watchlist'].append({
                    'code': code, 'name': name, 'pct': pct,
                    'action': '接近宽基建仓点', 'current': current
                })
            elif pct <= BROAD_BUY_DROP:
                opportunities['broad'].append({
                    'code': code, 'name': name, 'pct': pct,
                    'action': '【宽基建仓】大跌触发', 'current': current
                })
                
        elif code in SECTOR_ETFS:
            if SECTOR_BUY_DROP * 0.7 <= pct < SECTOR_BUY_DROP:
                opportunities['watchlist'].append({
                    'code': code, 'name': name, 'pct': pct,
                    'action': '接近行业超跌买点', 'current': current
                })
            elif pct <= SECTOR_BUY_DROP:
                opportunities['sector'].append({
                    'code': code, 'name': name, 'pct': pct,
                    'action': '【行业超跌】跌3%触发', 'current': current
                })
    
    return opportunities


def format_morning_briefing(quotes: dict) -> str:
    """生成开盘前操作建议 (9:00)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"📊 ETF早盘操作建议 {now}",
        "=" * 40,
        "",
        "【持仓检查】",
    ]
    
    # 昨日收盘情况
    lines.append("\n【昨日收盘情况】")
    
    # 分组显示
    lines.append("\n▶ 债券ETF (1万仓位, T+0高频)")
    for code, name in BOND_ETFS.items():
        if code in quotes:
            q = quotes[code]
            pct = q.get('change_pct', 0)
            arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"
            lines.append(f"  {code} {name}: {q.get('current', 0):.3f} ({arrow}{abs(pct):.2f}%)")
    
    lines.append("\n▶ 宽基ETF (2万仓位, 中短线)")
    for code, name in BROAD_ETFS.items():
        if code in quotes:
            q = quotes[code]
            pct = q.get('change_pct', 0)
            arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"
            lines.append(f"  {code} {name}: {q.get('current', 0):.3f} ({arrow}{abs(pct):.2f}%)")
    
    lines.append("\n▶ 行业ETF (2万仓位, 超跌反弹)")
    for code, name in SECTOR_ETFS.items():
        if code in quotes:
            q = quotes[code]
            pct = q.get('change_pct', 0)
            arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"
            lines.append(f"  {code} {name}: {q.get('current', 0):.3f} ({arrow}{abs(pct):.2f}%)")
    
    # 操作建议
    lines.append("\n【今日操作策略】")
    
    opportunities = analyze_opportunities(quotes)
    
    has_action = False
    
    if opportunities['bond']:
        has_action = True
        lines.append("\n🔴 债券ETF操作信号:")
        for op in opportunities['bond']:
            lines.append(f"  → {op['action']}: {op['code']} {op['name']} ({op['pct']:.2f}%)")
            lines.append(f"     建议: 用1000-2000元买入, 目标涨{BOND_SELL_RISE}%, 止损{BOND_STOP_LOSS}%")
    
    if opportunities['broad']:
        has_action = True
        lines.append("\n🔴 宽基ETF建仓信号:")
        for op in opportunities['broad']:
            lines.append(f"  → {op['action']}: {op['code']} {op['name']} ({op['pct']:.2f}%)")
            lines.append(f"     建议: 用5000-10000元建仓, 目标涨{BROAD_TAKE_PROFIT}%, 止损{BROAD_STOP_LOSS}%")
    
    if opportunities['sector']:
        has_action = True
        lines.append("\n🔴 行业ETF超跌信号:")
        for op in opportunities['sector']:
            lines.append(f"  → {op['action']}: {op['code']} {op['name']} ({op['pct']:.2f}%)")
            lines.append(f"     建议: 尾盘5000元买入, 明日目标涨{SECTOR_TAKE_PROFIT}%, 止损{SECTOR_STOP_LOSS}%")
    
    if opportunities['watchlist']:
        lines.append("\n👀 关注列表 (接近阈值):")
        for op in opportunities['watchlist']:
            lines.append(f"  ○ {op['code']} {op['name']}: {op['pct']:.2f}% ({op['action']})")
    
    if not has_action and not opportunities['watchlist']:
        lines.append("  ✅ 今日暂无明显操作信号，等待机会")
        lines.append("  → 建议: 观望为主，关注大盘走势")
    
    lines.append("\n【风控提醒】")
    lines.append("  ⚠️ 单日亏损超500元立即停止操作")
    lines.append("  ⚠️ 严格执行止损，不抗单")
    lines.append("  ⚠️ T+1限制: 今日买的非债券ETF明日才能卖")
    
    return "\n".join(lines)


def format_evening_review(quotes: dict, trades_today: list = None) -> str:
    """生成收盘复盘 (15:15)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"📈 ETF收盘复盘 {now}",
        "=" * 40,
        "",
    ]
    
    # 今日涨跌
    lines.append("【今日收盘情况】")
    
    # 找出今日涨跌幅最大的
    sorted_by_change = sorted(
        [(code, info) for code, info in quotes.items() if info and 'change_pct' in info],
        key=lambda x: x[1]['change_pct']
    )
    
    big_drops = [(c, i) for c, i in sorted_by_change if i['change_pct'] <= -2.0]
    big_rises = [(c, i) for c, i in sorted_by_change[::-1] if i['change_pct'] >= 2.0]
    
    if big_drops:
        lines.append("\n📉 今日大跌 (≥2%):")
        for code, info in big_drops[:3]:
            name = ALL_ETFS.get(code, code)
            lines.append(f"  {code} {name}: {info['change_pct']:.2f}%")
            if code in SECTOR_ETFS:
                lines.append(f"  ⚡ 明日关注: {code} 超跌反弹机会")
    
    if big_rises:
        lines.append("\n📈 今日大涨 (≥2%):")
        for code, info in big_rises[:3]:
            name = ALL_ETFS.get(code, code)
            lines.append(f"  {code} {name}: {info['change_pct']:.2f}%")
    
    # 明日策略
    lines.append("\n【明日关注品种】")
    
    # 超跌行业ETF (今日跌幅>2.5%, 明日可能反弹)
    rebound_candidates = [
        (code, info) for code, info in quotes.items()
        if code in SECTOR_ETFS and info and info.get('change_pct', 0) <= -2.5
    ]
    
    if rebound_candidates:
        lines.append("超跌反弹候选:")
        for code, info in sorted(rebound_candidates, key=lambda x: x[1]['change_pct']):
            name = SECTOR_ETFS.get(code, code)
            lines.append(f"  → {code} {name}: 今日{info['change_pct']:.2f}%, 明日关注高开低走/高开持续")
    
    return "\n".join(lines)


def run_monitor():
    """实时监控模式 (交易时间内每分钟刷新)"""
    print("ETF实时监控启动...")
    symbols = list(ALL_ETFS.keys())
    
    while True:
        now = datetime.now()
        # 只在交易时间运行
        if now.weekday() >= 5:  # 周末
            print(f"{now.strftime('%H:%M')} 今天是周末，休市")
            time.sleep(3600)
            continue
        
        hour = now.hour
        if not ((9 <= hour < 12) or (13 <= hour < 15) or hour == 15 and now.minute <= 5):
            if hour < 9:
                wait = (9 - hour) * 3600 - now.minute * 60
                print(f"等待开盘... (剩余约{wait//60}分钟)")
                time.sleep(min(wait, 300))
            else:
                print("休市时间")
                time.sleep(600)
            continue
        
        quotes = get_quote_simple(symbols)
        
        if not quotes:
            print("获取行情失败")
            time.sleep(60)
            continue
        
        # 检查是否有紧急信号
        opportunities = analyze_opportunities(quotes)
        
        urgent = opportunities['bond'] + opportunities['broad'] + opportunities['sector']
        
        if urgent:
            print(f"\n⚡ {now.strftime('%H:%M:%S')} 发现操作信号!")
            for op in urgent:
                print(f"  {op['action']}: {op['code']} {op['name']} {op['pct']:.2f}%")
        
        time.sleep(60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        run_monitor()
    elif len(sys.argv) > 1 and sys.argv[1] == "morning":
        symbols = list(ALL_ETFS.keys())
        quotes = get_quote_simple(symbols)
        print(format_morning_briefing(quotes))
    elif len(sys.argv) > 1 and sys.argv[1] == "evening":
        symbols = list(ALL_ETFS.keys())
        quotes = get_quote_simple(symbols)
        print(format_evening_review(quotes))
    else:
        print("用法:")
        print("  python etf_monitor.py morning  # 生成早盘建议")
        print("  python etf_monitor.py evening  # 生成收盘复盘")
        print("  python etf_monitor.py monitor  # 实时监控模式")
