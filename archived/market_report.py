#!/usr/bin/env python3
"""
Hourly market report during A-share trading hours.
Uses Tencent Finance API (qt.gtimg.cn) - free, no auth required.
Fallback: Sina Finance (hq.sinajs.cn).
"""
import urllib.request
import json
import re
import os
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")  # Set via env var, never hardcode
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")  # Set via env var, never hardcode

# Our positions (cost basis)
POSITIONS = {
    "510500": {"name": "中证500ETF", "shares": 1800, "cost": 8.116},
    "159915": {"name": "创业板ETF",  "shares": 4500, "cost": 3.289},
    "588000": {"name": "科创50ETF",  "shares": 10400, "cost": 1.437},
}

# Exchange prefix mapping
EXCHANGE = {
    "510500": "sh", "159915": "sz", "588000": "sh",
    "000001": "sh", "399001": "sz", "399006": "sz",
    "000688": "sh", "000905": "sh",
}

INDICES = [
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("399006", "创业板指"),
    ("000688", "科创50"),
    ("000905", "中证500"),
]


def fetch_tencent(symbols):
    """Fetch quotes from Tencent Finance. symbols like ['sh510500','sz159915']"""
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    req = urllib.request.Request(url, headers={
        "Referer": "https://finance.qq.com",
        "User-Agent": "Mozilla/5.0"
    })
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode("gbk", errors="replace")


def parse_etf_tencent(raw, full_sym):
    """Parse ETF full quote from Tencent. full_sym e.g. 'sh510500'"""
    m = re.search(rf'v_{re.escape(full_sym)}="([^"]*)"', raw)
    if not m:
        return None
    fields = m.group(1).split("~")
    if len(fields) < 6:
        return None
    try:
        current = float(fields[3])
        prev_close = float(fields[4])
        change_pct = (current - prev_close) / prev_close * 100 if prev_close else 0
        return {"price": current, "prev_close": prev_close, "change_pct": change_pct}
    except (ValueError, IndexError):
        return None


def parse_index_tencent(raw, full_sym):
    """Parse index quote from Tencent. full_sym e.g. 's_sh000001'"""
    m = re.search(rf'v_{re.escape(full_sym)}="([^"]*)"', raw)
    if not m:
        return None
    fields = m.group(1).split("~")
    if len(fields) < 6:
        return None
    try:
        name = fields[1]
        current = float(fields[3])
        change_pct = float(fields[5])
        return {"name": name, "price": current, "change_pct": change_pct}
    except (ValueError, IndexError):
        return None


def fmt_pct(pct):
    return f"+{pct:.2f}%" if pct > 0 else f"{pct:.2f}%"


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read()).get("ok", False)


def main():
    now = datetime.now()
    total_min = now.hour * 60 + now.minute
    in_morning = (9 * 60 + 25) <= total_min <= (11 * 60 + 35)
    in_afternoon = (12 * 60 + 55) <= total_min <= (15 * 60 + 5)
    if not (in_morning or in_afternoon):
        print(f"Not in trading hours ({now.strftime('%H:%M')}), skipping.")
        return

    # --- Fetch index quotes ---
    index_syms = [f"s_{EXCHANGE[code]}{code}" for code, _ in INDICES]
    raw_idx = fetch_tencent(index_syms)

    idx_lines = []
    for code, name in INDICES:
        sym = f"s_{EXCHANGE[code]}{code}"
        q = parse_index_tencent(raw_idx, sym)
        if q:
            arrow = "↑" if q["change_pct"] > 0 else ("↓" if q["change_pct"] < 0 else "→")
            idx_lines.append(f"  {arrow} {q['name']}: {q['price']:.2f} ({fmt_pct(q['change_pct'])})")
        else:
            idx_lines.append(f"  ? {name}: 获取失败")

    # --- Fetch ETF position quotes ---
    etf_syms = [f"{EXCHANGE[code]}{code}" for code in POSITIONS]
    raw_etf = fetch_tencent(etf_syms)

    pos_lines = []
    total_cost = total_value = 0.0
    for code, pos in POSITIONS.items():
        sym = f"{EXCHANGE[code]}{code}"
        q = parse_etf_tencent(raw_etf, sym)
        cost_total = pos["shares"] * pos["cost"]
        total_cost += cost_total
        if q:
            val = pos["shares"] * q["price"]
            total_value += val
            pnl = val - cost_total
            pnl_pct = pnl / cost_total * 100
            arrow = "↑" if pnl > 0 else ("↓" if pnl < 0 else "→")
            pos_lines.append(
                f"  {arrow} {pos['name']}({code})\n"
                f"     现价 {q['price']:.3f}  今日{fmt_pct(q['change_pct'])}\n"
                f"     {pos['shares']}股 | 浮盈 {pnl:+.0f}元 ({pnl_pct:+.2f}%)"
            )
        else:
            total_value += cost_total
            pos_lines.append(f"  ? {pos['name']}({code}): 获取失败")

    total_pnl = total_value - total_cost
    total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0
    emoji = "📈" if total_pnl >= 0 else "📉"

    msg = (
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} 行情播报\n\n"
        f"📊 大盘指数:\n" + "\n".join(idx_lines) + "\n\n"
        f"💼 持仓状态:\n" + "\n".join(pos_lines) + "\n\n"
        f"{emoji} 总市值: {total_value:.0f}元\n"
        f"   成本: {total_cost:.0f}元 | 浮盈: {total_pnl:+.0f}元 ({total_pnl_pct:+.2f}%)"
    )

    print(msg)
    ok = send_telegram(msg)
    print(f"Telegram sent: {ok}")


if __name__ == "__main__":
    main()
