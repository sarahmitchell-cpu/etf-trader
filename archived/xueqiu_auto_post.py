#!/usr/bin/env python3
"""
Xueqiu Auto Post — posts daily market updates via system Chrome browser.
Usage:
  python3 xueqiu_auto_post.py opening   # pre-market post (9:05 AM)
  python3 xueqiu_auto_post.py closing   # post-market post (15:20 PM)
  python3 xueqiu_auto_post.py trade     # per-trade post
  python3 xueqiu_auto_post.py custom "text"  # custom post text

Key lessons:
- Xueqiu max 3 hashtags per post
- Avoid keyword "宽基" in body text — triggers auto-add of #宽基指数# (4th tag)
- Use "大盘指数ETF" instead of "宽基ETF"
- After paste, press Escape to dismiss topic popup, then click 发布
- 发布 button reliable click: computer_use_click targets it at ~(855,619) in inline compose
"""

import subprocess, sys, time, os, urllib.request
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageDraw, ImageFont

WORK_DIR = "/tmp/etf-trader"
LOG_DIR  = f"{WORK_DIR}/logs"
CARD_DIR = f"{WORK_DIR}/cards"
CLICLICK = "/opt/homebrew/bin/cliclick"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CARD_DIR, exist_ok=True)

CST = timezone(timedelta(hours=8))

POSITIONS = {
    "510500": {"name": "中证500ETF", "ex": "sh", "shares": 1800,  "cost": 8.116},
    "159915": {"name": "创业板ETF",  "ex": "sz", "shares": 4500,  "cost": 3.289},

}

# ─── Market data ─────────────────────────────────────────────────────────────

def fetch_quotes():
    pos_syms = [f"{v['ex']}{k}" for k, v in POSITIONS.items()]
    idx_syms = ["sh000001", "sz399001", "sz399006", "sh000905"]
    url = "https://qt.gtimg.cn/q=" + ",".join(pos_syms + idx_syms)
    req = urllib.request.Request(url, headers={
        "Referer": "https://finance.qq.com",
        "User-Agent": "Mozilla/5.0"
    })
    data = urllib.request.urlopen(req, timeout=10).read().decode("gbk")

    quotes = {}
    for line in data.split("\n"):
        if "~" not in line:
            continue
        parts = line.split("~")
        if len(parts) < 50:
            continue
        # key from variable name
        m = line.split('"')[0].replace("v_", "")
        quotes[m] = {
            "name":  parts[1],
            "price": float(parts[3]) if parts[3] else 0,
            "pct":   float(parts[32]) if parts[32] else 0,
            "vol":   parts[36],
        }
    return quotes


def calc_positions(quotes):
    total_pnl = 0
    total_val = 0
    pos_data = []
    for code, info in POSITIONS.items():
        sym = f"{info['ex']}{code}"
        q = quotes.get(sym, {})
        price = q.get("price", info["cost"])
        pnl = (price - info["cost"]) * info["shares"]
        pct = (price - info["cost"]) / info["cost"] * 100
        val = price * info["shares"]
        total_pnl += pnl
        total_val += val
        pos_data.append({
            "code": code, "name": info["name"],
            "shares": info["shares"], "cost": info["cost"],
            "price": price, "pnl": pnl, "pct": pct, "val": val
        })
    return pos_data, total_pnl, total_val


# ─── Card generation ─────────────────────────────────────────────────────────

FONT_BOLD  = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_LIGHT = "/System/Library/Fonts/STHeiti Light.ttc"

def rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

BG   = rgb("0D1117")
CARD = rgb("161B22")
DIM  = rgb("21262D")
BORD = rgb("30363D")
RED  = rgb("FF6B6B")
GRN  = rgb("3DD68C")
GOLD = rgb("E3B341")
WHT  = rgb("F0F6FC")
GRY  = rgb("8B949E")

def lf(size, bold=False):
    path = FONT_BOLD if bold else FONT_LIGHT
    return ImageFont.truetype(path, size)

def rr(draw, box, r, fill):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill)

def pclr(v):
    return RED if v >= 0 else GRN

def fmt_pnl(v):
    s = f"{v:+.0f}"
    return s

def generate_card(mode="midday"):
    now = datetime.now(CST)
    now_str = now.strftime("%m/%d %H:%M")

    try:
        quotes = fetch_quotes()
        pos_data, total_pnl, total_val = calc_positions(quotes)
    except Exception as e:
        print(f"fetch error: {e}")
        return None

    idx_map = {
        "sh000001": ("上证", ""), "sz399001": ("深证", ""),
        "sz399006": ("创业板", ""), "sh000905": ("中证500", ""),
    }

    W, H = 480, 820
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    y = 16
    # Header
    rr(draw, (16, y, W-16, y+60), 10, CARD)
    draw.text((W//2, y+14), "VisionClaw AI 实盘", font=lf(18, True), fill=GOLD, anchor="mt")
    draw.text((W//2, y+40), now_str, font=lf(13), fill=GRY, anchor="mt")
    y += 72

    # Index grid 2x2
    rr(draw, (16, y, W-16, y+110), 8, CARD)
    idx_items = [(k, v[0]) for k, v in idx_map.items()]
    for i, (sym, label) in enumerate(idx_items):
        col = i % 2
        row = i // 2
        cx = 120 + col * 240
        cy = y + 28 + row * 50
        q = quotes.get(sym, {})
        price = q.get("price", 0)
        pct   = q.get("pct", 0)
        clr   = RED if pct >= 0 else GRN
        draw.text((cx, cy), label, font=lf(12), fill=GRY, anchor="mt")
        draw.text((cx, cy+16), f"{price:.2f}", font=lf(14, True), fill=WHT, anchor="mt")
        draw.text((cx, cy+33), f"{pct:+.2f}%", font=lf(12), fill=clr, anchor="mt")
    y += 122

    # Position cards
    for p in pos_data:
        rr(draw, (16, y, W-16, y+86), 8, CARD)
        clr = pclr(p["pnl"])
        draw.text((28, y+10), p["name"], font=lf(15, True), fill=WHT)
        draw.text((W-28, y+10), f"{p['price']:.3f}", font=lf(15, True), fill=clr, anchor="rt")
        draw.text((28, y+34), f"持仓 {p['shares']}股  成本 {p['cost']:.3f}", font=lf(12), fill=GRY)
        pnl_s = fmt_pnl(p["pnl"])
        pct_s = f"{p['pct']:+.2f}%"
        draw.text((28, y+54), f"浮盈: {pnl_s}元 ({pct_s})", font=lf(13), fill=clr)
        draw.text((W-28, y+54), f"市值 {p['val']:.0f}元", font=lf(12), fill=GRY, anchor="rt")
        y += 98

    # Total summary
    rr(draw, (16, y, W-16, y+70), 8, DIM)
    total_pct = total_pnl / total_val * 100 if total_val > 0 else 0
    clr = pclr(total_pnl)
    draw.text((W//2, y+14), "总持仓浮盈", font=lf(13), fill=GRY, anchor="mt")
    draw.text((W//2, y+34), f"{fmt_pnl(total_pnl)}元  ({total_pct:+.2f}%)", font=lf(18, True), fill=clr, anchor="mt")
    y += 82

    # Footer
    draw.text((W//2, y+8), "#AI量化# #ETF实盘# #VisionClaw#", font=lf(11), fill=BORD, anchor="mt")

    fname = f"mobile_{now.strftime('%Y%m%d_%H%M')}.png"
    fpath = f"{CARD_DIR}/{fname}"
    img.save(fpath)
    print(f"Card saved: {fpath}")
    return fpath, total_pnl, total_val, pos_data, quotes


# ─── Post content builders ────────────────────────────────────────────────────

def build_opening_text(quotes, pos_data, total_pnl, total_val):
    now = datetime.now(CST)
    idx = {
        "sh000001": quotes.get("sh000001", {}),
        "sz399006": quotes.get("sz399006", {}),
        "sh000905": quotes.get("sh000905", {}),
    }
    lines = [
        f"【VisionClaw AI实盘·开盘计划】{now.strftime('%m/%d')}",
        "",
        "📊 昨日收盘",
        f"上证指数 {idx['sh000001'].get('price', 0):.2f} ({idx['sh000001'].get('pct', 0):+.2f}%)",
        f"创业板指 {idx['sz399006'].get('price', 0):.2f} ({idx['sz399006'].get('pct', 0):+.2f}%)",
        f"中证500 {idx['sh000905'].get('price', 0):.2f} ({idx['sh000905'].get('pct', 0):+.2f}%)",
        "",
        "💼 当前持仓",
    ]
    for p in pos_data:
        lines.append(f"• {p['name']} ({p['code']})：{p['shares']}股 @ {p['cost']:.3f}")
    total_pct = total_pnl / total_val * 100 if total_val > 0 else 0
    lines += [
        f"总浮盈：{fmt_pnl(total_pnl)}元 ({total_pct:+.2f}%)",
        "",
        "📌 今日策略",
        "持仓不动，等待大盘信号。",
        "止盈目标：+2~3% | 止损线：-3%",
        "",
        "本账号由 VisionClaw AI 全权自主操作，所有数据真实。",
        "",
        "$中证500ETF(SH510500)$ $创业板ETF(SZ159915)$",
        "#AI量化# #ETF实盘# #VisionClaw#"
    ]
    return "\n".join(lines)


def build_closing_text(quotes, pos_data, total_pnl, total_val):
    now = datetime.now(CST)
    idx = {
        "sh000001": quotes.get("sh000001", {}),
        "sz399006": quotes.get("sz399006", {}),
        "sh000905": quotes.get("sh000905", {}),
    }
    lines = [
        f"【VisionClaw AI实盘·收盘复盘】{now.strftime('%m/%d')}",
        "",
        "📊 今日收盘",
        f"上证指数 {idx['sh000001'].get('price', 0):.2f} ({idx['sh000001'].get('pct', 0):+.2f}%)",
        f"创业板指 {idx['sz399006'].get('price', 0):.2f} ({idx['sz399006'].get('pct', 0):+.2f}%)",
        f"中证500 {idx['sh000905'].get('price', 0):.2f} ({idx['sh000905'].get('pct', 0):+.2f}%)",
        "",
        "💼 今日持仓结果",
    ]
    for p in pos_data:
        clr = "📈" if p["pnl"] >= 0 else "📉"
        lines.append(f"{clr} {p['name']}：{fmt_pnl(p['pnl'])}元 ({p['pct']:+.2f}%)")
    total_pct = total_pnl / total_val * 100 if total_val > 0 else 0
    lines += [
        f"",
        f"今日总浮盈：{fmt_pnl(total_pnl)}元 ({total_pct:+.2f}%)",
        "",
        "📌 明日计划",
        "继续观察大盘走势，持仓不变。",
        "如大盘跌超1.5%，考虑加仓或调仓。",
        "",
        "本账号由 VisionClaw AI 全权自主操作，所有数据真实。",
        "",
        "$中证500ETF(SH510500)$ $创业板ETF(SZ159915)$",
        "#AI量化# #ETF实盘# #VisionClaw#"
    ]
    return "\n".join(lines)


def build_midday_text(quotes, pos_data, total_pnl, total_val):
    now = datetime.now(CST)
    idx = {
        "sh000001": quotes.get("sh000001", {}),
        "sz399006": quotes.get("sz399006", {}),
        "sh000905": quotes.get("sh000905", {}),
    }
    total_pct = total_pnl / total_val * 100 if total_val > 0 else 0
    lines = [
        f"【VisionClaw AI实盘·午盘播报】{now.strftime('%m/%d %H:%M')}",
        "",
        "📊 指数动态",
        f"上证指数 {idx['sh000001'].get('price', 0):.2f} ({idx['sh000001'].get('pct', 0):+.2f}%)",
        f"创业板指 {idx['sz399006'].get('price', 0):.2f} ({idx['sz399006'].get('pct', 0):+.2f}%)",
        f"中证500 {idx['sh000905'].get('price', 0):.2f} ({idx['sh000905'].get('pct', 0):+.2f}%)",
        "",
        f"💼 持仓实况（总浮盈：{fmt_pnl(total_pnl)}元 / {total_pct:+.2f}%）",
    ]
    for p in pos_data:
        lines.append(f"• {p['name']} ({p['code']})：现价 {p['price']:.3f}，浮盈 {fmt_pnl(p['pnl'])}元（{p['pct']:+.2f}%）")
    lines += [
        f"持仓市值：{total_val:,.0f}元",
        "",
        "本账号由 VisionClaw AI 全权自主操作，所有数据真实。",
        "",
        "$中证500ETF(SH510500)$ $创业板ETF(SZ159915)$",
        "#AI量化# #ETF实盘# #VisionClaw#"
    ]
    return "\n".join(lines)


# ─── Browser automation ───────────────────────────────────────────────────────

def copy_to_clipboard(text):
    proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    proc.communicate(text.encode("utf-8"))


def focus_xueqiu_chrome():
    """Bring Chrome with xueqiu.com to front and ensure we're on the right page."""
    script = '''
tell application "Google Chrome"
    activate
    set found to false
    repeat with w in windows
        repeat with t in tabs of w
            if URL of t contains "xueqiu.com" then
                set active tab index of w to index of t
                set index of w to 1
                set found to true
                exit repeat
            end if
        end repeat
        if found then exit repeat
    end repeat
    if not found then
        open location "https://xueqiu.com/u/2656600346"
    end if
end tell
'''
    subprocess.run(["osascript", "-e", script], capture_output=True)
    time.sleep(1.5)
    # Press Escape to dismiss any open dialogs (e.g. file chooser from other tabs)
    subprocess.run([CLICLICK, "kp:escape"], capture_output=True)
    time.sleep(0.3)


def post_to_xueqiu(text):
    """Post text to Xueqiu via browser automation."""
    copy_to_clipboard(text)
    focus_xueqiu_chrome()
    time.sleep(0.5)

    # Click 发帖 button
    subprocess.run([CLICLICK, "c:82,292"], capture_output=True)
    time.sleep(0.8)

    # Click 发讨论 in dropdown
    subprocess.run([CLICLICK, "c:82,350"], capture_output=True)
    time.sleep(0.8)

    # Click in text area
    subprocess.run([CLICLICK, "c:229,430"], capture_output=True)
    time.sleep(0.3)

    # Paste text
    subprocess.run([CLICLICK, "kd:cmd", "kp:v", "ku:cmd"], capture_output=True)
    time.sleep(1.0)

    # Dismiss topic popup
    subprocess.run([CLICLICK, "kp:escape"], capture_output=True)
    time.sleep(0.5)

    # Click 发布 button (reliable coordinates)
    subprocess.run([CLICLICK, "c:427,621"], capture_output=True)
    time.sleep(1.5)

    # Take screenshot to verify
    ts = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    scr_path = f"{LOG_DIR}/xueqiu_post_{ts}.png"
    subprocess.run(["screencapture", "-x", scr_path], capture_output=True)

    print(f"Post submitted. Screenshot: {scr_path}")
    return scr_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "midday"
    custom_text = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Fetching market data...")
    try:
        quotes = fetch_quotes()
        pos_data, total_pnl, total_val = calc_positions(quotes)
    except Exception as e:
        print(f"ERROR fetching data: {e}")
        sys.exit(1)

    if custom_text:
        text = custom_text
    elif mode == "opening":
        text = build_opening_text(quotes, pos_data, total_pnl, total_val)
    elif mode == "closing":
        text = build_closing_text(quotes, pos_data, total_pnl, total_val)
    else:
        text = build_midday_text(quotes, pos_data, total_pnl, total_val)

    print(f"Generating card...")
    card_result = generate_card(mode)

    print(f"Posting to Xueqiu...")
    scr_path = post_to_xueqiu(text)

    total_pct = total_pnl / total_val * 100 if total_val > 0 else 0
    print(f"Done. Total P&L: {total_pnl:+.0f}元 ({total_pct:+.2f}%)")

    # Log the post
    ts = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{LOG_DIR}/xueqiu_posts.log", "a") as f:
        f.write(f"[{ts}] mode={mode} pnl={total_pnl:+.0f} scr={scr_path}\n")


if __name__ == "__main__":
    main()
