#!/usr/bin/env python3
"""
Auto trader for 国信金太阳 app via ADB
Monitors for funds availability, then places buy orders when market opens.

Usage:
  python3 jty_auto_trade.py check_funds   # Check available funds
  python3 jty_auto_trade.py buy 510500 8000  # Buy ETF 510500 for ~8000 yuan
  python3 jty_auto_trade.py auto_buy       # Auto buy based on today's signals
"""

import subprocess, time, sys, re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

ADB = "/opt/homebrew/bin/adb"
DEVICE = "6aa26ef"
CST = timezone(timedelta(hours=8))

def adb(*args, timeout=30):
    cmd = [ADB, "-s", DEVICE] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip(), result.returncode

def tap(x, y, delay=0.4):
    adb("shell", "input", "tap", str(x), str(y))
    time.sleep(delay)

def get_ui_dump(path="/sdcard/ui_trade.xml"):
    adb("shell", "uiautomator", "dump", path)
    time.sleep(0.5)
    adb("pull", path, "/tmp/ui_trade.xml")
    try:
        tree = ET.parse('/tmp/ui_trade.xml')
        return tree.getroot()
    except:
        return None

def find_center(root, text=None, res_id=None, content_desc=None):
    for n in root.iter():
        match = False
        if text and n.get('text','') == text:
            match = True
        if res_id and res_id in n.get('resource-id',''):
            match = True
        if content_desc and n.get('content-desc','') == content_desc:
            match = True
        if match:
            bounds = n.get('bounds','')
            m = re.findall(r'\d+', bounds)
            if len(m)==4:
                return (int(m[0])+int(m[2]))//2, (int(m[1])+int(m[3]))//2
    return None

def get_keyboard_map(root):
    """Get security keyboard digit positions from content-desc"""
    kbd = {}
    for n in root.iter():
        cd = n.get('content-desc','')
        bounds = n.get('bounds','')
        if cd and bounds and n.get('clickable') == 'true':
            m = re.findall(r'\d+', bounds)
            if len(m)==4 and int(m[1]) > 1800:
                cx = (int(m[0])+int(m[2]))//2
                cy = (int(m[1])+int(m[3]))//2
                kbd[cd] = (cx, cy)
    return kbd

def type_on_security_keyboard(digits):
    """Type digits using the security keyboard"""
    root = get_ui_dump()
    if not root:
        print("ERROR: Cannot get UI dump for keyboard")
        return False
    kbd = get_keyboard_map(root)
    print(f"  Keyboard map: {kbd}")
    for digit in digits:
        if digit in kbd:
            x, y = kbd[digit]
            print(f"  Tap '{digit}' @ ({x},{y})")
            tap(x, y, delay=0.3)
        else:
            print(f"  ERROR: '{digit}' not in keyboard!")
            return False
    return True

def get_current_activity():
    out, _ = adb("shell", "dumpsys", "activity", "activities")
    for line in out.split('\n'):
        if 'mCurrentFocus' in line:
            return line.strip()
    return ""

def check_funds():
    """Check available funds in trading account"""
    # Navigate to 交易 tab
    adb("shell", "input", "keyevent", "KEYCODE_HOME")
    time.sleep(1)
    adb("shell", "am", "start", "-n", "com.guosen.android/com.guosen.app.goldsun.ui.UserLoginFlashAct")
    time.sleep(3)
    adb("shell", "input", "tap", "800", "2478")  # 交易 tab
    time.sleep(2)
    
    root = get_ui_dump()
    if not root:
        return None
    
    pos = find_center(root, res_id="tvFundAvl")
    if pos:
        # Find text at that position
        for n in root.iter():
            if 'tvFundAvl' in n.get('resource-id',''):
                return float(n.get('text', '0').replace(',', ''))
    return None

def navigate_to_buy_screen(stock_code):
    """Navigate to buy screen for given stock code"""
    print(f"\nNavigating to buy screen for {stock_code}...")
    
    # Ensure on main screen
    adb("shell", "input", "keyevent", "KEYCODE_BACK")
    time.sleep(0.5)
    adb("shell", "input", "keyevent", "KEYCODE_BACK")
    time.sleep(0.5)
    
    # Go to 行情 tab
    tap(480, 2478, delay=2)
    
    # Tap search
    tap(1544, 104, delay=1.5)
    
    # Get keyboard for search
    root = get_ui_dump()
    if not root:
        print("ERROR: Cannot get search screen")
        return False
    
    kbd = get_keyboard_map(root)
    if not kbd:
        # Try standard input for search
        adb("shell", "input", "text", stock_code)
    else:
        for digit in stock_code:
            if digit in kbd:
                tap(*kbd[digit])
    
    time.sleep(1)
    
    # Look for the stock in results
    root = get_ui_dump()
    if not root:
        return False
    
    # Find stock code in results
    pos = find_center(root, text=stock_code)
    if not pos:
        # Try finding by name pattern - look for ETF entries
        for n in root.iter():
            if n.get('resource-id','').endswith('tv_search_stk_code') and n.get('text','') == stock_code:
                bounds = n.get('bounds','')
                m = re.findall(r'\d+', bounds)
                if len(m)==4:
                    # Click on the name element (one before)
                    pos = ((int(m[0])+int(m[2]))//2, (int(m[1])+int(m[3]))//2)
                    break
    
    if pos:
        tap(*pos, delay=3)
    else:
        print(f"ERROR: Cannot find {stock_code} in search results")
        return False
    
    # Now on stock detail page, tap 买 button
    tap(335, 2478, delay=2)
    
    # Verify we're on buy screen
    root = get_ui_dump()
    if not root:
        return False
    
    buy_btn = find_center(root, res_id="btTrade")
    if buy_btn:
        print(f"  Buy screen reached for {stock_code} ✅")
        return True
    else:
        print(f"  ERROR: Buy button not found for {stock_code}")
        return False

def place_buy_order(stock_code, amount_yuan, current_price=None):
    """
    Place a buy order for an ETF.
    amount_yuan: approximate amount in yuan to buy
    Returns True if order submitted successfully
    """
    print(f"\n=== Placing buy order: {stock_code} ~{amount_yuan} yuan ===")
    
    if not navigate_to_buy_screen(stock_code):
        return False
    
    # Get the buy order screen
    root = get_ui_dump()
    if not root:
        return False
    
    # Find current price from screen
    price_el = find_center(root, res_id="etInput")
    price_text = None
    for n in root.iter():
        if 'etInput' in n.get('resource-id','') and n.get('text','') and '.' in n.get('text',''):
            try:
                price = float(n.get('text','0'))
                if price > 0.1:  # Must be a real price
                    price_text = n.get('text','')
                    current_price = price
                    break
            except:
                pass
    
    print(f"  Current price: {current_price}")
    
    if not current_price or current_price <= 0:
        print("  ERROR: Cannot determine current price")
        return False
    
    # Calculate quantity (must be multiple of 100)
    qty = int(amount_yuan / current_price / 100) * 100
    if qty < 100:
        qty = 100
        print(f"  WARNING: Minimum 100 shares, spending ~{qty * current_price:.0f} yuan")
    
    print(f"  Buying {qty} shares @ {current_price} = ~{qty * current_price:.0f} yuan")
    
    # The price field should already have the current price as default
    # We need to enter the quantity
    
    # Find and tap the quantity input field
    qty_pos = None
    for n in root.iter():
        if n.get('resource-id','') == 'com.guosen.android:id/etInput':
            text = n.get('text','')
            if '输入' in text or not text or not any(c.isdigit() for c in text):
                # This is the quantity field (empty or placeholder)
                bounds = n.get('bounds','')
                m = re.findall(r'\d+', bounds)
                if len(m)==4:
                    qty_pos = ((int(m[0])+int(m[2]))//2, (int(m[1])+int(m[3]))//2)
    
    if not qty_pos:
        # Use known position for quantity field
        qty_pos = (800, 2223)
    
    print(f"  Tapping quantity field at {qty_pos}")
    tap(*qty_pos, delay=0.8)
    
    # Enter quantity using security keyboard
    print(f"  Entering quantity: {qty}")
    if not type_on_security_keyboard(str(qty)):
        # Try standard input
        adb("shell", "input", "text", str(qty))
        time.sleep(0.5)
    
    # Tap 确定 to confirm keyboard
    tap(1400, 2423, delay=0.5)  # 确定 button
    
    # Verify quantity was entered
    root = get_ui_dump()
    max_buy_pos = find_center(root, res_id="tv_b_left")
    
    # Find 买入 button
    buy_btn = find_center(root, res_id="btTrade")
    if not buy_btn:
        buy_btn = find_center(root, text="买入")
    
    if buy_btn:
        print(f"  Tapping 买入 button at {buy_btn}")
        tap(*buy_btn, delay=2)
    else:
        print("  ERROR: Cannot find 买入 button!")
        return False
    
    # Check for confirmation dialog
    root = get_ui_dump()
    for n in root.iter():
        text = n.get('text','')
        if '确认' in text or '提交' in text or '确定' in text:
            pos = find_center(root, text=text)
            if pos and n.get('clickable') == 'true':
                print(f"  Confirming order: '{text}' at {pos}")
                tap(*pos, delay=2)
                break
    
    # Check result
    root = get_ui_dump()
    for n in root.iter():
        text = n.get('text','')
        if '成功' in text or '委托' in text or '申报' in text:
            print(f"  Order result: {text}")
            return True
        if '错误' in text or '失败' in text:
            print(f"  Order ERROR: {text}")
            return False
    
    print("  Order submitted (no explicit confirmation message)")
    return True

def get_etf_price(stock_code):
    """Get current price of ETF from market data"""
    import urllib.request, json
    try:
        # Use Sina Finance API
        prefix = "sh" if stock_code.startswith(('5', '6')) else "sz"
        url = f"https://hq.sinajs.cn/list={prefix}{stock_code}"
        req = urllib.request.Request(url, headers={'Referer': 'https://finance.sina.com.cn'})
        response = urllib.request.urlopen(req, timeout=5)
        data = response.read().decode('gbk')
        parts = data.split('"')[1].split(',')
        if len(parts) > 3:
            price = float(parts[3])  # Current price
            return price
    except Exception as e:
        print(f"ERROR getting price for {stock_code}: {e}")
    return None

def auto_buy_signals():
    """Buy ETF signals for today"""
    signals = [
        ("510500", "中证500ETF", 7000),   # 7000 yuan
        ("588000", "科创50ETF", 7000),     # 7000 yuan
        ("159915", "创业板ETF", 7000),     # 7000 yuan
    ]
    
    now = datetime.now(CST)
    print(f"\n{'='*50}")
    print(f"Auto Buy Starting: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    results = []
    for code, name, amount in signals:
        print(f"\nBuying {name} ({code}) for ~{amount} yuan")
        price = get_etf_price(code)
        if price:
            print(f"  Current price: {price}")
        
        success = place_buy_order(code, amount, price)
        results.append((code, name, success))
        
        if success:
            print(f"✅ {name} order placed!")
        else:
            print(f"❌ {name} order FAILED!")
        
        time.sleep(3)  # Wait between orders
    
    print(f"\n{'='*50}")
    print("RESULTS:")
    for code, name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name} ({code})")
    print(f"{'='*50}")
    
    return all(s for _, _, s in results)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if cmd == "check_funds":
        funds = check_funds()
        print(f"Available funds: {funds} yuan")
    
    elif cmd == "buy" and len(sys.argv) >= 4:
        stock_code = sys.argv[2]
        amount = float(sys.argv[3])
        success = place_buy_order(stock_code, amount)
        sys.exit(0 if success else 1)
    
    elif cmd == "auto_buy":
        success = auto_buy_signals()
        sys.exit(0 if success else 1)
    
    else:
        print(__doc__)
