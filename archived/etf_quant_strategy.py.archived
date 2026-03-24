#!/usr/bin/env python3
"""
ETF双维度动态配置策略 v2.0 (9×9精细版)
Dual-Dimension Dynamic ETF Allocation Strategy

维度1: PE百分位 9档 (L1极度低估 → L9极度高估)
维度2: 趋势强度 9档 (T1极强牛市 → T9极强熊市)

组合成 9×9=81 格策略矩阵，精细决定仓位和品种

Usage:
    python3 etf_quant_strategy.py          # 运行策略，输出今日建议
    python3 etf_quant_strategy.py --test   # 回测模式
    python3 etf_quant_strategy.py --debug  # 显示详细指标
    python3 etf_quant_strategy.py --no-cache  # 强制重新获取数据
"""

import os, sys, json, datetime, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("请先安装akshare: pip3 install akshare pandas scipy")
    sys.exit(1)

from scipy.stats import percentileofscore

# ============================================================
# 常量配置
# ============================================================

STATE_FILE  = '/tmp/etf-trader/quant_state.json'
CACHE_FILE  = '/tmp/etf-trader/quant_cache.json'

# ETF标的池
ETF_POOL = {
    # 宽基（按弹性升序）
    'defensive':   [('510880','红利ETF'),  ('159905','中证100ETF')],
    'value':       [('510300','沪深300ETF'),('510500','中证500ETF')],
    'growth':      [('159915','创业板ETF'), ('512100','中证1000ETF')],
    'aggressive':  [('588000','科创50ETF'), ('159915','创业板ETF')],
    # 行业进攻
    'sector_tech': [('512760','芯片ETF'),   ('159819','人工智能ETF')],
    'sector_mfg':  [('512660','军工ETF'),   ('515030','新能源车ETF')],
    'sector_health':[('512170','医疗ETF'),  ('159992','创新药ETF')],
    # 现金
    'cash':        [('CASH','货币基金/国债逆回购')],
}

# ============================================================
# 维度1: PE百分位 9档
# ============================================================
# L1: 0–11%   极度低估  historic bargain
# L2: 11–22%  深度低估
# L3: 22–33%  低估
# L4: 33–44%  偏低
# L5: 44–56%  合理中性
# L6: 56–67%  偏贵
# L7: 67–78%  高估
# L8: 78–89%  深度高估
# L9: 89–100% 极度高估  bubble territory

PE_LEVELS = [
    (0,   11,  'L1', '极度低估', '🟢🟢'),
    (11,  22,  'L2', '深度低估', '🟢'),
    (22,  33,  'L3', '低估',     '🟢'),
    (33,  44,  'L4', '偏低',     '🟡'),
    (44,  56,  'L5', '合理',     '🟡'),
    (56,  67,  'L6', '偏贵',     '🟠'),
    (67,  78,  'L7', '高估',     '🔴'),
    (78,  89,  'L8', '深度高估', '🔴'),
    (89, 100,  'L9', '极度高估', '🔴🔴'),
]

def get_pe_level(pe_pct):
    """将PE百分位映射到1-9档"""
    for lo, hi, code, label, emoji in PE_LEVELS:
        if lo <= pe_pct < hi or (hi == 100 and pe_pct >= 89):
            return int(code[1]), code, label, emoji
    return 5, 'L5', '合理', '🟡'

# ============================================================
# 维度2: 趋势强度 9档
# ============================================================
# T1: 极强牛市  strong bull
# T2: 强牛市
# T3: 偏牛
# T4: 弱牛/震荡偏强
# T5: 猴市中性
# T6: 震荡偏弱
# T7: 偏熊
# T8: 强熊市
# T9: 极强熊市

TREND_LEVELS = [
    (1, 'T1', '极强牛市', '🚀'),
    (2, 'T2', '强牛市',   '🐂🐂'),
    (3, 'T3', '偏牛',     '🐂'),
    (4, 'T4', '震荡偏强', '📈'),
    (5, 'T5', '猴市中性', '🐒'),
    (6, 'T6', '震荡偏弱', '📉'),
    (7, 'T7', '偏熊',     '🐻'),
    (8, 'T8', '强熊市',   '🐻🐻'),
    (9, 'T9', '极强熊市', '💀'),
]

def compute_trend_score(price_df):
    """
    计算趋势评分 1-9 (1=极强牛, 9=极强熊)

    综合7个子指标，每个贡献0-2分（越大越偏熊），加权求和后映射到1-9
    """
    closes = price_df['close'].values
    n = len(closes)

    # 需要至少120天数据
    if n < 120:
        return 5, {}  # 默认中性

    def safe_ma(window):
        return pd.Series(closes).rolling(window=window, min_periods=window).mean().values

    ma20  = safe_ma(20)
    ma60  = safe_ma(60)
    ma120 = safe_ma(120)

    c  = closes[-1]
    m20  = ma20[-1]
    m60  = ma60[-1]
    m120 = ma120[-1]

    # 斜率（最近5日归一化）
    def slope5(arr):
        v = pd.Series(arr).dropna().values
        if len(v) < 5:
            return 0
        x = np.arange(5)
        return np.polyfit(x, v[-5:], 1)[0] / v[-1]

    sl20  = slope5(ma20)
    sl60  = slope5(ma60)
    sl120 = slope5(ma120)

    # 收益率
    ret5  = c / closes[-5]  - 1 if n >= 5  else 0
    ret20 = c / closes[-20] - 1 if n >= 20 else 0
    ret60 = c / closes[-60] - 1 if n >= 60 else 0

    # RSI
    rsi_val = float(calc_rsi(pd.Series(closes)).iloc[-1])
    if np.isnan(rsi_val):
        rsi_val = 50

    # 子指标评分 (0=极强牛, 14=极强熊)
    scores = {}

    # 1. 价格与MA20关系 (0-2)
    diff20 = (c - m20) / m20
    if   diff20 > 0.05:  scores['c_vs_ma20'] = 0
    elif diff20 > 0:     scores['c_vs_ma20'] = 0.5
    elif diff20 > -0.03: scores['c_vs_ma20'] = 1.0
    elif diff20 > -0.07: scores['c_vs_ma20'] = 1.5
    else:                scores['c_vs_ma20'] = 2.0

    # 2. 价格与MA60关系 (0-2)
    diff60 = (c - m60) / m60
    if   diff60 > 0.10:  scores['c_vs_ma60'] = 0
    elif diff60 > 0.02:  scores['c_vs_ma60'] = 0.5
    elif diff60 > -0.02: scores['c_vs_ma60'] = 1.0
    elif diff60 > -0.08: scores['c_vs_ma60'] = 1.5
    else:                scores['c_vs_ma60'] = 2.0

    # 3. MA均线排列 (0-2)
    if   m20 > m60 > m120:  scores['ma_order'] = 0    # 多头排列
    elif m20 > m60:         scores['ma_order'] = 0.5
    elif m20 > m120:        scores['ma_order'] = 1.0
    elif m60 > m120:        scores['ma_order'] = 1.5
    else:                   scores['ma_order'] = 2.0  # 空头排列

    # 4. MA20斜率 (0-2)
    if   sl20 > 0.003:   scores['sl_ma20'] = 0
    elif sl20 > 0.001:   scores['sl_ma20'] = 0.5
    elif sl20 > -0.001:  scores['sl_ma20'] = 1.0
    elif sl20 > -0.003:  scores['sl_ma20'] = 1.5
    else:                scores['sl_ma20'] = 2.0

    # 5. 20日收益 (0-2)
    if   ret20 > 0.08:   scores['ret20'] = 0
    elif ret20 > 0.03:   scores['ret20'] = 0.5
    elif ret20 > -0.03:  scores['ret20'] = 1.0
    elif ret20 > -0.08:  scores['ret20'] = 1.5
    else:                scores['ret20'] = 2.0

    # 6. 60日收益 (0-2)
    if   ret60 > 0.15:   scores['ret60'] = 0
    elif ret60 > 0.05:   scores['ret60'] = 0.5
    elif ret60 > -0.05:  scores['ret60'] = 1.0
    elif ret60 > -0.15:  scores['ret60'] = 1.5
    else:                scores['ret60'] = 2.0

    # 7. RSI (0-2)
    if   rsi_val > 65:   scores['rsi'] = 0
    elif rsi_val > 55:   scores['rsi'] = 0.5
    elif rsi_val > 45:   scores['rsi'] = 1.0
    elif rsi_val > 35:   scores['rsi'] = 1.5
    else:                scores['rsi'] = 2.0

    total = sum(scores.values())  # 0-14
    # 映射到1-9
    level = int(round(1 + total / 14 * 8))
    level = max(1, min(9, level))

    details = {
        'close': round(float(c), 2),
        'ma20': round(float(m20), 2),
        'ma60': round(float(m60), 2),
        'ma120': round(float(m120), 2),
        'ret5_pct': round(ret5*100, 2),
        'ret20_pct': round(ret20*100, 2),
        'ret60_pct': round(ret60*100, 2),
        'rsi': round(rsi_val, 1),
        'sub_scores': {k: v for k, v in scores.items()},
        'total_score': round(total, 2),
        'raw_level': level,
    }

    return level, details


def get_trend_info(level):
    for lv, code, label, emoji in TREND_LEVELS:
        if lv == level:
            return code, label, emoji
    return 'T5', '猴市', '🐒'

# ============================================================
# 9×9 决策矩阵
# ============================================================
# 输出: (target_position_pct, etf_categories, action_note)

def build_decision(pe_level, trend_level):
    """
    基于PE档(1-9)和趋势档(1-9)计算目标仓位和策略

    仓位公式:
    base_pos = 0.95 - (pe_level-1)*0.055 - (trend_level-1)*0.055
    并根据极端组合额外调整

    ETF品种:
    - 趋势强(T1-3) + 估值低(L1-3): 激进进攻 (科创/创业板/行业ETF)
    - 趋势偏强(T4) + 估值中(L4-6): 宽基均衡
    - 猴市(T5) + 任意: 宽基波段
    - 偏熊(T6-7) + 估值高(L7-9): 防御/红利
    - 强熊(T8-9): 现金为主
    """

    # 仓位计算 (线性插值)
    # pe_level=1,trend_level=1 → 95%
    # pe_level=9,trend_level=9 → 5%
    base_pos = 0.95 - (pe_level - 1) * 0.055 - (trend_level - 1) * 0.055

    # 极端情况调整
    if pe_level >= 8 and trend_level >= 7:
        base_pos = min(base_pos, 0.10)  # 高估+强熊，强制轻仓
    if pe_level <= 2 and trend_level <= 2:
        base_pos = min(base_pos, 0.98)  # 不超过98%

    target_pos = max(0.05, min(0.95, round(base_pos / 0.05) * 0.05))  # 以5%为步长

    # ETF品种选择
    if trend_level >= 8:
        # 强熊/极熊：现金为主
        categories = ['cash']
        action = '强熊保本，持现金/逆回购'
    elif trend_level == 7:
        # 偏熊
        if pe_level <= 3:
            categories = ['value', 'defensive']
            action = '偏熊但估值低，持红利+宽基轻仓定投'
        else:
            categories = ['defensive', 'cash']
            action = '偏熊高估，减仓防御，持红利+现金'
    elif trend_level == 6:
        # 震荡偏弱
        if pe_level <= 3:
            categories = ['value']
            action = '震荡偏弱+低估，持宽基波段，等候反转'
        elif pe_level <= 6:
            categories = ['value', 'defensive']
            action = '震荡偏弱，宽基+红利防守'
        else:
            categories = ['defensive', 'cash']
            action = '震荡偏弱+高估，保守持仓'
    elif trend_level == 5:
        # 猴市中性
        if pe_level <= 2:
            categories = ['growth', 'value']
            action = '猴市+极低估，积极建仓，宽基波段'
        elif pe_level <= 4:
            categories = ['value']
            action = '猴市+低估，宽基ETF波段操作（RSI买卖）'
        elif pe_level <= 6:
            categories = ['value']
            action = '猴市合理估值，宽基小仓波段'
        elif pe_level <= 7:
            categories = ['defensive', 'value']
            action = '猴市偏贵，降仓+波段，红利防守'
        else:
            categories = ['defensive', 'cash']
            action = '猴市高估，大幅降仓'
    elif trend_level == 4:
        # 震荡偏强
        if pe_level <= 3:
            categories = ['growth', 'value']
            action = '偏强+低估，宽基+成长均衡加仓'
        elif pe_level <= 6:
            categories = ['value', 'growth']
            action = '偏强合理，宽基均衡持仓'
        else:
            categories = ['value']
            action = '偏强但偏贵，只持宽基，不追行业'
    elif trend_level == 3:
        # 偏牛
        if pe_level <= 2:
            categories = ['aggressive', 'sector_tech']
            action = '偏牛+极低估，科创/创业板+科技行业进攻'
        elif pe_level <= 4:
            categories = ['growth', 'value']
            action = '偏牛+低估，成长ETF为主加仓'
        elif pe_level <= 6:
            categories = ['value', 'growth']
            action = '偏牛合理，宽基均衡+少量成长'
        else:
            categories = ['value']
            action = '偏牛但高估，只持宽基，注意止盈'
    elif trend_level == 2:
        # 强牛
        if pe_level <= 3:
            categories = ['aggressive', 'sector_tech', 'sector_mfg']
            action = '强牛+低估！满仓进攻，科创+创业板+行业ETF'
        elif pe_level <= 5:
            categories = ['aggressive', 'growth']
            action = '强牛合理估值，高弹性品种持仓'
        elif pe_level <= 7:
            categories = ['growth', 'value']
            action = '强牛高估，持仓但设止损，宽基+成长'
        else:
            categories = ['value']
            action = '强牛泡沫区，只持宽基且设严格止损'
    elif trend_level == 1:
        # 极强牛市
        if pe_level <= 4:
            categories = ['aggressive', 'sector_tech', 'sector_mfg', 'sector_health']
            action = '极强牛+低估！全力进攻，科创+行业ETF轮动'
        elif pe_level <= 6:
            categories = ['aggressive', 'sector_tech']
            action = '极强牛合理估值，高弹性品种满仓'
        elif pe_level <= 7:
            categories = ['growth', 'aggressive']
            action = '极强牛高估，持仓但随时准备减仓'
        else:
            categories = ['value']
            action = '极强牛+泡沫，严格止损，见顶迹象即减仓'
    else:
        categories = ['value']
        action = '默认策略'

    return target_pos, categories, action


def get_etf_list(categories, pe_level, trend_level):
    """根据品种类别和市场状态返回推荐ETF列表"""
    result = []
    seen = set()
    for cat in categories:
        if cat in ETF_POOL:
            for code, name in ETF_POOL[cat]:
                if code not in seen:
                    seen.add(code)
                    result.append((code, name))
    return result[:5]  # 最多5个


# ============================================================
# 数据获取
# ============================================================

def fetch_index_daily(symbol='sh000300', years=2):
    """获取指数日线数据"""
    print(f"  获取指数 {symbol} 行情...")
    df = ak.stock_zh_index_daily(symbol=symbol)
    df = df[['date', 'close']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    return df[df['date'] >= cutoff].reset_index(drop=True)


def fetch_pe_history():
    """
    获取沪深300历史滚动市盈率（20年历史，2005-至今）
    返回 DataFrame(date, pe) 或 None
    """
    print("  获取PE历史数据...")

    # 方法1: stock_index_pe_lg 沪深300 (20年历史，最优！)
    try:
        df = ak.stock_index_pe_lg(symbol='沪深300')
        if df is not None and len(df) > 500:
            df = df[['日期', '滚动市盈率']].rename(columns={'日期': 'date', '滚动市盈率': 'pe'})
            df['date'] = pd.to_datetime(df['date'])
            df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
            df = df.dropna(subset=['pe'])
            df = df.sort_values('date').reset_index(drop=True)
            print(f"  PE数据(沪深300·20年): {len(df)}条，范围{df.iloc[0]['date'].date()}~{df.iloc[-1]['date'].date()}，最新PE={df.iloc[-1]['pe']:.2f}")
            return df
    except Exception as e:
        print(f"  PE方法1失败: {e}")

    # 方法2: stock_market_pe_lg 上证 (约7年历史，备用)
    try:
        df = ak.stock_market_pe_lg(symbol='sh')
        if df is not None and len(df) > 100:
            df = df[['日期', '市盈率']].rename(columns={'日期': 'date', '市盈率': 'pe'})
            df['date'] = pd.to_datetime(df['date'])
            df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
            df = df.dropna(subset=['pe'])
            df = df.sort_values('date').reset_index(drop=True)
            print(f"  PE数据(上证·7年): {len(df)}条，范围{df.iloc[0]['date'].date()}~{df.iloc[-1]['date'].date()}，最新PE={df.iloc[-1]['pe']:.2f}")
            return df
    except Exception as e:
        print(f"  PE方法2失败: {e}")

    # 方法3: 深市
    try:
        df = ak.stock_market_pe_lg(symbol='sz')
        if df is not None and len(df) > 100:
            df = df[['日期', '市盈率']].rename(columns={'日期': 'date', '市盈率': 'pe'})
            df['date'] = pd.to_datetime(df['date'])
            df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
            df = df.dropna(subset=['pe'])
            df = df.sort_values('date').reset_index(drop=True)
            print(f"  PE数据(深市·7年): {len(df)}条")
            return df
    except Exception as e:
        print(f"  PE方法3失败: {e}")

    print("  警告：PE数据获取失败，将使用价格位置估算")
    return None


def estimate_pe_from_price(price_df):
    """
    当PE数据不可用时，用价格历史位置估算估值百分位
    使用价格的10年历史百分位作为估值代理
    """
    closes = price_df['close'].values
    current = closes[-1]
    pct = percentileofscore(closes, current, kind='rank')
    print(f"  使用价格百分位估算估值: {pct:.1f}%")
    return pct


def calc_rsi(prices, period=14):
    """计算RSI"""
    prices = pd.Series(prices)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_wave_signal(etf_code, trend_level, debug=False):
    """
    猴市波段操作信号（T4-T6范围使用）
    """
    if trend_level < 4 or trend_level > 6:
        return None

    today = datetime.date.today()
    try:
        etf_df = ak.fund_etf_hist_em(
            symbol=etf_code,
            period="daily",
            start_date=(today - datetime.timedelta(days=60)).strftime('%Y%m%d'),
            end_date=today.strftime('%Y%m%d'),
            adjust="qfq"
        )
        close_col = [c for c in etf_df.columns if '收盘' in c][0]
        closes = etf_df[close_col].values
        if len(closes) < 15:
            return None

        rsi = calc_rsi(pd.Series(closes))
        current_rsi = float(rsi.iloc[-1])

        if np.isnan(current_rsi):
            return None

        # 波段信号
        if current_rsi < 35:
            signal, emoji = 'strong_buy', '🟢🟢'
        elif current_rsi < 42:
            signal, emoji = 'buy', '🟢'
        elif current_rsi > 72:
            signal, emoji = 'strong_sell', '🔴🔴'
        elif current_rsi > 65:
            signal, emoji = 'sell', '🔴'
        else:
            signal, emoji = 'hold', '🟡'

        return {
            'signal': signal,
            'emoji': emoji,
            'rsi': round(current_rsi, 1),
            'close': round(float(closes[-1]), 4),
        }
    except Exception as e:
        if debug:
            print(f"    波段信号获取失败 {etf_code}: {e}")
        return None


# ============================================================
# 缓存
# ============================================================

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        if cache.get('date') == datetime.date.today().isoformat():
            print("  使用今日本地缓存")
            return cache
    except:
        pass
    return None


def save_cache(data):
    data['date'] = datetime.date.today().isoformat()
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f, ensure_ascii=False, default=str)


# ============================================================
# 主策略运行
# ============================================================

def run_strategy(debug=False, use_cache=True):
    today = datetime.date.today()
    print(f"\n{'='*55}")
    print(f"ETF双维度策略 v2.0 (9×9精细版)  {today}")
    print(f"{'='*55}")

    cache = load_cache() if use_cache else None

    # --- 价格数据 ---
    print("\n[1] 获取市场数据...")
    if cache and 'price_data' in cache:
        price_df = pd.DataFrame(cache['price_data'])
        price_df['date'] = pd.to_datetime(price_df['date'])
        print(f"  缓存价格数据: {len(price_df)}条")
    else:
        try:
            price_df = fetch_index_daily('sh000300', years=3)
        except Exception as e:
            print(f"  价格数据获取失败: {e}")
            return None

    # --- PE数据 ---
    if cache and 'pe_data' in cache and cache['pe_data']:
        pe_df = pd.DataFrame(cache['pe_data'])
        pe_df['date'] = pd.to_datetime(pe_df['date'])
        print(f"  缓存PE数据: {len(pe_df)}条")
    else:
        pe_df = fetch_pe_history()

    if not cache:
        save_cache({
            'price_data': price_df.to_dict('records'),
            'pe_data': pe_df.to_dict('records') if pe_df is not None else None,
        })

    # --- 趋势评分 ---
    print("\n[2] 计算趋势评分 (1-9)...")
    trend_level, trend_details = compute_trend_score(price_df)
    trend_code, trend_label, trend_emoji = get_trend_info(trend_level)
    print(f"  趋势: T{trend_level} {trend_emoji} {trend_label}")
    if debug:
        print(f"  子指标: {trend_details.get('sub_scores', {})}")
        print(f"  总分: {trend_details.get('total_score', 0):.2f}/14")
        for k in ['close','ma20','ma60','ma120','ret20_pct','ret60_pct','rsi']:
            print(f"  {k}: {trend_details.get(k, 'N/A')}")

    # --- PE评分 ---
    print("\n[3] 计算估值评分 (1-9)...")
    current_pe = None
    if pe_df is not None and len(pe_df) > 50:
        # 用10年历史计算百分位
        cutoff_10y = pd.Timestamp.now() - pd.DateOffset(years=10)
        hist = pe_df[pe_df['date'] >= cutoff_10y].dropna(subset=['pe'])
        if len(hist) > 50:
            current_pe = float(hist.iloc[-1]['pe'])
            pe_pct = percentileofscore(hist['pe'].values, current_pe, kind='rank')
        else:
            pe_pct = estimate_pe_from_price(price_df)
    else:
        pe_pct = estimate_pe_from_price(price_df)

    pe_level, pe_code, pe_label, pe_emoji = get_pe_level(pe_pct)
    pe_display = f"PE={current_pe:.1f} " if current_pe else ""
    print(f"  估值: L{pe_level} {pe_emoji} {pe_label} ({pe_display}百分位{pe_pct:.1f}%)")

    # --- 决策 ---
    print("\n[4] 9×9决策矩阵计算...")
    target_pos, categories, action = build_decision(pe_level, trend_level)
    etf_list = get_etf_list(categories, pe_level, trend_level)
    print(f"  目标仓位: {target_pos*100:.0f}%")
    print(f"  策略: {action}")
    print(f"  推荐ETF: {[f'{c} {n}' for c,n in etf_list]}")

    # --- 波段信号（震荡区T4-T6）---
    wave_signals = {}
    if 4 <= trend_level <= 6:
        print("\n[5] 计算波段操作信号...")
        for code, name in [('510300','沪深300'), ('159915','创业板'), ('510500','中证500')]:
            sig = calc_wave_signal(code, trend_level, debug=debug)
            if sig:
                wave_signals[f"{code} {name}"] = sig
                print(f"  {sig['emoji']} {name}: RSI={sig['rsi']}, 信号={sig['signal']}, 价格={sig['close']}")

    # --- 构建结果 ---
    result = {
        'date': str(today),
        'index_close': float(price_df.iloc[-1]['close']),
        'pe_level': pe_level,
        'pe_code': pe_code,
        'pe_label': pe_label,
        'pe_emoji': pe_emoji,
        'pe_percentile': round(pe_pct, 1),
        'current_pe': round(current_pe, 2) if current_pe else None,
        'trend_level': trend_level,
        'trend_code': trend_code,
        'trend_label': trend_label,
        'trend_emoji': trend_emoji,
        'trend_details': trend_details,
        'target_position_pct': target_pos,
        'categories': categories,
        'etf_recommendations': etf_list,
        'action': action,
        'wave_signals': wave_signals,
        'matrix_key': f"L{pe_level} × T{trend_level}",
    }

    return result


def format_report(result, compact=False):
    """格式化策略报告（Telegram版）"""
    if not result:
        return "策略运行失败"

    pos_bar = '█' * int(result['target_position_pct'] * 10) + '░' * (10 - int(result['target_position_pct'] * 10))

    lines = [
        f"📊 ETF双维度策略日报",
        f"📅 {result['date']}",
        f"",
        f"沪深300: {result['index_close']:.2f}",
        f"",
        f"估值: {result['pe_emoji']} L{result['pe_level']} {result['pe_label']}",
    ]
    if result['current_pe']:
        lines.append(f"  PE={result['current_pe']:.1f} 历史{result['pe_percentile']:.1f}%分位")
    else:
        lines.append(f"  价格百分位={result['pe_percentile']:.1f}%")

    lines += [
        f"",
        f"趋势: {result['trend_emoji']} T{result['trend_level']} {result['trend_label']}",
        f"  RSI={result['trend_details'].get('rsi','N/A')} | 20日{result['trend_details'].get('ret20_pct','N/A')}% | 60日{result['trend_details'].get('ret60_pct','N/A')}%",
        f"",
        f"策略矩阵: L{result['pe_level']} × T{result['trend_level']}",
        f"目标仓位: {result['target_position_pct']*100:.0f}% [{pos_bar}]",
        f"",
        f"策略: {result['action']}",
        f"",
        f"推荐ETF:",
    ]

    for code, name in result['etf_recommendations']:
        lines.append(f"  • {code} {name}")

    if result['wave_signals']:
        lines.append("")
        lines.append("波段信号:")
        signal_map = {
            'strong_buy':  '🟢🟢 强买入',
            'buy':         '🟢 买入',
            'hold':        '🟡 持有',
            'sell':        '🔴 卖出',
            'strong_sell': '🔴🔴 强卖出',
        }
        for etf, sig in result['wave_signals'].items():
            stext = signal_map.get(sig['signal'], sig['signal'])
            lines.append(f"  {etf}: {stext} (RSI={sig['rsi']})")

    lines += [
        "",
        "操作说明:",
    ]

    tl = result['trend_level']
    if tl <= 3:
        lines.append("  牛市: 持仓为主，强势品种跟进，不轻易止盈")
    elif tl <= 6:
        lines.append("  震荡: RSI<40买入，RSI>68或涨6%卖出")
        lines.append("  单次操作≤总资金10%，每周最多2次")
    else:
        lines.append("  熊市: 严控仓位，定投为主，等待反转信号")

    return "\n".join(lines)


# ============================================================
# 回测模块
# ============================================================

def simple_backtest(years_back=3, debug=False):
    """历史回测"""
    print(f"\n{'='*55}")
    print(f"回测模式 (过去{years_back}年)")
    print(f"{'='*55}")

    print("获取数据...")
    price_df = fetch_index_daily('sh000300', years=years_back+1)
    pe_df = fetch_pe_history()

    dates = price_df['date'].tolist()
    closes = price_df['close'].values

    portfolio_value = 100.0
    peak_value = 100.0
    current_pos = 0.5
    nav_history = []
    rebalance_log = []
    last_month = None

    min_history = 130

    for i, date in enumerate(dates):
        if i < min_history:
            nav_history.append({'date': date, 'nav': portfolio_value, 'pos': current_pos})
            continue

        # 月度再平衡
        month = date.month
        if month != last_month:
            last_month = month
            sub_price = price_df.iloc[:i+1].copy()

            # 趋势
            t_level, _ = compute_trend_score(sub_price)

            # PE
            if pe_df is not None:
                sub_pe = pe_df[pe_df['date'] <= date]
                if len(sub_pe) > 50:
                    cutoff_10y = date - pd.DateOffset(years=10)
                    hist = sub_pe[sub_pe['date'] >= cutoff_10y].dropna(subset=['pe'])
                    if len(hist) > 30:
                        cur_pe = float(hist.iloc[-1]['pe'])
                        pe_pct_val = percentileofscore(hist['pe'].values, cur_pe, kind='rank')
                    else:
                        pe_pct_val = 50
                else:
                    pe_pct_val = 50
            else:
                pe_pct_val = percentileofscore(closes[:i+1], closes[i], kind='rank')

            p_level, _, _, _ = get_pe_level(pe_pct_val)
            target_pos, cats, action = build_decision(p_level, t_level)
            current_pos = target_pos

            if debug:
                rebalance_log.append({
                    'date': str(date.date()),
                    'pe_level': p_level,
                    'trend_level': t_level,
                    'pos': target_pos,
                    'action': action[:30],
                })

        # 当日净值
        if i > 0:
            daily_ret = closes[i] / closes[i-1] - 1
            portfolio_value *= (1 + daily_ret * current_pos)
            peak_value = max(peak_value, portfolio_value)

        nav_history.append({'date': date, 'nav': portfolio_value, 'pos': current_pos})

    nav_df = pd.DataFrame(nav_history).set_index('date')

    start = nav_df.index[min_history]
    nav_df = nav_df.iloc[min_history:]

    total_ret = nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0] - 1
    years = (nav_df.index[-1] - nav_df.index[0]).days / 365
    annual_ret = (1 + total_ret) ** (1/max(years, 0.1)) - 1

    peak_series = nav_df['nav'].cummax()
    dd_series = (nav_df['nav'] - peak_series) / peak_series
    max_dd = dd_series.min()

    # 对比：买入持有
    bnh_prices = price_df[price_df['date'] >= start]['close'].values
    bnh_ret = bnh_prices[-1] / bnh_prices[0] - 1
    bnh_annual = (1 + bnh_ret) ** (1/max(years, 0.1)) - 1

    # 夏普（简化）
    nav_series = nav_df['nav']
    daily_nav_rets = nav_series.pct_change().dropna()
    sharpe = (daily_nav_rets.mean() * 252) / (daily_nav_rets.std() * np.sqrt(252)) if daily_nav_rets.std() > 0 else 0

    print(f"\n回测区间: {start.date()} → {nav_df.index[-1].date()}")
    print(f"\n--- 策略表现 ---")
    print(f"总收益:    {total_ret*100:.1f}%")
    print(f"年化收益:  {annual_ret*100:.1f}%")
    print(f"最大回撤:  {max_dd*100:.1f}%")
    print(f"夏普比率:  {sharpe:.2f}")
    print(f"\n--- 买入持有(沪深300) ---")
    print(f"总收益:    {bnh_ret*100:.1f}%")
    print(f"年化收益:  {bnh_annual*100:.1f}%")
    print(f"\n--- 胜出 ---")
    print(f"超额年化:  {(annual_ret - bnh_annual)*100:.1f}%")

    if debug and rebalance_log:
        print(f"\n--- 最近10次调仓 ---")
        for log in rebalance_log[-10:]:
            print(f"  {log['date']} PE=L{log['pe_level']} T={log['trend_level']} → 仓位{log['pos']*100:.0f}% | {log['action']}")

    return {'total_ret': total_ret, 'annual_ret': annual_ret, 'max_drawdown': max_dd, 'sharpe': sharpe}


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    debug     = '--debug'    in args
    test_mode = '--test'     in args
    no_cache  = '--no-cache' in args

    if test_mode:
        years = 3
        for a in args:
            if a.startswith('--years='):
                years = int(a.split('=')[1])
        simple_backtest(years_back=years, debug=debug)
    else:
        result = run_strategy(debug=debug, use_cache=not no_cache)
        if result:
            report = format_report(result)
            print(f"\n{'='*55}")
            print("Telegram报告预览")
            print(f"{'='*55}")
            print(report)

            out = f'/tmp/etf-trader/quant_report_{datetime.date.today()}.json'
            with open(out, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n结果已保存: {out}")
        else:
            print("策略运行失败")
            sys.exit(1)
