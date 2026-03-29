# ETF & 个股量化策略体系

> 10套独立策略 · A股/港股/美股/债券全覆盖 · 无前瞻偏差 · 自动化信号推送

一套完整的量化交易策略体系，覆盖债券、A股ETF、港股ETF、美股QDII、指数择时和个股因子选股。

---

## 策略总览

| 策略 | 名称 | 标的 | 核心逻辑 | CAGR | MDD | Sharpe | 调仓频率 |
|------|------|------|---------|------|-----|--------|---------|
| **A** | 债券分散持有 | 4只债券ETF | 等权分散+季度再平衡 | 4.1% | -2.1% | 1.045 | 季度 |
| **B** | 因子ETF轮动 | 4只A股Smart Beta ETF | 动量排名+Regime+Vol缩放 | 15.6% | -18.3% | 1.157 | 每周 |
| **C** | 跨境ETF纯动量 | 3只恒生系ETF | Top1纯动量，100%持仓 | 17.0% | -23.3% | 0.70 | 每周 |
| **D** | CSI300低波动 | 沪深300成分股 | 20周波动率最低TOP10 | 13.9% | -8.5% | 0.977 | 双周 |
| **E** | 逆向价值+质量 | 28只A股龙头 | 10w反转+低波质量过滤 | 22.0% | -17.6% | 0.937 | 月度 |
| **E v2** | 增强价值投资 | 28只A股龙头 | PE/PB基本面+逆向+质量 | — | — | — | 月度 |
| **F** | 美股QDII动量 | 纳指/标普QDII ETF | 双MA趋势+回撤锁定 | 13.0% | -18.9% | 0.809 | 每周 |
| **G** | CSI500低波价值 | 中证500成分股 | 50%低波+50%低PB | 13.5% | -11.9% | 0.794 | 双周 |
| **H** | 指数超跌买入 | 科创50指数 | 超跌/追涨信号+定期持有 | 13.9% | -6.6% | 1.04 | 事件驱动 |
| **L** | 300成长MA60趋势 | 沪深300成长指数 | 纯MA60趋势跟踪 | 18.9% | -31.7% | 1.050 | 日频信号 |

> 另有**个股因子策略**（CSI800 低换手+12M动量 Top20，CAGR=21.4%），见 `research/stock_strategy_live.py`

### 策略互补关系

```
               +-- 策略A (债券分散持有)          <-- 防守底仓，极低波动
               |
               +-- 策略B (因子ETF轮动)          <-- A股SmartBeta，Regime自适应
               |
               +-- 策略C (跨境ETF动量)           <-- 港股独立配置
               |
               +-- 策略D (CSI300低波) --+
               |                        +--  成分股策略，数据无后视偏差
资金分配 ------+-- 策略G (CSI500低波) --+
               |
               +-- 策略E (逆向价值)              <-- 28只龙头中选低估股 (有后视偏差风险)
               |
               +-- 策略F (美股QDII)              <-- 美股配置，趋势跟踪
               |
               +-- 策略H (指数超跌买入)          <-- 科创50事件驱动，低仓位占用
               |
               +-- 策略L (300成长MA60)           <-- 大盘成长趋势择时，极简
```

---

## 快速开始

### 环境要求

```bash
Python 3.8+
pip3 install -r requirements.txt
```

- `baostock`: 策略D和G需要（获取沪深300/中证500历史成分股数据）
- `akshare`: 策略E v2和个股因子策略需要（获取PE/PB基本面数据）
- 其他策略仅需 `pandas numpy requests`

### 运行信号

```bash
git clone https://github.com/sarahmitchell-cpu/etf-trader.git
cd etf-trader

# 策略A - 债券分散持有
python3 strategy_a_weekly_signal.py

# 策略B - 因子ETF轮动
python3 strategy_b_weekly_signal.py

# 策略C - 跨境ETF纯动量
python3 strategy_c_weekly_signal.py

# 策略D - CSI300低波动（双周）
python3 strategy_d_weekly_signal.py

# 策略E - 逆向价值+质量（月度）
python3 strategy_e_weekly_signal.py

# 策略E v2 - 增强价值投资（月度）
python3 strategy_e_v2_weekly_signal.py

# 策略F - 美股QDII动量
python3 strategy_f_weekly_signal.py

# 策略G - CSI500低波价值（双周）
python3 strategy_g_weekly_signal.py

# 策略H - 科创50超跌买入（事件驱动）
python3 strategy_h_weekly_signal.py

# 策略L - 300成长MA60趋势择时
python3 strategy_l_weekly_signal.py

# 个股因子策略 - CSI800 低换手+动量 Top20
python3 research/stock_strategy_live.py
```

### 运行回测

```bash
# 各策略均支持 --backtest 参数
python3 strategy_a_weekly_signal.py --backtest
python3 strategy_b_weekly_signal.py --backtest
python3 strategy_d_weekly_signal.py --backtest
python3 strategy_g_weekly_signal.py --backtest
python3 strategy_h_weekly_signal.py --backtest
python3 strategy_l_weekly_signal.py --backtest
# ...
```

### JSON输出

```bash
# 各策略均支持 --json 输出结构化信号
python3 strategy_a_weekly_signal.py --json
python3 strategy_b_weekly_signal.py --json
python3 strategy_h_weekly_signal.py --json
# ...
```

---

## 策略详细说明

每个策略都有独立的详细文档，位于 `docs/` 目录。以下是概要说明。

---

### 策略A: 债券分散持有

> 详细文档: [docs/strategy_a.md](docs/strategy_a.md)

**逻辑**: 在4只债券ETF中等权分散持有（各25%），每季度再平衡。纯被动策略，目标获取债券市场平均收益。

**ETF池**:

| 指数 | ETF | 代码 | 定位 |
|------|-----|------|------|
| 十年国债 (H11009) | 国泰十年国债ETF | 511260 | 长久期利率债 |
| 城投债 (H11015) | 海富通城投债ETF | 511220 | 中久期信用债 |
| 信用债 (H11073) | 平安公司债ETF | 511030 | 中短久期信用债 |
| 短债 (H11006) | 海富通短融ETF | 511360 | 短久期现金替代 |

**参数**: `weights=[25%, 25%, 25%, 25%], rebalance=quarterly, txn_cost=5bp`

**回测 (15年)**: CAGR 4.1%, MDD -2.1%, Sharpe 1.045, Calmar 1.97

**偏差处理**: 回测使用中证债券指数（含票息收入），非ETF价格。无选择偏差（标准债券指数）。

---

### 策略B: 4因子ETF轮动

> 详细文档: [docs/strategy_b.md](docs/strategy_b.md)

**逻辑**: 4只SmartBeta ETF按4周动量排名 -> 50/30/20/0集中配置 -> Regime判断（牛/熊/过渡）调整仓位 -> 波动率缩放动态调节。

**ETF池**: 红利低波(515080)、自由现金流(159201)、国信价值(512040)、创业成长(159967)

**参数**: `momentum_lookback=4, vol_target=15%, regime: MA13w/MA40w, vol_scale_cap=1.3`

**回测 (11年)**: CAGR 15.6%, MDD -18.3%, Sharpe 1.157

**杠杆风险提示**: vol_scale_cap=1.3意味着低波动市场中仓位可达130%，需要融资。如不使用杠杆，建议将vol_scale_cap设为1.0。

---

### 策略C: 跨境ETF纯动量

> 详细文档: [docs/strategy_c.md](docs/strategy_c.md)

**逻辑**: 3只恒生系ETF取4周动量最强的1只，100%持仓。极致简洁，无趋势过滤、无Regime择时。

**ETF池**: 恒生高股息(159726)、恒生科技(513180)、恒生医疗(513060)

**参数**: `momentum_lookback=4, top_n=1, txn_cost=8bp`

**回测 (4.2年)**: CAGR 17.0%, MDD -23.3%, Sharpe 0.70

**局限**: 仅4.2年数据，2025港股大涨贡献大部分收益，统计意义有限。无止损机制。

---

### 策略D: CSI300低波动TOP10

> 详细文档: [docs/strategy_d.md](docs/strategy_d.md)

**逻辑**: 从沪深300**当期真实成分股**中，选20周年化波动率最低的10只，等权配置，双周调仓。使用baostock获取历史成分股数据，完全消除生存偏差。

**参数**: `vol_lookback=20w, top_n=10, rebal_freq=2w, txn_cost=8bp`

**回测 (~4.2年, 2022-01~2026-03)**: CAGR 13.9%, MDD -8.5%, Calmar 1.64

**偏差处理**:
- 生存偏差: baostock历史成分股API，回测每期仅使用该期真实成分股
- 前瞻偏差: T周数据做决策，T+1周收益计算

**注意**: v1版本（28只龙头股动量）已归档至`strategy_d_v1_weekly_signal.py`。

---

### 策略E: 逆向价值+质量

> 详细文档: [docs/strategy_e.md](docs/strategy_e.md)

**逻辑**: 在28只A股龙头中，选10周跌幅最大的（逆向）x 低波质量过滤，取Top4等权配置。过滤4周跌幅>20%的暴跌股。

**参数**: `value_lookback=10w, quality_weight=0.2, top_n=4, sector_max=1, txn_cost=8bp`

**回测**: CAGR 22.0%, MDD -17.6%, Sharpe 0.937

**重要偏差警告**: 28只龙头股是手工选择的当前行业领军者，存在严重后视选择偏差。建议将回测结果打折30-50%来估计真实未来表现。

---

### 策略E v2: 增强价值投资

> 基于策略E，加入PE/PB基本面价值因子

**逻辑**: 在原版策略E基础上，加入akshare获取的PE/PB基本面数据。多因子合成:
- 30% PE/PB基本面价值（近3年历史百分位）
- 20% 逆向动量（10周负动量）
- 30% 低波动质量
- 20% 盈利质量（ROE-PB组合，用PB/PE作ROE代理）

**参数**: `sector_max=1, top_n=4, monthly rebalance`

**偏差警告**: 同策略E，28只龙头股存在后视选择偏差。

---

### 策略F: 美股QDII动量轮动

> 详细文档: [docs/strategy_f.md](docs/strategy_f.md)

**逻辑**: 纳指100 vs 标普500按4周动量排名，Top1持100%。纳指低于20周MA时进入防守，全仓短债ETF。

**ETF池**:

| 指数 | ETF | 代码 | 定位 |
|------|-----|------|------|
| 中证纳斯达克100 (H30533) | 广发纳指100ETF | 159941 | 美股科技 |
| 中证标普500 (H30140) | 博时标普500ETF | 513500 | 美股宽基 |
| 中证短债 (H11006) | 海富通短融ETF | 511360 | 避险 |

**参数**: `momentum_lookback=4, ma_fast=8w, ma_slow=20w, txn_cost=15bp`

**回测 (15年)**: CAGR 13.0%, MDD -18.9%, Sharpe 0.809

**注意**: 使用中证编制的人民币计价指数回测，已含汇率波动。实盘需注意QDII限额和折溢价风险。

---

### 策略G: CSI500低波价值

> 详细文档: [docs/strategy_g.md](docs/strategy_g.md)

**逻辑**: 从中证500**当期真实成分股**中，综合50%低波动+50%低PB因子打分，选Top10等权配置。本质是"价值均值回归"——在中盘股中买便宜且稳定的标的。

**因子构成**:
- 50% 低波动率（12周滚动波动率，越低越好）
- 50% 低PB（市净率越低越好）
- 截面Z-score标准化后加权

**参数**: `vol_lookback=12w, factor_weights={low_vol:0.5, low_pb:0.5}, top_n=10, rebal_freq=2w, txn_cost=8bp`

**回测 (~5年, 2021-01~2026-03)**: CAGR 13.5%, MDD -11.9%, Calmar 1.13, 12月滚动胜率 97.5%

**偏差处理**:
- 生存偏差: baostock历史成分股（928只唯一股票，11个调仓期）
- 前瞻偏差: PE/PB用日频数据resample到周（取调仓日前最后一个值）
- 数据覆盖: 约400/928只股票有PE/PB数据，只选有完整数据的股票

**策略逻辑**: 中证500纯动量完全失效（测试过，CAGR全为负）。低PB捕捉"真正便宜"的股票，低波动过滤投机/困境股。

---

### 策略H: 指数超跌买入/追涨买入

> 详细文档: [docs/strategy_h.md](docs/strategy_h.md)

**逻辑**: 系统化捕捉短期价格极端——超跌后买入（均值回归）或大涨后追入（动量延续），持有固定天数后退出。

**6个验证通过的子变体（全部为科创50）**:

| 变体 | 方向 | 信号 | 持有 | Sharpe | CAGR | MDD | 胜率 |
|------|------|------|------|--------|------|-----|------|
| H1 | 超跌 | 3日跌>7% | 4日 | 1.04 | 13.9% | -6.6% | 72% |
| H2 | 追涨 | 5日涨>7% | 20日 | 0.68 | 15.9% | -23.4% | 57% |
| H3 | 超跌 | 3日跌>8% | 4日 | 1.07 | 13.4% | -5.5% | 83% |
| H4 | 超跌 | 4日跌>7% | 3日 | 1.03 | 13.7% | -8.9% | 75% |
| H5 | 超跌 | 3日跌>7% | 1日 | 0.90 | 8.6% | -3.0% | 74% |
| H6 | 超跌 | 6日跌>10% | 3日(SL-3%) | 1.13 | 12.8% | -3.6% | 83% |

**验证方法**: V2版本——IS期（2020-2022）参数搜索 + 邻域稳健性检验 + 盲OOS验证（2023-2026），修正了前瞻偏差（次日开盘价入场）、加入10bps交易成本。

**局限**: 全部变体集中在科创50（最高波动指数），依赖科创50波动率环境。

---

### 策略L: 300成长纯MA60趋势择时

> 详细文档: [docs/strategy_l.md](docs/strategy_l.md)

**逻辑**: 极简趋势跟踪——沪深300成长全收益指数收盘价在MA60上方满仓，下方空仓。T日生成信号，T+1执行。

**标的**: 沪深300成长全收益指数 (H00918)，对应ETF: 310398

**参数**: `trend_ma=60, txn_cost=8bp`

**回测 (~20.5年, 2005~2026)**:

| 指标 | 策略 | 买入持有 |
|------|------|---------|
| CAGR | 18.9% | 14.5% |
| Sharpe | 1.050 | — |
| Max Drawdown | -31.7% | >-70% |

**策略逻辑**: 300成长指数弹性大（成长风格），趋势跟踪效果好。MA60（约3个月均线）过滤掉大的下跌趋势（2008、2015、2018、2022大熊市都能躲过大部分），代价是震荡市会有来回交易成本。20年回测择时比买入持有CAGR高4.4个百分点，MDD从-70%降到-31.7%。

**偏差处理**: 信号T日生成T+1执行（shift(1)），严格无前瞻偏差。数据来源中证指数官网API。

---

### 个股因子策略: CSI800 低换手+12M动量

> 代码: `research/stock_strategy_live.py` | 研究系列: `research/stock_factor_research_v*.py`

**逻辑**: 从CSI800（沪深300+中证500）历史真实成分股中，用多因子模型选股:
1. 20日平均换手率（越低越好，过滤投机股）
2. 12个月动量-skip1M（避开短期反转噪音）
3. 多因子等权rank合成，选Top20等权持有
4. 每月底最后一个交易日调仓

**参数**: `top_n=20, monthly rebalance, TC=30bps one-side`

**回测 (2016-01~2026-03, ~10年)**: CAGR 21.4%, Sharpe 0.966, MDD -25.4%

**偏差处理**: 使用baostock历史成分股（非当前成分），消除生存偏差。

---

## 复现说明

### 数据源

| 策略 | 数据源 | 说明 |
|------|-------|------|
| A | CSIndex API | 中证指数公司官方债券指数日线 |
| B | CSIndex API (主) / 东方财富 (备) | 全收益指数日线 |
| C | Yahoo Finance API | v8 chart endpoint，日线 |
| D | baostock + Yahoo Finance | baostock获取成分股，Yahoo获取行情 |
| E | Yahoo Finance API | v8 chart endpoint，周线 |
| E v2 | Yahoo Finance + akshare | 价格(Yahoo) + PE/PB(akshare) |
| F | CSIndex API | 中证纳斯达克100/标普500指数 |
| G | baostock | 成分股+行情+基本面(PB) |
| H | baostock + akshare + yfinance | 本地CSV存储，可复现 |
| L | CSIndex API | 中证300成长全收益指数 |
| 个股因子 | akshare + baostock | 行情+成分股+换手率 |

**缓存策略**: 所有数据缓存3天，过期自动刷新。API不可用时使用过期缓存兜底。

### 完整复现步骤

```bash
# 1. 克隆代码
git clone https://github.com/sarahmitchell-cpu/etf-trader.git
cd etf-trader

# 2. 安装依赖
pip3 install -r requirements.txt

# 3. 运行回测（自动下载数据）
python3 strategy_a_weekly_signal.py --backtest   # 策略A
python3 strategy_b_weekly_signal.py --backtest   # 策略B
python3 strategy_c_weekly_signal.py --backtest   # 策略C
python3 strategy_d_weekly_signal.py --backtest   # 策略D
python3 strategy_e_weekly_signal.py --backtest   # 策略E
python3 strategy_e_v2_weekly_signal.py --backtest # 策略E v2
python3 strategy_f_weekly_signal.py --backtest   # 策略F
python3 strategy_g_weekly_signal.py --backtest   # 策略G
python3 strategy_h_weekly_signal.py --backtest   # 策略H
python3 strategy_l_weekly_signal.py --backtest   # 策略L

# 4. 生成JSON信号
python3 strategy_b_weekly_signal.py --json > signal_b.json

# 5. 策略D/G首次运行需baostock下载成分股数据，约需5分钟
```

### 回测数据

`data/` 目录包含预缓存的历史数据和回测结果：
- `*_daily.csv` / `*_weekly.csv`: 指数/个股历史行情
- `*_backtest.json` / `*_backtest_result.json`: 回测结果
- `*_latest_signal.json`: 最新信号
- `baostock_cache/`: 沪深300/中证500历史成分股缓存
- `fundamental_cache/`: 基本面数据缓存(PB等)
- `strategy_h/`: H策略本地CSV数据

---

## 目录结构

```
etf-trader/
|
+-- README.md                          # 策略总览与复现说明
+-- requirements.txt                   # Python依赖
|
+-- 策略信号脚本 (10套策略)
|   +-- strategy_a_weekly_signal.py    # A: 债券分散持有
|   +-- strategy_b_weekly_signal.py    # B: 因子ETF轮动 (v6)
|   +-- strategy_c_weekly_signal.py    # C: 跨境ETF纯动量
|   +-- strategy_d_weekly_signal.py    # D: CSI300低波动TOP10
|   +-- strategy_e_weekly_signal.py    # E: 逆向价值+质量
|   +-- strategy_e_v2_weekly_signal.py # E v2: 增强价值投资
|   +-- strategy_f_weekly_signal.py    # F: 美股QDII动量
|   +-- strategy_g_weekly_signal.py    # G: CSI500低波价值
|   +-- strategy_h_weekly_signal.py    # H: 科创50超跌买入
|   +-- strategy_l_weekly_signal.py    # L: 300成长MA60趋势
|   +-- stock_data_common.py           # D/E共用数据模块
|
+-- docs/                              # 详细策略文档
|   +-- strategy_a.md
|   +-- strategy_b.md
|   +-- strategy_c.md
|   +-- strategy_d.md
|   +-- strategy_d_v1.md              # D v1 (旧版28龙头)
|   +-- strategy_e.md
|   +-- strategy_f.md
|   +-- strategy_g.md
|   +-- strategy_h.md
|   +-- strategy_l.md
|
+-- research/                          # 研究&优化脚本
|   +-- stock_strategy_live.py         # 个股因子策略 (CSI800)
|   +-- stock_factor_research_v3.py    # 个股因子研究 V3
|   +-- stock_rank_contribution_analysis.py
|   +-- stock_momentum_window_test.py
|   +-- strategy_d_research.py
|   +-- csi300_multifactor_research.py
|   +-- csi500_*.py                    # CSI500因子研究系列
|   +-- strategy_de_optimizer*.py
|   +-- strategy_h_v2_validation.py
|   +-- factor_timing_*.py             # 因子择时研究
|   +-- trend_reversal_combo.py        # 趋势反转组合研究 (策略L来源)
|   +-- astock_10x_*.py               # A股筛选研究
|
+-- strategies/                        # 额外策略研究
|   +-- panic_buy_3factor_research.py
|   +-- panic_buy_signal.py
|   +-- strategy_j_growth_momentum.py
|   +-- strategy_k_growth_value_rotation.py
|
+-- data/                              # 数据缓存 (自动更新)
|   +-- *_daily.csv                    # 指数日线数据
|   +-- *_weekly.csv                   # 个股周线数据
|   +-- baostock_cache/                # 成分股历史数据
|   +-- fundamental_cache/             # 基本面数据
|   +-- strategy_h/                    # H策略本地数据
|   +-- stock_cache_v*/                # 个股因子策略缓存
|   +-- *_latest_signal.json           # 各策略最新信号
|   +-- *_backtest*.json               # 回测结果
|
+-- archived/                          # 历史版本/废弃脚本
|   +-- sector_rotation_v*.py
|
+-- generate_position_report_ppt.py    # PPT建仓报告生成器
```

---

## 关键设计原则

### 1. 无前瞻偏差 (No Look-Ahead Bias)

- **策略D/G**: 使用baostock历史成分股API，回测每期仅使用该期真实成分股，完全消除生存偏差
- **策略H V2**: 信号当日收盘检测，次日开盘价入场
- **策略L**: 信号T日生成，T+1执行（`shift(1)`）
- **所有策略**: T期数据做决策 -> T+1期收益计算

### 2. 包含交易成本

| 策略 | 单边成本 | 说明 |
|------|---------|------|
| A | 5bp | 债券ETF低成本 |
| B | 10bp | A股ETF（佣金+冲击） |
| C | 8bp | 跨境ETF |
| D/G | 8bp | A股个股 |
| E/E v2 | 8bp | A股个股 |
| F | 15bp | QDII ETF（含汇兑） |
| H | 10bp(双边) | 指数ETF |
| L | 8bp | A股ETF |
| 个股因子 | 30bp | A股个股（含冲击成本） |

### 3. 验证方法

| 策略 | 验证方法 |
|------|---------|
| D/G | baostock历史成分股消除生存偏差 |
| H | IS/OOS分割 + 邻域稳健性检验 + 盲OOS验证 |
| L | 20年超长回测（2005-2026），多周期覆盖 |
| 个股因子 | 10年回测（2016-2026），历史成分股 |
| B | 11年回测，含多轮牛熊 |
| E | 5折walk-forward交叉验证 |

---

## 已知局限性

- **策略B**: `vol_scale_cap=1.3`允许仓位达130%（隐含杠杆），需融资账户支持
- **策略C**: 仅4.2年数据，2025港股大涨贡献大部分收益，统计意义有限；无止损/回撤保护机制
- **策略D(v2)**: 回测仅~4.2年，统计检验力不足；持仓可能金融股集中度过高，无sector_max约束
- **策略E/E v2**: 28只龙头是当前视角选出的，存在严重后视选择偏差
- **策略F**: 15bp交易成本较高，QDII ETF折溢价风险未完全建模
- **策略G**: 回测仅~5年，约400/928只股票有PE/PB数据（baostock限制）
- **策略H**: 全部变体集中在科创50，依赖该指数高波动特性；IS仅3年数据
- **策略L**: 震荡市频繁交易产生摩擦成本；MDD -31.7%仍较大
- **Sharpe比率**: 各策略计算公式不完全一致，跨策略直接比较需谨慎

## 风险提示

- 本项目为个人量化研究工具，**不构成任何投资建议**
- 历史回测不代表未来收益
- 策略B在低波动环境下仓位可能超过100%（杠杆），放大亏损风险
- 策略C/L无下行保护，极端行情下可能遭受较大回撤
- 请根据自身风险承受能力决定是否参考

---

## 版本历史

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-03-29 | v4.0 | 新增策略H(指数超跌)/L(300成长MA60), 个股因子策略; 全策略文档完善; 10策略体系 |
| 2026-03-24 | v3.0 | 新增策略A(纯债)/F(美股QDII)/G(CSI500低波价值)，策略D改为CSI300低波TOP10，7策略模拟盘启动 |
| 2026-03-22 | v2.2 | 修复B vol scaling，抽取D/E公共模块，重跑全部回测 |
| 2026-03-22 | v2.0 | 新增策略C/D/E，整理仓库结构，添加详细文档 |
| 2026-03-21 | v1.5 | 策略D+E完成5轮优化，walk-forward验证 |
| 2026-03-18 | v1.0 | 初版：宏观定仓+行业轮动+策略B |

---

*由 VisionClaw AI 代理维护 · Sarah Mitchell · sarahmitchell@visionclaw.dev*
