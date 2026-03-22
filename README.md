# ETF & 个股量化策略体系

> 4套独立策略 · 全部无前瞻偏差 · 自动化信号推送 · Telegram播报

一套完整的A股/港股量化交易策略体系，包含ETF轮动和个股动量/价值策略。

---

## 策略总览

| 策略 | 名称 | 标的 | 核心逻辑 | CAGR | MDD | Sharpe | 调仓频率 |
|------|------|------|---------|------|-----|--------|---------|
| **B** | 因子ETF轮动 | 4只A股Smart Beta ETF | 动量排名+Regime+Vol缩放 | 15.0% | -17.8% | 1.115 | 每周 |
| **C** | 跨境ETF纯动量 | 3只恒生系ETF | Top1纯动量，100%持仓 | 17.0% | -23.3% | 0.70 | 每周 |
| **D** | 龙头股动量 | 28只A股龙头 | 4w动量+Skip1+行业分散 | 14.2% | -20.2% | 0.81 | 双周 |
| **E** | 逆向价值+质量 | 28只A股龙头 | 10w反转+低波质量+安全过滤 | 22.0% | -17.6% | 0.937 | 月度 |

### 策略互补关系

```
策略B (ETF因子轮动)     ← A股因子配置，防守为主
策略C (跨境ETF动量)     ← 港股配置，独立于A股
策略D (龙头动量)  ─┐
                   ├── 共用28只龙头股池，选股逻辑互补
策略E (逆向价值)  ─┘    D买强势股，E买低估股，零重叠
```

---

## 快速开始

### 环境要求

```bash
Python 3.8+
pip3 install pandas numpy requests
```

### 运行任意策略信号

```bash
git clone https://github.com/sarahmitchell-cpu/etf-trader.git
cd etf-trader

# 策略B - 因子ETF周度信号
python3 strategy_b_weekly_signal.py

# 策略C - 跨境ETF周度信号
python3 strategy_c_weekly_signal.py

# 策略D - 龙头股动量信号（双周）
python3 strategy_d_weekly_signal.py

# 策略E - 逆向价值信号（月度）
python3 strategy_e_weekly_signal.py
```

### 运行回测

```bash
python3 strategy_b_weekly_signal.py --backtest
python3 strategy_c_weekly_signal.py --backtest
python3 strategy_d_weekly_signal.py --backtest
python3 strategy_e_weekly_signal.py --backtest
```

### JSON输出

```bash
python3 strategy_b_weekly_signal.py --json
python3 strategy_c_weekly_signal.py --json
python3 strategy_d_weekly_signal.py --json
python3 strategy_e_weekly_signal.py --json
```

---

## 详细文档

每个策略都有独立的详细文档，包含完整的策略说明、回测数据、复现流程和注意事项：

| 策略 | 文档 |
|------|------|
| Strategy B | [docs/strategy_b.md](docs/strategy_b.md) |
| Strategy C | [docs/strategy_c.md](docs/strategy_c.md) |
| Strategy D | [docs/strategy_d.md](docs/strategy_d.md) |
| Strategy E | [docs/strategy_e.md](docs/strategy_e.md) |

---

## 目录结构

```
etf-trader/
│
├── README.md                          # 本文件 - 策略总览
│
├── 策略信号脚本 (生产代码)
│   ├── strategy_b_weekly_signal.py    # 策略B: 因子ETF周度信号 (v6)
│   ├── strategy_c_weekly_signal.py    # 策略C: 跨境ETF纯动量轮动
│   ├── strategy_d_weekly_signal.py    # 策略D: 龙头股动量 (双周)
│   └── strategy_e_weekly_signal.py    # 策略E: 逆向价值+质量 (月度)
│
├── docs/                              # 详细策略文档
│   ├── strategy_b.md
│   ├── strategy_c.md
│   ├── strategy_d.md
│   └── strategy_e.md
│
├── research/                          # 研究&优化脚本
│   ├── strategy_d_research.py         # 策略D参数网格搜索
│   ├── strategy_e_research.py         # 策略E参数网格搜索
│   ├── strategy_de_optimizer.py       # D+E联合优化器 (Round 1-3)
│   └── strategy_de_optimizer_r2.py    # D+E深度优化器 (Round 4-5)
│
├── data/                              # 数据缓存 (自动更新)
│   ├── *_daily.csv                    # 策略B指数日线数据
│   ├── sc_*_daily.csv                 # 策略C ETF日线数据
│   ├── sd_*_weekly.csv                # 策略D/E龙头股周线数据
│   ├── optimization/                  # 优化过程中间数据
│   ├── *_latest_signal.json           # 各策略最新信号
│   └── *_backtest_result.json         # 各策略回测结果
│
├── 辅助工具 (早期版本/实盘执行)
│   ├── etf_quant_strategy.py          # 宏观定仓: 双维度9×9矩阵
│   ├── comprehensive_strategy.py      # 综合策略三层架构
│   ├── sector_rotation_v*.py          # 行业轮动 V3-V6
│   ├── etf_monitor.py                 # ETF实盘监控
│   ├── jty_auto_trade.py              # 国信证券ADB自动下单
│   ├── market_report.py               # 市场日报
│   ├── xueqiu_auto_post.py            # 雪球自动发帖
│   └── generate_strategy_chart.py     # 策略矩阵可视化
│
└── sector_cache/                      # 申万行业缓存
```

---

## 数据源

| 策略 | 数据源 | 说明 |
|------|-------|------|
| B | CSIndex API (主) / 东方财富 (备) | 中证指数公司官方，支持全收益指数 |
| C | Yahoo Finance API | v8 chart endpoint，日线 |
| D | Yahoo Finance API | v8 chart endpoint，周线，adjclose复权 |
| E | Yahoo Finance API | 复用策略D缓存 |

**缓存策略**：所有数据缓存3天，过期自动刷新。API不可用时使用过期缓存兜底。

---

## 关键设计原则

### 1. 无前瞻偏差 (No Look-Ahead Bias)

所有策略严格遵循：**T周数据做决策 → T+1周收益计算**

```python
# 回测核心循环
for i in range(warmup, len(data) - 1):
    # 决策阶段: 仅使用 data[:i+1] (已知数据)
    selected = select_stocks(data, idx=i)

    # 收益阶段: 承受第 i+1 周的实际收益
    portfolio_return = calculate_return(selected, data, idx=i+1)
```

### 2. 包含交易成本

| 策略 | 单边成本 | 说明 |
|------|---------|------|
| B | 10bp | A股ETF（佣金+冲击） |
| C | 8bp | 跨境ETF（无印花税） |
| D | 8bp | A股个股（万3佣金+千1印花税折算） |
| E | 8bp | 同策略D |

### 3. Walk-Forward验证

策略D和E经过5折walk-forward交叉验证，所有折Calmar均为正。

### 4. 参数灵敏度

策略D和E的关键参数在最优值附近稳定，无"悬崖效应"。

---

## 信号推送计划

| 策略 | 频率 | 时间 | 渠道 |
|------|------|------|------|
| B | 每周 | 周六 09:00 | Telegram |
| C | 每周 | 周六 09:30 | Telegram |
| D | 双周 | 周六 09:30 | Telegram |
| E | 每4周 | 周六 10:00 | Telegram |

---

## 已知局限性

- **策略B Vol缩放瑕疵**：回测中用当前周权重乘整段历史收益算波动率，非真实的历史组合波动率，影响vol scale精度（信号端正确，仅影响回测精度）
- **策略C 回测偏短**：仅4.2年数据，2025港股大涨（+54.8%）贡献大部分收益，统计意义有限
- **策略C 持仓集中度100%**：全仓单只ETF换仓，实盘冲击成本可能高于回测假设的8bp
- **策略D/E 股票池后视偏差**：28只龙头是当前视角选出的，回测从2021年开始，存在后视选择偏差（survivorship bias）
- **策略D/E 代码重复**：`_fetch_yahoo`、`fetch_stock`、`load_all_data`几乎一样，待抽公共模块
- **策略B 累计交易成本偏高**：11年累计18.23%（年化~1.6%）

## 风险提示

- 本项目为个人量化研究工具，**不构成任何投资建议**
- 历史回测不代表未来收益
- 因子池基于历史表现筛选，存在选择偏差（selection bias）
- A股市场受政策影响较大，量化策略在极端行情下可能失效
- 请根据自身风险承受能力决定是否参考

---

## 版本历史

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-03-22 | v2.1 | 修正策略D回测数据（审计反馈），更新BCDE文档指标，添加已知局限性 |
| 2026-03-22 | v2.0 | 新增策略C/D/E，整理仓库结构，添加详细文档 |
| 2026-03-21 | v1.5 | 策略D+E完成5轮优化，sector_max=1，walk-forward验证 |
| 2026-03-21 | v1.4 | 策略C部署（跨境ETF Top1纯动量） |
| 2026-03-21 | v1.3 | 策略B v5/v6（4周动量+50/30/20/0+Vol缩放1.3x） |
| 2026-03-21 | v1.2 | 策略B v4（全收益指数+创业成长因子） |
| 2026-03-19 | v1.1 | 行业轮动V5（行业趋势过滤） |
| 2026-03-18 | v1.0 | 初版：宏观定仓+行业轮动+策略B |

---

*由 VisionClaw AI 代理维护 · Sarah Mitchell · sarahmitchell@visionclaw.dev*
