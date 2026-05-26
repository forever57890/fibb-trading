# fibb_trading — FiBB 15m BTC 多層通道策略

對應 TradingView 腳本 **「FiBB 15m BTC Layer Strategy Backtest」**：以 SMA + ATR × Fibonacci 倍率建構三層上下軌，觸軌分層進場、固定百分比止盈止損，支援最多 6 個獨立 leg（pyramiding）。

獨立專案，不依賴其他策略 repo。結構：`core`（策略）+ `backtest`（回測）+ `trade`（實盤）。

---

## 策略邏輯

### 通道

| 項目 | 公式 |
|------|------|
| basis | SMA(close, `length`)，預設 20 |
| avg | `ta.atr(length)`（Wilder RMA，與 TradingView 相同） |
| 上軌 T1/T2/T3 | basis + avg × (1.618 / 2.618 / 4.236) |
| 下軌 B1/B2/B3 | basis - avg × (同上) |

### 進場（當根 K 首次觸及該軌）

| 條件 | 方向 | 數量 (BTC) | entry_id |
|------|------|------------|----------|
| high 穿越 T1 | SHORT | 1 | T1 Short |
| high 穿越 T2 | SHORT | 2 | T2 Short |
| high 穿越 T3 | SHORT | 3 | T3 Short |
| low 穿越 B1 | LONG | 1 | B1 Long |
| low 穿越 B2 | LONG | 2 | B2 Long |
| low 穿越 B3 | LONG | 3 | B3 Long |

觸及定義（與 Pine 一致）：`high >= topX` 且 `high[1] < topX[1]`（做空）；`low <= bottX` 且 `low[1] > bottX[1]`（做多）。  
同一 `entry_id` 僅允許一筆未平倉（`isOpen`）。

### 出場（與目前 `FibbParams` 預設一致）

**止盈（預設：`use_channel_tp=False`）**  
全部 leg：進場收盤價 ± `tp_pct`（預設 0.5%），持倉期間止盈價**不**隨通道重算。

**可選：通道止盈（`use_channel_tp=True`）**  
T1/B1→`basis`；T2/B2→`top1`/`bott1`；T3/B3→`top2`/`bott2`；每根 K 更新止盈價。回測請加：`--channel-tp`。

**止損（預設：`use_deferred_channel_sl=True`）**

- **T1 / T3 / B1 / B3**：不設止損
- **T2 / B2**：進場無止損；觸及外層 **T3 / B3** 後，止損設在當根 **T2 / B2 通道價**
- 進場價：訊號 K 線**收盤價**（`process_orders_on_close`）
- 持倉期間用每根 K 的 high/low 檢查；同根 TP 與通道止損同時觸及時**止損優先**

若要 Pine 經典「每 leg 都有 % 止損」：Python 設 `use_deferred_channel_sl=False`（或舊版全 `bracket_prices`）。

---

## TradingView 如何改成與 Python 一致

Python 把 **T1～T3、B1～B3** 當成 **六筆互不影響的虛擬倉位**（可同時多、空並存）。TradingView 若仍是**單向／淨持倉**，新開的空單會先沖掉多單，成交清單會出現「B1 出場訊號顯示成 T1」這類現象，**不是**同一套策略。

請依序做：

1. **使用專案內 Pine**  
   將 `pine/FiBB_15m_BTC_Deferred_Channel_SL.pine` 貼到 Pine 編輯器並加入圖表（邏輯對應 `fibb_logic.py`：`process_orders_on_close`、觸軌進場、固定 % 止盈、T2/B2 延遲通道止損）。

2. **持倉模式改為「可同持多空」**（與 Python 六 leg 對齊的**必要**條件）  
   在 TradingView 策略／圖表設定中，改為 **Hedge／雙向／Long 與 Short 分開**（名稱依版本與商品類型而異）。  
   若維持單向，TV **無法**完整重現 Python 的成交路徑。

3. **商品與 K 線**  
   與 Python 預設一致：**Binance USD-M 永續** `BTCUSDT`、**15m**。比對時請固定圖表時區（建議 **UTC**），並與 Python `--start` / `--end` 使用同一語意。

4. **佣金**  
   Python 預設 `fee_rate=0`；TV 策略屬性佣金設 **0%**（腳本內 `commission_value = 0`）。

5. **參數**  
   Pine 的 `Length`、費波倍率、`Take Profit %`、回測起訖，請與 Python `FibbParams` / CLI 對齊。

6. **通道止盈**  
   若 Python 使用 `--channel-tp`，目前此 Pine **僅實作固定 % 止盈**；要完全對齊需再擴充 Pine（或改回 Python 不加 `--channel-tp` 與 TV 比對）。

匯出 TV 交易表後，可用 `python3 -m fibb_trading.backtest.compare_tv` 與 `trade_details.json` 比對。

---

## 專案結構

```
fibb_trading/
├── core/
│   ├── fibb_config.py    # 參數與 leg 定義
│   ├── fibb_logic.py     # 通道、訊號、回測引擎
│   └── data_fetch.py     # Binance K 線（15m 等）
├── backtest/
│   └── fibb_backtest.py  # CLI 回測
├── pine/
│   └── FiBB_15m_BTC_Deferred_Channel_SL.pine  # 與 Python 延遲止損版對齊的 TV 腳本
├── trade/                 # 實盤／排程（若已啟用）
├── path_setup.py
└── requirements.txt
```

---

## 快速開始

```bash
cd /path/to/fibb_trading
pip install -r requirements.txt

# 回測（拉 Binance 15m K 線）
python3 -m backtest.fibb_backtest \
  --start "2024-01-01 00:00:00" \
  --end "2026-12-31 23:59:59"

# 通道止盈回測（預設）
python3 -m fibb_trading.backtest.fibb_backtest \
  --start "2025-05-25 07:15:00" \
  --end "2026-05-25 22:30:00"

# 固定 % 止盈（舊版）
python3 -m fibb_trading.backtest.fibb_backtest --pct-tp --tp-pct 0.5 --sl-pct 0.5

# 調參
python3 -m backtest.fibb_backtest --length 20

# 與 TradingView 匯出的交易清單比對（CSV 時間預設為圖表時區 UTC+8）
python3 -m fibb_trading.backtest.compare_tv \
  --tv-csv ~/Downloads/FiBB_Backtest_*.csv \
  --py-json backtest/test_data/trade_details.json \
  --tz-offset 8

# 匯出輸贏不一致的前 20 筆（含平倉 K 線 high/low）
python3 -m fibb_trading.backtest.compare_tv \
  --tv-csv ~/Downloads/FiBB_Backtest_*.csv \
  --export-disagreements --limit 20
```

回測預設 `initial_capital=100000`、`leverage=2`。不檢查資金：`--initial-capital 0`。

完整套件路徑（父目錄需在 PYTHONPATH）：

```bash
export PYTHONPATH="$(dirname "$(pwd)"):${PYTHONPATH:-}"
python3 -m fibb_trading.backtest.fibb_backtest
```

---

## 實盤執行（15m）

### 1) 準備 `.env`

在專案根目錄建立 `.env`（可參考 `.env.example`）：

```env
bn_api_key=your_key
bn_api_secret=your_secret

# 先用模擬模式驗證流程
FIBB_DRY_RUN=1

# 首次可設 1，自動嘗試切換為雙向持倉（hedge mode）
FIBB_ENABLE_HEDGE_MODE=1

# 策略參數（與 Python 預設一致）
FIBB_SYMBOL=BTCUSDT
FIBB_TP_PCT=0.5
FIBB_DEFERRED_SL=1
FIBB_CHANNEL_TP=0
FIBB_FEE_RATE=0
FIBB_INITIAL_CAPITAL=100000
FIBB_LEVERAGE=2

# 開倉後先掛固定 % TP；之後每 15m 依 basis（中線）重掛 TP（1=開，0=關）
FIBB_REPRICE_TP_TO_BASIS=1
```

### 2) 單次執行（手動）

先跑模擬（不下單）：

```bash
export PYTHONPATH="$(dirname "$(pwd)"):${PYTHONPATH:-}"
FIBB_DRY_RUN=1 python3 -m fibb_trading.trade.fibb_15m_trader
```

確認輸出與 log 正常後，再改實盤：

```bash
export PYTHONPATH="$(dirname "$(pwd)"):${PYTHONPATH:-}"
FIBB_DRY_RUN=0 python3 -m fibb_trading.trade.fibb_15m_trader
```

### 3) 排程每 15 分鐘跑一次

專案已提供腳本：

```bash
bash trade/run_fibb_15m_cron.sh
```

crontab（UTC）範例：

```cron
TZ=UTC
1,16,31,46 * * * * /bin/bash /path/to/fibb_trading/trade/run_fibb_15m_cron.sh
```

### 4) 實盤檔案位置

- 狀態檔：`trade/runtime/fibb_15m_state.json`
- 每次執行記錄：`trade/runtime/fibb_15m_runs.log`（人類可讀總結 + JSON）
- cron 輸出：`trade/runtime/fibb_cron.log`（與終端機相同之總結區塊）

每次執行會輸出 **本根總結**（例如「本根無觸軌進場」「T1 Short 已有持倉」）及 **Leg 進場診斷**（六個 leg 是否觸軌、high/low 與通道價對照）。有持倉時會列出 **HOLD** 與止盈/止損價是否觸及。

環境變數：`FIBB_LOG_JSON=0` 可關閉 runs.log 內的 JSON 區塊；`FIBB_PRINT_JSON=1` 可在終端額外印完整 JSON。`FIBB_REPRICE_TP_TO_BASIS=0` 時持倉 TP 維持開倉時的固定 %，不每根 K 重掛至中線。

### 5) 重要風險與檢查

- 請先確認交易所帳戶為可雙向持倉（hedge），否則多空 legs 會互相沖銷。
- 請先以 `FIBB_DRY_RUN=1` 連跑一段時間，確認訊號與出場符合預期。
- 若要與回測對齊，請固定商品（USD-M `BTCUSDT`）、週期（15m）、佣金設定與參數。

---

## 輸出（`backtest/test_data/`）

| 檔案 | 內容 |
|------|------|
| `trade_details.json` | 每筆平倉 leg |
| `summary.json` | 淨利、勝率、profit factor、max drawdown |
| `leg_summary.json` | 各 T1/T2/T3/B1/B2/B3 分組 |
| `equity_drawdown.png` | 權益與回撤 |
| `fibb_channels_trades.png` | 通道與進場示意 |

---

## 與 fng_trading 差異

| | fng_trading | fibb_trading |
|---|-------------|--------------|
| 訊號來源 | CMC 恐慌指數日線 | Binance 15m K 線 |
| 持倉 | 單一方向、日線持倉 | 最多 6 leg 同時 |
| 出場 | 日 K TP/SL / 隔日平倉 | 固定 % TP/SL |
| 實盤 | 有 `trade/` | 有 `trade/`（FiBB 15m） |

---

## 風險提示

回測結果不代表未來表現。實盤前請自行驗證滑價、手續費與保證金要求。
