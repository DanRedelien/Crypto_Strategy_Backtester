# Crypto Strategy Backtester (DCA + Options)

An institutional-grade tool for backtesting cryptocurrency investment strategies. This project evaluates traditional Dollar Cost Averaging (DCA) against sophisticated hedging and asset-replacement techniques using options.

## Overview

The system simulates **four** distinct capital management scenarios:

| Strategy | Description | Risk Profile |
|----------|-------------|--------------|
| **Benchmark (DCA)** | Standard periodic spot purchases | Full upside, full downside |
| **Hedged (Put Spread)** | Spot + monthly Put Spread (95%/80%) | Reduced drawdown, premium drag |
| **Call Replacement** | Treasuries + ATM Call options | Synthetic exposure, cash earns interest |
| **Collar** | Spot + Sell Call OTM + Put Spread | Net credit, capped upside, limited protection |

## Project Structure

```
src/btc_backtest/
├── settings.py      # Pydantic-based configuration
├── data_lake.py     # Parquet caching layer
├── data_fetcher.py  # CCXT exchange connector
├── engine.py        # Core simulation logic
├── options.py       # Option pricing models
└── analytics.py     # Risk metrics & visualization
```

## Key Features

* **CVaR 95% (Expected Shortfall)** - Tail risk measurement beyond VaR
* **NAV-based Drawdown** - Deposit-adjusted risk metrics (per MATH_SPEC.md)
* **XIRR Calculation** - True annualized return with irregular cash flows
* **Parquet Caching** - Fast data reload with automatic cache management
* **Collar Strategy** - Net credit structure with configurable strikes

## Installation & Usage

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn ccxt pydantic-settings pyarrow

# Run backtest
python run.py
```

## Configuration

Edit `src/btc_backtest/settings.py`:

```python
START_DATE = "2022-01-01 00:00:00"
END_DATE = "2026-01-31 00:00:00"
INITIAL_CAPITAL = 10_000
MONTHLY_DEPOSIT = 1_000

# Collar parameters
COLLAR_CALL_STRIKE_PCT = 1.20   # Sell Call at 120%
COLLAR_PUT_LONG_PCT = 0.95      # Buy Put at 95%
COLLAR_PUT_SHORT_PCT = 0.70     # Sell Put at 70%
```

## Sample Output

```
METRIC                         | BENCH        | HEDGED       | CALL         | COLLAR      
-------------------------------------------------------------------------------------------
MWR / XIRR (Investor Yield)    | 31.36%       | 21.59%       | 3.66%        | 37.62%      
Total Net Profit               | $62,586      | $38,870      | $5,448       | $80,153     
Max Drawdown                   | -67.38%      | -42.31%      | -39.29%      | -31.81%     
CVaR 95% (Expected Shortfall)  | -1.34%       | -1.12%       | -0.88%       | -1.08%      
```

## Limitations

* **Simplified Option Pricing**: Brenner-Subrahmanyam approximation, linear time decay
* **No IV Dynamics**: Fixed implied volatility, no smile/skew modeling
* **Perfect Execution**: No slippage, bid-ask spreads, or liquidity constraints
* **No Taxation**: Capital gains not modeled

## Development

Codebase architected with assistance from **Gemini 3.0 Pro** and **Claude Opus**.

Framework documentation:
- `dev_context/MATH_SPEC.md` - Financial mathematics specifications
- `dev_context/CLEAN_CODE_MCP.md` - Code quality standards
- `dev_context/VISUALIZATION_SPEC.md` - Chart styling guidelines
