# Crypto Strategy Backtester (DCA + Options)

A tool for backtesting cryptocurrency investment strategies. This project evaluates traditional Dollar Cost Averaging (DCA) against sophisticated hedging and asset-replacement techniques using options.

<img width="1578" height="799" alt="image" src="https://github.com/user-attachments/assets/434d13be-74b8-43de-8644-3fe70e956ddc" />


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

<img width="823" height="476" alt="image" src="https://github.com/user-attachments/assets/62977b71-8e11-4cec-b3b5-88da075256d3" />

## Limitations

* **Simplified Option Pricing**: Brenner-Subrahmanyam approximation, linear time decay
* **No IV Dynamics**: Fixed implied volatility, no smile/skew modeling
* **Perfect Execution**: No slippage, bid-ask spreads, or liquidity constraints
* **No Taxation**: Capital gains not modeled

## Development

Codebase architected with assistance from **Gemini 3.0 Pro** and **Claude Opus**.
