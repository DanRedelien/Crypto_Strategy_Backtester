# Crypto Strategy Backtester (DCA + Options)

An institutional-grade tool for backtesting cryptocurrency investment strategies. This project evaluates traditional Dollar Cost Averaging (DCA) against sophisticated hedging and asset-replacement techniques using options.

## Overview

The system simulates three distinct capital management scenarios:

1. **Benchmark (DCA Spot):** Standard periodic asset purchases at market price.
2. **Hedged Strategy:** Spot asset ownership combined with a **Put Spread** to protect against tail risk (market crashes).
3. **Call Replacement:** A "Cash + Call" approach where capital is kept in interest-bearing instruments while market exposure is gained via ATM (At-The-Money) Call options.

### Key Features:

* **Institutional Analytics:** Calculation of XIRR (Internal Rate of Return), Ulcer Index (stress measure), and NAV-based Maximum Drawdown.
* **Option Modeling:** Parametric valuation of Put Spreads and Long Calls accounting for time decay (Theta) and intrinsic value.
* **Automated Data:** Seamless integration with Binance via the CCXT library.

## Installation & Usage

1. **Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn ccxt

```


2. **Configure parameters:** Set your dates, initial capital, and option strike percentages in `config.py`.
3. **Run the simulation:**
```bash
python main.py

```

## Performance Visualization
<img width="1558" height="795" alt="image" src="https://github.com/user-attachments/assets/5c5c0785-9631-46b9-a4db-186878eb0754" />

## Professional Metrics

The engine generates a detailed console report including:

* **XIRR:** Annualized return accounting for irregular monthly deposits.
* **Alpha:** Excess return relative to the DCA benchmark.
* **Crisis Coverage:** Efficiency of the insurance (Put Spreads) during market sell-offs.

<img width="681" height="524" alt="image" src="https://github.com/user-attachments/assets/844e638c-5d08-436b-abcb-57090576b2ac" />

## Limitations & Disclaimers

Users should be aware of the following model assumptions:

* **Simplified Option Pricing:** Option values are calculated using the Brenner-Subrahmanyam approximation and linear time decay. This may deviate from real-world premiums during periods of extreme volatility (IV spikes).
* **Execution & Liquidity:** The backtest assumes perfect execution at hourly close prices. It does not account for slippage or the wide bid-ask spreads often found in crypto option markets.
* **Taxation:** The model does not include capital gains tax or other fiscal obligations.

**Development Note:** This codebase was primarily architected and written by **Gemini 3.0** and **Claude**. While it implements complex financial logic (XIRR, NAV adjustments), it should be thoroughly reviewed before being used for actual trading or financial decisions.
