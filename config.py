from dataclasses import dataclass

@dataclass
class Config:
    # --- GENERAL PARAMETERS ---
    EXCHANGE_ID: str = 'binance'
    SYMBOL: str = 'BTCUSDT'
    TIMEFRAME: str = '1h'
    
    # Test Period
    START_DATE: str = '2025-01-01 00:00:00' 
    END_DATE: str = '2026-01-20 00:00:00'

    # --- CAPITAL MANAGEMENT ---
    INITIAL_CAPITAL: float = 10_000.0
    MONTHLY_DEPOSIT: float = 1_000.0
    DEPOSIT_DAY: int = 1
    
    # --- ECONOMICS ---
    # 4% APY on cash (T-Bills / Staking stablecoins)
    RISK_FREE_RATE: float = 0.04    
    # 0.1% exchange fee + slippage
    FEE_RATE: float = 0.001         

    # --- 1. PUT SPREAD STRATEGY (HEDGING) ---
    # We spend a fixed % of the portfolio on insurance
    HEDGE_COST_PCT: float = 0.03   # 1.5% of Equity per month (Budget)
    
    # Strike parameters (relative to entry price)
    HEDGE_STRIKE_LONG_PCT: float = 0.95  # Protection start (-5%)
    HEDGE_STRIKE_SHORT_PCT: float = 0.80 # Protection limit (-20%)
    
    # Hedge duration
    HEDGE_DURATION_DAYS: int = 30

    # --- 2. ATM CALL STRATEGY (SPOT REPLACEMENT) ---
    # Implied volatility for ATM option pricing
    # 55% - average BTC volatility over recent years
    CALL_STRAT_IV: float = 0.55 
    CALL_DURATION_DAYS: int = 30
