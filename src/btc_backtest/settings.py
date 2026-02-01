"""
Settings module using pydantic-settings for type-safe configuration.
Single source of truth for all parameters.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Backtest configuration with type validation.
    Loads from environment variables with QUANT_ prefix or .env file.
    """
    model_config = SettingsConfigDict(
        env_prefix="QUANT_",
        env_file=".env",
        extra="ignore"
    )

    # --- GENERAL PARAMETERS ---
    EXCHANGE_ID: str = "binance"
    SYMBOL: str = "BTCUSDT"
    TIMEFRAME: str = "1h"
    
    # Test Period
    START_DATE: str = "2022-01-01 00:00:00"
    END_DATE: str = "2026-01-31 00:00:00"

    # --- CAPITAL MANAGEMENT ---
    INITIAL_CAPITAL: float = Field(default=10_000.0, gt=0)
    MONTHLY_DEPOSIT: float = Field(default=1_000.0, ge=0)
    DEPOSIT_DAY: int = Field(default=1, ge=1, le=28)
    
    # --- ECONOMICS ---
    # 4% APY on cash (T-Bills / Staking stablecoins)
    RISK_FREE_RATE: float = 0.04
    # 0.1% exchange fee + slippage
    FEE_RATE: float = 0.0001

    # --- 1. PUT SPREAD STRATEGY (HEDGING) ---
    # Fixed % of the portfolio spent on insurance
    HEDGE_COST_PCT: float = 0.023  # 3% of Equity per month (Budget)
    
    # Strike parameters (relative to entry price)
    HEDGE_STRIKE_LONG_PCT: float = 0.95   # Protection start (-5%)
    HEDGE_STRIKE_SHORT_PCT: float = 0.80  # Protection limit (-20%)
    
    # Hedge duration
    HEDGE_DURATION_DAYS: int = 30

    # --- 2. ATM CALL STRATEGY (SPOT REPLACEMENT) ---
    # Implied volatility for ATM option pricing
    # 55% - average BTC volatility over recent years
    CALL_STRAT_IV: float = 0.55
    CALL_DURATION_DAYS: int = 30
    
    # --- 3. COLLAR STRATEGY (LOW-COST HEDGE) ---
    # Sell Call OTM + Buy Put OTM + Sell Put OTM
    # Goal: Reduce hedge cost by selling upside and deep OTM put
    COLLAR_CALL_STRIKE_PCT: float = 1.20   # Sell Call at 105% (cap upside at +5%)
    COLLAR_PUT_LONG_PCT: float = 0.95      # Buy Put at 95% (protection from -5%)
    COLLAR_PUT_SHORT_PCT: float = 0.70     # Sell Put at 80% (cap protection at -20%)
    COLLAR_IV: float = 0.55                # IV for premium calculation
    COLLAR_DURATION_DAYS: int = 30
    
    # --- INFRASTRUCTURE ---
    CACHE_DIR: Path = Path("data/cache")
    
    # --- ANNUALIZATION ---
    # Crypto trades 365 days/year (per MATH_SPEC.md §2)
    ANNUALIZATION_FACTOR: int = 365
    
    def get_cache_path(self) -> Path:
        """Returns cache directory path, creating it if needed."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self.CACHE_DIR


# Singleton Access
_settings: Settings | None = None


def get_settings() -> Settings:
    """Returns singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings