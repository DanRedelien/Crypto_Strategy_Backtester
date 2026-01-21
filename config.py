from dataclasses import dataclass

@dataclass
class Config:
    # --- ОБЩИЕ ПАРАМЕТРЫ ---
    EXCHANGE_ID: str = 'binance'
    SYMBOL: str = 'BTCUSDT'
    TIMEFRAME: str = '1h'
    
    # Период теста
    START_DATE: str = '2025-01-01 00:00:00' 
    END_DATE: str = '2026-01-20 00:00:00'

    # --- УПРАВЛЕНИЕ КАПИТАЛОМ ---
    INITIAL_CAPITAL: float = 10_000.0
    MONTHLY_DEPOSIT: float = 1_000.0
    DEPOSIT_DAY: int = 1
    
    # --- ЭКОНОМИКА ---
    # 4% годовых на кэш (T-Bills / Staking stablecoins)
    RISK_FREE_RATE: float = 0.04    
    # 0.1% комиссия биржи + проскальзывание
    FEE_RATE: float = 0.001         

    # --- 1. СТРАТЕГИЯ PUT SPREAD (ЗАЩИТА) ---
    # Мы тратим фиксированный % от портфеля на страховку
    HEDGE_COST_PCT: float = 0.03   # 1.5% от Equity в месяц (Budget)
    
    # Параметры страйков (относительно цены входа)
    HEDGE_STRIKE_LONG_PCT: float = 0.95  # Начало защиты (-5%)
    HEDGE_STRIKE_SHORT_PCT: float = 0.80 # Лимит защиты (-20%)
    
    # Длительность хеджа
    HEDGE_DURATION_DAYS: int = 30

    # --- 2. СТРАТЕГИЯ ATM CALL (ЗАМЕНА СПОТА) ---
    # Предполагаемая волатильность для оценки стоимости ATM опциона
    # 55% - средняя волатильность BTC за последние годы
    CALL_STRAT_IV: float = 0.55 
    CALL_DURATION_DAYS: int = 30