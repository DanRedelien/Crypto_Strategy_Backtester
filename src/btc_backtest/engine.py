"""
Backtest engine for DCA strategies with hedging.
Core simulation logic for Benchmark, Hedged (Put Spread), Call, and Collar strategies.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from .settings import Settings
from .options import ParametricPutSpread, ParametricATMCall, ParametricCollar


@dataclass
class BacktestResult:
    """Single timestep result for strategy comparison."""
    timestamp: pd.Timestamp
    bench_equity: float
    hedged_equity: float
    call_equity: float
    collar_equity: float
    price: float
    deposit_flow: float  # For correct TWR calculation
    invested_total: float


class BacktestEngine:
    """
    DCA backtest engine with multiple strategy variants.
    
    Strategies:
    1. Benchmark: Pure DCA into spot BTC
    2. Hedged: DCA + monthly put spread protection
    3. Call: Treasuries + ATM call options (synthetic exposure)
    4. Collar: DCA + collar (sell call, buy put spread) for low-cost hedge
    """
    
    def __init__(self, settings: Settings, data: pd.DataFrame):
        self.settings = settings
        self.data = data
        
        # --- INIT STATE ---
        # 1. Benchmark (DCA Spot)
        self.bench_cash = self.settings.INITIAL_CAPITAL
        self.bench_btc = 0.0
        
        # 2. Hedged (Spot + Put Spread)
        self.hedged_cash = self.settings.INITIAL_CAPITAL
        self.hedged_btc = 0.0
        self.active_hedge: Optional[ParametricPutSpread] = None
        self.hedge_stats = {'cost': 0.0, 'payoff': 0.0}
        
        # 3. Call Replacement (Treasuries + Calls)
        self.call_cash = self.settings.INITIAL_CAPITAL
        self.active_call: Optional[ParametricATMCall] = None
        self.call_stats = {'premium_paid': 0.0, 'payoff': 0.0}
        
        # 4. Collar Strategy (Spot + Collar)
        self.collar_cash = self.settings.INITIAL_CAPITAL
        self.collar_btc = 0.0
        self.active_collar: Optional[ParametricCollar] = None
        self.collar_stats = {'premium_received': 0.0, 'payoff': 0.0}
        
        # Helpers
        self.last_month = -1
        self.invested_total = self.settings.INITIAL_CAPITAL
        self.final_stats: Dict[str, float] = {}
    
    def run(self) -> pd.DataFrame:
        """
        Run the backtest simulation.
        
        Returns:
            pd.DataFrame with equity curves for all strategies.
        """
        history = []
        print("Starting Simulation...")
        
        # Initial Buy
        start_price = self.data.iloc[0]['close']
        self._buy_spot_bench(start_price)
        self._buy_spot_hedged(start_price)
        self._buy_spot_collar(start_price)
        # Call strategy buys the option in the first loop step
        
        for i, row in self.data.iterrows():
            date = row['timestamp']
            price = row['close']
            deposit_flow = 0.0
            
            # --- 1. MONTHLY DCA ---
            if date.day == self.settings.DEPOSIT_DAY and date.month != self.last_month:
                self.last_month = date.month
                
                # Deposit
                dep = self.settings.MONTHLY_DEPOSIT
                deposit_flow = dep
                
                self.bench_cash += dep
                self.hedged_cash += dep
                self.call_cash += dep
                self.collar_cash += dep
                self.invested_total += dep
                
                # Buy Spot
                self._buy_spot_bench(price)
                self._buy_spot_hedged(price)
                self._buy_spot_collar(price)
            
            # --- 2. STRATEGY LOGIC ---
            self._process_hedge_strategy(date, price)
            self._process_call_strategy(date, price)
            self._process_collar_strategy(date, price)
            
            # --- 3. EQUITY CALCULATION ---
            
            # A. Benchmark
            eq_bench = self.bench_cash + (self.bench_btc * price)
            
            # B. Hedged
            hedge_mtm = 0.0
            if self.active_hedge:
                hedge_mtm = self.active_hedge.get_mtm_value(price, date)
            eq_hedged = self.hedged_cash + (self.hedged_btc * price) + hedge_mtm
            
            # C. Call Strategy
            call_mtm = 0.0
            if self.active_call:
                call_mtm = self.active_call.get_mtm_value(price, date)
            eq_call = self.call_cash + call_mtm
            
            # D. Collar Strategy
            collar_mtm = 0.0
            if self.active_collar:
                collar_mtm = self.active_collar.get_mtm_value(price, date)
            eq_collar = self.collar_cash + (self.collar_btc * price) + collar_mtm
            
            history.append(BacktestResult(
                timestamp=date,
                bench_equity=eq_bench,
                hedged_equity=eq_hedged,
                call_equity=eq_call,
                collar_equity=eq_collar,
                price=price,
                deposit_flow=deposit_flow,
                invested_total=self.invested_total
            ))
            
            self.final_stats = {
                'hedge_cost': self.hedge_stats['cost'],
                'hedge_payoff': self.hedge_stats['payoff'],
                'call_premium': self.call_stats['premium_paid'],
                'call_payoff': self.call_stats['payoff'],
                'collar_premium_received': self.collar_stats['premium_received'],
                'collar_payoff': self.collar_stats['payoff']
            }
        
        return pd.DataFrame(history)
    
    # --- HELPERS ---
    
    def _buy_spot_bench(self, price: float) -> None:
        """Buy spot with all cash (Benchmark)."""
        if self.bench_cash > 0.01:
            amount = (self.bench_cash * (1 - self.settings.FEE_RATE)) / price
            self.bench_btc += amount
            self.bench_cash = 0
    
    def _buy_spot_hedged(self, price: float) -> None:
        """Buy spot with all cash (Hedged)."""
        # Strategy holds 100% in BTC, hedge is paid by selling part of BTC
        if self.hedged_cash > 0.01:
            amount = (self.hedged_cash * (1 - self.settings.FEE_RATE)) / price
            self.hedged_btc += amount
            self.hedged_cash = 0
    
    def _process_hedge_strategy(self, date: pd.Timestamp, price: float) -> None:
        """Process put spread hedge logic."""
        # 1. Check Expiry
        if self.active_hedge and date >= self.active_hedge.expiry:
            # Receive payoff (Cash Settlement)
            payoff = self.active_hedge.get_total_payoff(price)
            if payoff > 0:
                self.hedged_cash += payoff
                self.hedge_stats['payoff'] += payoff
            
            self.active_hedge = None
        
        # 2. Open New Hedge (if none exists)
        if self.active_hedge is None:
            # Calculate current portfolio Equity
            current_equity = (self.hedged_btc * price) + self.hedged_cash
            
            # Hedge Budget (Cost)
            hedge_budget = current_equity * self.settings.HEDGE_COST_PCT
            
            # How much BTC to sell to pay for hedge?
            # Per MATH_SPEC.md §4: Fee is deducted from sale proceeds
            # To receive NET = deficit, we must sell GROSS = deficit / (1 - fee_rate)
            if self.hedged_cash < hedge_budget:
                deficit = hedge_budget - self.hedged_cash
                # Sell BTC: gross_value = btc * price, net_received = gross * (1 - fee)
                # So: btc_to_sell = deficit / (price * (1 - fee_rate))
                btc_to_sell = deficit / (price * (1 - self.settings.FEE_RATE))
                if self.hedged_btc >= btc_to_sell:
                    self.hedged_btc -= btc_to_sell
                    self.hedged_cash += deficit
            
            # Pay
            self.hedged_cash -= hedge_budget
            self.hedge_stats['cost'] += hedge_budget
            
            # Create hedge object
            # QTY: How much BTC are we protecting? (Notional = Equity)
            qty_to_hedge = current_equity / price
            
            expiry = date + pd.Timedelta(days=self.settings.HEDGE_DURATION_DAYS)
            
            self.active_hedge = ParametricPutSpread(
                spot_price=price,
                expiry_date=expiry,
                total_premium_paid=hedge_budget,
                quantity=qty_to_hedge,
                strike_long_pct=self.settings.HEDGE_STRIKE_LONG_PCT,
                strike_short_pct=self.settings.HEDGE_STRIKE_SHORT_PCT
            )
    
    def _process_call_strategy(self, date: pd.Timestamp, price: float) -> None:
        """Process ATM call strategy logic."""
        # 1. Accrue Interest on Cash (Hourly)
        r_hour = self.settings.RISK_FREE_RATE / (365 * 24)
        self.call_cash *= (1 + r_hour)
        
        # 2. Check Expiry
        if self.active_call and date >= self.active_call.expiry:
            payoff = self.active_call.get_total_payoff(price)
            self.call_cash += payoff
            self.call_stats['payoff'] += payoff
            self.active_call = None
        
        # 3. Open New Call
        if self.active_call is None:
            # We want 1:1 exposure to capital.
            # If we have $10,000 cash, we want to control $10,000 in Bitcoin.
            # Qty = Capital / Price
            
            expiry = date + pd.Timedelta(days=self.settings.CALL_DURATION_DAYS)
            
            # Preliminary Qty calculation
            available_capital = self.call_cash
            qty_needed = available_capital / price
            
            temp_call = ParametricATMCall(price, expiry, self.settings.CALL_STRAT_IV, qty_needed)
            total_premium = temp_call.total_cost
            
            # ATM call premium usually 4-5%. We have 100% cash. Plenty.
            self.call_cash -= total_premium
            self.call_stats['premium_paid'] += total_premium
            self.active_call = temp_call
    
    def _buy_spot_collar(self, price: float) -> None:
        """Buy spot with all cash (Collar)."""
        if self.collar_cash > 0.01:
            amount = (self.collar_cash * (1 - self.settings.FEE_RATE)) / price
            self.collar_btc += amount
            self.collar_cash = 0
    
    def _process_collar_strategy(self, date: pd.Timestamp, price: float) -> None:
        """
        Process collar strategy logic.
        
        Collar = Sell Call OTM + Buy Put OTM + Sell Put OTM
        Net premium can be credit (received) or debit (paid).
        """
        # 1. Check Expiry
        if self.active_collar and date >= self.active_collar.expiry:
            payoff = self.active_collar.get_total_payoff(price)
            # Payoff can be positive (put protection) or negative (call capped)
            self.collar_cash += payoff
            self.collar_stats['payoff'] += payoff
            self.active_collar = None
        
        # 2. Open New Collar (if none exists)
        if self.active_collar is None:
            current_equity = (self.collar_btc * price) + self.collar_cash
            qty_to_collar = current_equity / price
            
            expiry = date + pd.Timedelta(days=self.settings.COLLAR_DURATION_DAYS)
            
            collar = ParametricCollar(
                spot_price=price,
                expiry_date=expiry,
                quantity=qty_to_collar,
                implied_vol=self.settings.COLLAR_IV,
                strike_call_pct=self.settings.COLLAR_CALL_STRIKE_PCT,
                strike_put_long_pct=self.settings.COLLAR_PUT_LONG_PCT,
                strike_put_short_pct=self.settings.COLLAR_PUT_SHORT_PCT
            )
            
            # Net premium: positive = receive cash, negative = pay cash
            net_premium = collar.net_premium_total
            self.collar_cash += net_premium  # Receive credit or pay debit
            self.collar_stats['premium_received'] += net_premium
            
            self.active_collar = collar
