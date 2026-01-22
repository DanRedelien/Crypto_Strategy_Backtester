import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from config import Config
from options import ParametricPutSpread, ParametricATMCall

@dataclass
class BacktestResult:
    timestamp: pd.Timestamp
    bench_equity: float
    hedged_equity: float
    call_equity: float
    price: float
    deposit_flow: float # For correct TWR calculation
    invested_total: float

class BacktestEngine:
    def __init__(self, cfg: Config, data: pd.DataFrame):
        self.cfg = cfg
        self.data = data
        
        # --- INIT STATE ---
        # 1. Benchmark (DCA Spot)
        self.bench_cash = self.cfg.INITIAL_CAPITAL
        self.bench_btc = 0.0
        
        # 2. Hedged (Spot + Put Spread)
        self.hedged_cash = self.cfg.INITIAL_CAPITAL
        self.hedged_btc = 0.0
        self.active_hedge: Optional[ParametricPutSpread] = None
        self.hedge_stats = {'cost': 0.0, 'payoff': 0.0}
        
        # 3. Call Replacement (Treasuries + Calls)
        self.call_cash = self.cfg.INITIAL_CAPITAL
        self.active_call: Optional[ParametricATMCall] = None
        
        # Helpers
        self.last_month = -1
        self.invested_total = self.cfg.INITIAL_CAPITAL

    def run(self):
        history = []
        print("Starting Simulation: Corrected Unit Logic...")

        # Initial Buy
        start_price = self.data.iloc[0]['close']
        self._buy_spot_bench(start_price)
        self._buy_spot_hedged(start_price)
        # Call strategy itself will buy the option in the first loop step

        for i, row in self.data.iterrows():
            date = row['timestamp']
            price = row['close']
            deposit_flow = 0.0

            # --- 1. MONTHLY DCA ---
            if date.day == self.cfg.DEPOSIT_DAY and date.month != self.last_month:
                self.last_month = date.month
                
                # Deposit
                dep = self.cfg.MONTHLY_DEPOSIT
                deposit_flow = dep
                
                self.bench_cash += dep
                self.hedged_cash += dep
                self.call_cash += dep
                self.invested_total += dep
                
                # Buy Spot
                self._buy_spot_bench(price)
                self._buy_spot_hedged(price)

            # --- 2. STRATEGY LOGIC ---
            self._process_hedge_strategy(date, price)
            self._process_call_strategy(date, price)

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

            history.append(BacktestResult(
                timestamp=date,
                bench_equity=eq_bench,
                hedged_equity=eq_hedged,
                call_equity=eq_call,
                price=price,
                deposit_flow=deposit_flow,
                invested_total=self.invested_total
            ))

            self.final_stats = {
                'hedge_cost': self.hedge_stats['cost'],
                'hedge_payoff': self.hedge_stats['payoff']
            }

        return pd.DataFrame(history)

    # --- HELPERS ---

    def _buy_spot_bench(self, price):
        """ Buy spot with all cash (Benchmark) """
        if self.bench_cash > 0.01:
            amount = (self.bench_cash * (1 - self.cfg.FEE_RATE)) / price
            self.bench_btc += amount
            self.bench_cash = 0

    def _buy_spot_hedged(self, price):
        """ Buy spot with all cash (Hedged) """
        # Strategy holds 100% in BTC, hedge is paid by selling part of BTC
        if self.hedged_cash > 0.01:
            amount = (self.hedged_cash * (1 - self.cfg.FEE_RATE)) / price
            self.hedged_btc += amount
            self.hedged_cash = 0

    def _process_hedge_strategy(self, date, price):
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
            hedge_budget = current_equity * self.cfg.HEDGE_COST_PCT
            
            # How much BTC to sell to pay for hedge?
            # If cash is sufficient (e.g. from past payoff) - ok.
            # If not - rebalance.
            if self.hedged_cash < hedge_budget:
                deficit = hedge_budget - self.hedged_cash
                # Sell BTC
                btc_to_sell = (deficit * (1 + self.cfg.FEE_RATE)) / price
                if self.hedged_btc >= btc_to_sell:
                    self.hedged_btc -= btc_to_sell
                    self.hedged_cash += deficit
            
            # Pay
            self.hedged_cash -= hedge_budget
            self.hedge_stats['cost'] += hedge_budget
            
            # Create hedge object
            # QTY: How much BTC are we protecting? (Notional = Equity)
            qty_to_hedge = current_equity / price
            
            expiry = date + pd.Timedelta(days=self.cfg.HEDGE_DURATION_DAYS)
            
            self.active_hedge = ParametricPutSpread(
                spot_price=price,
                expiry_date=expiry,
                total_premium_paid=hedge_budget,
                quantity=qty_to_hedge,
                strike_long_pct=self.cfg.HEDGE_STRIKE_LONG_PCT,
                strike_short_pct=self.cfg.HEDGE_STRIKE_SHORT_PCT
            )

    def _process_call_strategy(self, date, price):
        # 1. Accrue Interest on Cash (Hourly)
        # r_hourly approx
        r_hour = self.cfg.RISK_FREE_RATE / (365 * 24)
        self.call_cash *= (1 + r_hour)
        
        # 2. Check Expiry
        if self.active_call and date >= self.active_call.expiry:
            payoff = self.active_call.get_total_payoff(price)
            self.call_cash += payoff
            self.active_call = None
            
        # 3. Open New Call
        if self.active_call is None:
            # We want 1:1 exposure to capital.
            # If we have $10,000 cash, we want to control $10,000 in Bitcoin.
            # Qty = Capital / Price
            
            # First create object to find unit price
            expiry = date + pd.Timedelta(days=self.cfg.CALL_DURATION_DAYS)
            
            # Preliminary Qty calculation
            available_capital = self.call_cash
            qty_needed = available_capital / price
            
            temp_call = ParametricATMCall(price, expiry, self.cfg.CALL_STRAT_IV, qty_needed)
            total_premium = temp_call.total_cost
            
            # Check: enough money for premium?
            # ATM call premium usually 4-5%. We have 100% cash. Plenty.
            
            self.call_cash -= total_premium
            self.active_call = temp_call
