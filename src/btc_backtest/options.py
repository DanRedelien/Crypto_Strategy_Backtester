"""
Option pricing models for Put Spread and ATM Call strategies.
Uses parametric approximations (no Black-Scholes dependency).
"""
import numpy as np
from datetime import datetime
from typing import Union
import pandas as pd


class ParametricPutSpread:
    """
    Models a vertical Put Spread (Bear Put Spread).
    Buy Put (Strike Long), Sell Put (Strike Short).
    """
    
    def __init__(
        self,
        spot_price: float,
        expiry_date: Union[datetime, pd.Timestamp],
        total_premium_paid: float,
        quantity: float,
        strike_long_pct: float,
        strike_short_pct: float
    ):
        self.entry_price = spot_price
        self.expiry = expiry_date
        
        # Calculate strikes based on config
        self.strike_long = spot_price * strike_long_pct
        self.strike_short = spot_price * strike_short_pct
        
        self.cost_total = total_premium_paid
        self.qty = quantity  # How many "synthetic bitcoins" we are insuring
    
    def get_total_payoff(self, current_spot: float) -> float:
        """
        Returns the total payoff in $ for the entire position volume at expiration.
        
        Payoff = (Max(Long - Price, 0) - Max(Short - Price, 0)) * Qty
        """
        # Payoff per unit
        val_long = max(self.strike_long - current_spot, 0)
        val_short = max(self.strike_short - current_spot, 0)
        unit_payoff = val_long - val_short
        
        return unit_payoff * self.qty
    
    def get_mtm_value(
        self,
        current_spot: float,
        current_date: Union[datetime, pd.Timestamp]
    ) -> float:
        """
        Mark-to-Market value of the position within the month.
        Sum of intrinsic value (if dropped) and remaining time value.
        """
        days_total = 30  # Simplification for decay
        days_left = (self.expiry - current_date).days
        time_decay_factor = max(days_left / days_total, 0)
        
        # 1. Current Intrinsic Value
        intrinsic_total = self.get_total_payoff(current_spot)
        
        # 2. Remaining Premium (Time Value)
        # Linear decay of paid premium
        remaining_premium = self.cost_total * time_decay_factor
        
        # In the market, an option costs either Intrinsic (deep ITM),
        # TimeValue (OTM), or Mixed.
        # For conservativeness, we take the maximum.
        return max(intrinsic_total, remaining_premium)


class ParametricATMCall:
    """
    Models the purchase of an ATM Call option (Long Call).
    Uses Brenner-Subrahmanyam approximation for pricing.
    """
    
    def __init__(
        self,
        spot_price: float,
        expiry_date: Union[datetime, pd.Timestamp],
        implied_vol: float,
        quantity: float
    ):
        self.strike = spot_price
        self.expiry = expiry_date
        self.iv = implied_vol
        self.qty = quantity
        
        # Calculate cost of 1 option via Brenner-Subrahmanyam approximation
        # Price ~ 0.4 * Spot * Vol * sqrt(Time)
        time_years = 30 / 365.0
        self.unit_cost = 0.4 * spot_price * self.iv * np.sqrt(time_years)
        
        self.total_cost = self.unit_cost * self.qty
    
    def get_total_payoff(self, current_spot: float) -> float:
        """Payoff at expiration."""
        unit_payoff = max(current_spot - self.strike, 0)
        return unit_payoff * self.qty
    
    def get_mtm_value(
        self,
        current_spot: float,
        current_date: Union[datetime, pd.Timestamp]
    ) -> float:
        """Current market valuation."""
        # Intrinsic
        unit_intrinsic = max(current_spot - self.strike, 0)
        
        # Time Value (sqrt decay approximation)
        total_life = 30
        days_left = (self.expiry - current_date).days
        time_fraction = max(days_left / total_life, 0)
        
        # Estimate current time value
        current_time_val_unit = self.unit_cost * np.sqrt(time_fraction)
        
        unit_price = unit_intrinsic + current_time_val_unit
        return unit_price * self.qty


class ParametricCollar:
    """
    Collar with Put Spread: Low-cost downside protection with capped upside.
    
    Structure:
    - Sell Call OTM (105%) → Generates premium, caps upside
    - Buy Put OTM (95%) → Costs premium, protection starts at -5%
    - Sell Put OTM (80%) → Generates premium, caps protection at -20%
    
    Net effect: Cheaper hedge than pure put spread, but limited upside.
    """
    
    def __init__(
        self,
        spot_price: float,
        expiry_date: Union[datetime, pd.Timestamp],
        quantity: float,
        implied_vol: float,
        strike_call_pct: float = 1.05,   # Sell Call at 105%
        strike_put_long_pct: float = 0.95,  # Buy Put at 95%
        strike_put_short_pct: float = 0.80  # Sell Put at 80%
    ):
        self.entry_price = spot_price
        self.expiry = expiry_date
        self.qty = quantity
        self.iv = implied_vol
        
        # Calculate strikes
        self.strike_call = spot_price * strike_call_pct
        self.strike_put_long = spot_price * strike_put_long_pct
        self.strike_put_short = spot_price * strike_put_short_pct
        
        # Calculate premiums using Brenner-Subrahmanyam approximation
        # For OTM options, scale by moneyness
        time_years = 30 / 365.0
        base_premium = 0.4 * spot_price * implied_vol * np.sqrt(time_years)
        
        # OTM adjustments (approximate)
        # Call 105%: ~70% of ATM premium
        # Put 95%: ~70% of ATM premium
        # Put 80%: ~30% of ATM premium (deep OTM)
        call_otm_factor = 0.30
        put_95_factor = 0.70
        put_80_factor = 0.30
        
        # Premium flows (per unit)
        self.call_premium = base_premium * call_otm_factor  # RECEIVE (short)
        self.put_long_premium = base_premium * put_95_factor  # PAY (long)
        self.put_short_premium = base_premium * put_80_factor  # RECEIVE (short)
        
        # Net premium = received - paid
        # Positive = net credit, Negative = net debit
        self.net_premium_unit = (
            self.call_premium + self.put_short_premium - self.put_long_premium
        )
        self.net_premium_total = self.net_premium_unit * quantity
    
    def get_total_payoff(self, current_spot: float) -> float:
        """
        Payoff at expiration.
        
        Payoff = (Put Spread Payoff) - (Call Loss if above strike)
        """
        # 1. Put spread payoff (protection)
        put_long_payoff = max(self.strike_put_long - current_spot, 0)
        put_short_payoff = max(self.strike_put_short - current_spot, 0)
        put_spread_payoff = put_long_payoff - put_short_payoff
        
        # 2. Short call loss (capped upside)
        call_loss = max(current_spot - self.strike_call, 0)
        
        # Total payoff per unit
        unit_payoff = put_spread_payoff - call_loss
        
        return unit_payoff * self.qty
    
    def get_mtm_value(
        self,
        current_spot: float,
        current_date: Union[datetime, pd.Timestamp]
    ) -> float:
        """Mark-to-Market value of collar position."""
        days_total = 30
        days_left = (self.expiry - current_date).days
        time_decay_factor = max(days_left / days_total, 0)
        
        # Intrinsic value
        intrinsic = self.get_total_payoff(current_spot)
        
        # Time value = remaining net premium
        time_value = self.net_premium_total * time_decay_factor
        
        # MTM = intrinsic + time value (can be negative if short call ITM)
        return intrinsic + time_value
