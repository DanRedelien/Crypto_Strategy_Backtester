import numpy as np

class ParametricPutSpread:
    """ 
    Models a vertical Put Spread (Bear Put Spread).
    Buy Put (Strike Long), Sell Put (Strike Short).
    """
    def __init__(self, spot_price, expiry_date, total_premium_paid, quantity, strike_long_pct, strike_short_pct):
        self.entry_price = spot_price
        self.expiry = expiry_date
        
        # Calculate strikes based on config
        self.strike_long = spot_price * strike_long_pct
        self.strike_short = spot_price * strike_short_pct
        
        self.cost_total = total_premium_paid
        self.qty = quantity # How many "synthetic bitcoins" we are insuring

    def get_total_payoff(self, current_spot):
        """ 
        Returns the total payoff in $ for the entire position volume at expiration.
        Payoff = (Max(Long - Price, 0) - Max(Short - Price, 0)) * Qty
        """
        # Payoff per unit
        val_long = max(self.strike_long - current_spot, 0)
        val_short = max(self.strike_short - current_spot, 0)
        unit_payoff = val_long - val_short
        
        return unit_payoff * self.qty

    def get_mtm_value(self, current_spot, current_date):
        """ 
        Mark-to-Market value of the position within the month.
        Sum of intrinsic value (if dropped) and remaining time value.
        """
        days_total = 30 # Simplification for decay
        days_left = (self.expiry - current_date).days
        time_decay_factor = max(days_left / days_total, 0)
        
        # 1. Current Intrinsic Value
        intrinsic_total = self.get_total_payoff(current_spot)
        
        # 2. Remaining Premium (Time Value)
        # Linear decay of paid premium
        remaining_premium = self.cost_total * time_decay_factor
        
        # In the market, an option costs either Intrinsic (deep ITM), TimeValue (OTM), or Mixed.
        # For conservativeness, we take the maximum.
        return max(intrinsic_total, remaining_premium)


class ParametricATMCall:
    """ 
    Models the purchase of an ATM Call option (Long Call).
    """
    def __init__(self, spot_price, expiry_date, implied_vol, quantity):
        self.strike = spot_price
        self.expiry = expiry_date
        self.iv = implied_vol
        self.qty = quantity
        
        # Calculate cost of 1 option via Brenner-Subrahmanyam approximation
        # Price ~ 0.4 * Spot * Vol * sqrt(Time)
        time_years = 30 / 365.0
        self.unit_cost = 0.4 * spot_price * self.iv * np.sqrt(time_years)
        
        self.total_cost = self.unit_cost * self.qty

    def get_total_payoff(self, current_spot):
        """ Payoff at expiration """
        unit_payoff = max(current_spot - self.strike, 0)
        return unit_payoff * self.qty

    def get_mtm_value(self, current_spot, current_date):
        """ Current market valuation """
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
