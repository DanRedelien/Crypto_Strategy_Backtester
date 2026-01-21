import numpy as np

class ParametricPutSpread:
    """ 
    Моделирует вертикальный Put Spread (Bear Put Spread).
    Покупаем Put (Strike Long), Продаем Put (Strike Short).
    """
    def __init__(self, spot_price, expiry_date, total_premium_paid, quantity, strike_long_pct, strike_short_pct):
        self.entry_price = spot_price
        self.expiry = expiry_date
        
        # Рассчитываем страйки на основе конфига
        self.strike_long = spot_price * strike_long_pct
        self.strike_short = spot_price * strike_short_pct
        
        self.cost_total = total_premium_paid
        self.qty = quantity # Сколько "синтетических биткоинов" мы страхуем

    def get_total_payoff(self, current_spot):
        """ 
        Возвращает полную выплату в $ для всего объема позиции при экспирации.
        Payoff = (Max(Long - Price, 0) - Max(Short - Price, 0)) * Qty
        """
        # Выплата на 1 юнит
        val_long = max(self.strike_long - current_spot, 0)
        val_short = max(self.strike_short - current_spot, 0)
        unit_payoff = val_long - val_short
        
        return unit_payoff * self.qty

    def get_mtm_value(self, current_spot, current_date):
        """ 
        Mark-to-Market стоимость позиции внутри месяца.
        Сумма внутренней стоимости (если упали) и остатка временной стоимости.
        """
        days_total = 30 # Упрощение для распада
        days_left = (self.expiry - current_date).days
        time_decay_factor = max(days_left / days_total, 0)
        
        # 1. Текущая внутренняя стоимость (Intrinsic)
        intrinsic_total = self.get_total_payoff(current_spot)
        
        # 2. Оставшаяся премия (Time Value)
        # Линейный распад уплаченной премии
        remaining_premium = self.cost_total * time_decay_factor
        
        # На рынке опцион стоит либо Intrinsic (глубоко ITM), либо TimeValue (OTM), либо Mix.
        # Для консервативности берем максимум.
        return max(intrinsic_total, remaining_premium)


class ParametricATMCall:
    """ 
    Моделирует покупку ATM Call опциона (Long Call).
    """
    def __init__(self, spot_price, expiry_date, implied_vol, quantity):
        self.strike = spot_price
        self.expiry = expiry_date
        self.iv = implied_vol
        self.qty = quantity
        
        # Расчет стоимости 1 опциона через аппроксимацию Бреннера-Субрахманьяма
        # Price ~ 0.4 * Spot * Vol * sqrt(Time)
        time_years = 30 / 365.0
        self.unit_cost = 0.4 * spot_price * self.iv * np.sqrt(time_years)
        
        self.total_cost = self.unit_cost * self.qty

    def get_total_payoff(self, current_spot):
        """ Выплата при экспирации """
        unit_payoff = max(current_spot - self.strike, 0)
        return unit_payoff * self.qty

    def get_mtm_value(self, current_spot, current_date):
        """ Текущая рыночная оценка """
        # Intrinsic
        unit_intrinsic = max(current_spot - self.strike, 0)
        
        # Time Value (sqrt decay approximation)
        total_life = 30
        days_left = (self.expiry - current_date).days
        time_fraction = max(days_left / total_life, 0)
        
        # Оценка текущей временной стоимости
        current_time_val_unit = self.unit_cost * np.sqrt(time_fraction)
        
        unit_price = unit_intrinsic + current_time_val_unit
        return unit_price * self.qty