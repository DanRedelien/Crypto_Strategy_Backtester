"""
Backtest Integrity Tests.

Tests for:
1. Lookahead bias detection
2. Fee/slippage correctness
3. Option payoff math
4. CVaR calculation validity
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.btc_backtest.settings import Settings
from src.btc_backtest.engine import BacktestEngine
from src.btc_backtest.options import ParametricPutSpread, ParametricATMCall
from src.btc_backtest.analytics import Analytics


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_settings() -> Settings:
    """Create test settings with known values."""
    return Settings(
        SYMBOL="BTCUSDT",
        INITIAL_CAPITAL=10_000.0,
        MONTHLY_DEPOSIT=0.0,  # Disable DCA for simpler tests
        FEE_RATE=0.001,       # 0.1% fee
        HEDGE_COST_PCT=0.03,
        HEDGE_STRIKE_LONG_PCT=0.95,
        HEDGE_STRIKE_SHORT_PCT=0.80,
    )


@pytest.fixture
def simple_price_data() -> pd.DataFrame:
    """Create simple price data for testing."""
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    # Constant price - makes calculations deterministic
    prices = [100_000.0] * 100
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1000.0] * 100,
    })


@pytest.fixture
def crash_price_data() -> pd.DataFrame:
    """Create price data with a crash for testing hedge payoff."""
    # Need enough data for hedge to expire (30 days = 720 hours)
    # Create 800 hourly candles to ensure hedge expires
    dates = pd.date_range("2025-01-01", periods=800, freq="h")
    # Price holds at 100k for first 400 hours, then crashes to 70k
    prices = [100_000.0] * 400 + [70_000.0] * 400
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1000.0] * 800,
    })


# =============================================================================
# TEST: LOOKAHEAD BIAS
# =============================================================================

class TestLookaheadBias:
    """
    Tests to ensure no future data is used in decision making.
    
    Methodology:
    - Run backtest on data[0:N]
    - Re-run on data[0:N+1]
    - Equity at time N should be IDENTICAL (no retroactive changes)
    """
    
    def test_equity_deterministic_no_future_data(self, sample_settings, simple_price_data):
        """
        Core lookahead test: Adding future data should NOT change past equity.
        """
        # Run on subset (first 50 rows)
        subset_data = simple_price_data.iloc[:50].copy()
        engine1 = BacktestEngine(sample_settings, subset_data)
        result1 = engine1.run()
        
        # Run on full data (100 rows)
        full_data = simple_price_data.copy()
        engine2 = BacktestEngine(sample_settings, full_data)
        result2 = engine2.run()
        
        # Equity at timestamp 50 should be IDENTICAL
        # (past decisions should not be influenced by future data)
        eq1_at_50 = result1['bench_equity'].iloc[-1]
        eq2_at_50 = result2['bench_equity'].iloc[49]
        
        assert abs(eq1_at_50 - eq2_at_50) < 0.01, \
            f"Lookahead bias detected! Equity differs: {eq1_at_50} vs {eq2_at_50}"
    
    def test_hedge_decision_uses_only_current_price(self, sample_settings, crash_price_data):
        """
        Hedge is opened using CURRENT price only, not future prices.
        """
        engine = BacktestEngine(sample_settings, crash_price_data)
        
        # Before running, verify hedge is opened on first iteration
        # The hedge strike should be based on FIRST price (100k), not crash price (70k)
        result = engine.run()
        
        # Check that hedge was created with entry_price = 100k
        # (We can't directly access active_hedge after run, but we can verify
        # via the hedge_stats that payoff occurred with correct strikes)
        
        # If lookahead existed, hedge would be opened at 70k (post-crash)
        # which would give 0 payoff. Correct behavior gives payoff > 0.
        assert engine.final_stats['hedge_payoff'] > 0, \
            "Hedge should have paid off during crash if no lookahead bias"


# =============================================================================
# TEST: FEE AND SLIPPAGE
# =============================================================================

class TestFeesAndSlippage:
    """
    Tests to verify fees are applied correctly per MATH_SPEC.md §4.
    """
    
    def test_buy_fee_deducted(self, sample_settings, simple_price_data):
        """
        When buying BTC, fee should be deducted from the purchase amount.
        
        With $10,000 and 0.1% fee at $100,000/BTC:
        - Gross: $10,000
        - Fee: $10 (0.1%)
        - Net for purchase: $9,990
        - BTC received: 9,990 / 100,000 = 0.0999 BTC
        """
        engine = BacktestEngine(sample_settings, simple_price_data)
        
        # Initial buy happens before loop
        start_price = 100_000.0
        expected_btc = (10_000.0 * (1 - 0.001)) / start_price  # 0.0999 BTC
        
        engine._buy_spot_bench(start_price)
        
        assert abs(engine.bench_btc - expected_btc) < 1e-10, \
            f"Buy fee not applied correctly: got {engine.bench_btc}, expected {expected_btc}"
        assert engine.bench_cash == 0, "Cash should be zero after full buy"
    
    def test_sell_fee_deducted(self, sample_settings, simple_price_data):
        """
        When selling BTC to get cash, fee is deducted from proceeds.
        
        To receive NET $300, at 0.1% fee:
        - Need to sell GROSS = 300 / (1 - 0.001) = 300.3003 worth of BTC
        - BTC to sell = 300.3003 / price
        """
        settings = sample_settings
        settings.HEDGE_COST_PCT = 0.05  # 5% to force selling
        
        engine = BacktestEngine(settings, simple_price_data)
        
        # Do initial buy
        price = 100_000.0
        engine._buy_spot_hedged(price)
        
        # Now hedged_btc = 10000 * 0.999 / 100000 = 0.0999
        initial_btc = engine.hedged_btc
        
        # Process hedge - will need to sell BTC to pay premium
        date = simple_price_data.iloc[0]['timestamp']
        engine._process_hedge_strategy(date, price)
        
        # Hedge cost = equity * 0.05 = 9990 * 0.05 = 499.5
        # Since hedged_cash = 0, we need to sell BTC for 499.5
        # BTC sold = 499.5 / (100000 * 0.999) = 0.005
        
        btc_sold = initial_btc - engine.hedged_btc
        deficit = 9990 * 0.05  # hedge budget
        expected_btc_sold = deficit / (price * (1 - 0.001))
        
        assert abs(btc_sold - expected_btc_sold) < 1e-10, \
            f"Sell fee not applied correctly: sold {btc_sold}, expected {expected_btc_sold}"
    
    def test_fee_impact_on_final_equity(self, sample_settings, simple_price_data):
        """
        With constant price, equity should be LESS than invested due to fees.
        """
        engine = BacktestEngine(sample_settings, simple_price_data)
        result = engine.run()
        
        initial = sample_settings.INITIAL_CAPITAL
        final_bench = result['bench_equity'].iloc[-1]
        
        # With 0.1% fee on initial buy, equity = initial * (1 - 0.001) = 9990
        expected = initial * (1 - 0.001)
        
        assert abs(final_bench - expected) < 0.01, \
            f"Final equity should reflect fee drag: got {final_bench}, expected {expected}"


# =============================================================================
# TEST: PUT SPREAD PAYOFF MATH
# =============================================================================

class TestPutSpreadPayoff:
    """
    Tests for put spread option payoff calculation.
    """
    
    def test_payoff_below_long_strike(self):
        """
        When price < strike_long, payoff = (strike_long - price) * qty
        capped at (strike_long - strike_short) * qty
        """
        spread = ParametricPutSpread(
            spot_price=100_000,
            expiry_date=datetime.now() + timedelta(days=30),
            total_premium_paid=1000,
            quantity=1.0,
            strike_long_pct=0.95,   # Strike = 95,000
            strike_short_pct=0.80,  # Strike = 80,000
        )
        
        # Price at 90,000 (below long strike 95k, above short strike 80k)
        payoff = spread.get_total_payoff(90_000)
        expected = (95_000 - 90_000) * 1.0  # = 5,000
        assert abs(payoff - expected) < 0.01, f"Payoff wrong: {payoff} vs {expected}"
    
    def test_payoff_at_max_protection(self):
        """
        When price < strike_short, payoff is capped at max spread width.
        """
        spread = ParametricPutSpread(
            spot_price=100_000,
            expiry_date=datetime.now() + timedelta(days=30),
            total_premium_paid=1000,
            quantity=1.0,
            strike_long_pct=0.95,   # Strike = 95,000
            strike_short_pct=0.80,  # Strike = 80,000
        )
        
        # Price at 70,000 (below both strikes)
        payoff = spread.get_total_payoff(70_000)
        # Long put = 95k - 70k = 25k, Short put = 80k - 70k = 10k
        # Net = 25k - 10k = 15k (max spread width)
        expected = (95_000 - 80_000) * 1.0  # = 15,000
        assert abs(payoff - expected) < 0.01, f"Payoff wrong: {payoff} vs {expected}"
    
    def test_payoff_above_long_strike(self):
        """
        When price > strike_long, payoff = 0 (OTM).
        """
        spread = ParametricPutSpread(
            spot_price=100_000,
            expiry_date=datetime.now() + timedelta(days=30),
            total_premium_paid=1000,
            quantity=1.0,
            strike_long_pct=0.95,
            strike_short_pct=0.80,
        )
        
        # Price at 100,000 (above both strikes)
        payoff = spread.get_total_payoff(100_000)
        assert payoff == 0, f"OTM payoff should be 0, got {payoff}"


# =============================================================================
# TEST: CVaR CALCULATION
# =============================================================================

class TestCVaR:
    """
    Tests for CVaR (Expected Shortfall) calculation.
    """
    
    def test_cvar_basic_calculation(self):
        """
        CVaR = mean of returns below VaR threshold.
        
        For returns [-0.10, -0.05, -0.02, 0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
        5th percentile (VaR_95) = -0.10
        Tail returns = [-0.10] (only values <= -0.10)
        CVaR = mean([-0.10]) = -0.10
        """
        analytics = Analytics()
        returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15])
        
        cvar = analytics._calculate_cvar(returns, confidence=0.95)
        
        # 5% of 10 = 0.5, so VaR threshold is the 5th percentile
        # quantile(0.05) of this series = -0.10
        # Only -0.10 is <= -0.10
        # CVaR = -0.10
        assert abs(cvar - (-0.10)) < 0.01, f"CVaR wrong: {cvar}"
    
    def test_cvar_with_multiple_tail_values(self):
        """
        CVaR with multiple values in the tail.
        """
        analytics = Analytics()
        # 20 values, 5% tail = 1 value at 95% confidence
        # But let's use 90% confidence = 10% tail = 2 values
        returns = pd.Series([
            -0.20, -0.15, -0.10, -0.08, -0.05, 
            -0.02, 0.00, 0.02, 0.05, 0.08,
            0.10, 0.12, 0.15, 0.18, 0.20,
            0.22, 0.25, 0.28, 0.30, 0.35
        ])
        
        cvar = analytics._calculate_cvar(returns, confidence=0.90)
        
        # 10% tail of 20 values = 2 values
        # VaR threshold = quantile(0.10) = -0.15 (approximately)
        # Tail = [-0.20, -0.15]
        # CVaR = mean = -0.175
        var_threshold = returns.quantile(0.10)
        tail = returns[returns <= var_threshold]
        expected_cvar = tail.mean()
        
        assert abs(cvar - expected_cvar) < 0.01, f"CVaR wrong: {cvar} vs {expected_cvar}"
    
    def test_cvar_worse_than_var(self):
        """
        CVaR should always be worse (more negative) than VaR.
        """
        analytics = Analytics()
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.02)  # 2% daily vol
        
        var_95 = returns.quantile(0.05)
        cvar_95 = analytics._calculate_cvar(returns, confidence=0.95)
        
        assert cvar_95 <= var_95, \
            f"CVaR ({cvar_95}) should be <= VaR ({var_95})"


# =============================================================================
# TEST: MATH SPEC COMPLIANCE
# =============================================================================

class TestMathSpecCompliance:
    """
    Tests to verify compliance with MATH_SPEC.md requirements.
    """
    
    def test_annualization_factor_crypto(self, sample_settings):
        """
        Per MATH_SPEC.md §2: Crypto uses 365 days.
        """
        assert sample_settings.ANNUALIZATION_FACTOR == 365, \
            "Crypto annualization should be 365 days"
    
    def test_simple_returns_for_portfolio(self, sample_settings, simple_price_data):
        """
        Per MATH_SPEC.md §1B: Portfolio returns should use simple returns.
        
        The engine correctly calculates equity = cash + btc * price,
        which is equivalent to simple return aggregation.
        """
        engine = BacktestEngine(sample_settings, simple_price_data)
        result = engine.run()
        
        # Equity calculation: cash + btc * price
        # This is correct for portfolio value (not log returns)
        bench_eq = result['bench_equity'].iloc[-1]
        
        # Manual calculation
        expected_btc = 10_000 * (1 - 0.001) / 100_000  # After fee
        expected_eq = expected_btc * 100_000
        
        assert abs(bench_eq - expected_eq) < 0.01, \
            f"Portfolio equity calculation incorrect: {bench_eq} vs {expected_eq}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
