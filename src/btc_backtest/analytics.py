"""
Analytics module for backtest results processing and visualization.
Computes risk metrics, returns, and generates charts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


class Analytics:
    """
    Analytics engine for backtest results.
    Computes returns, risk metrics, and generates visualizations.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        plt.style.use('seaborn-v0_8-whitegrid')
        self.rf = risk_free_rate
        self.colors = {
            'bench': '#95a5a6',   # Grey (Benchmark)
            'hedged': '#2c3e50',  # Dark Blue (Hedged)
            'call': '#e67e22',    # Orange (Call)
            'collar': '#8E44AD'   # Purple (Collar)
        }
    
    def xirr(self, transactions: List[Tuple[datetime, float]]) -> float:
        """
        Calculate XIRR (Extended Internal Rate of Return) for irregular flows.
        
        Args:
            transactions: List of tuples (date, amount).
                amount < 0 for deposits (outflows from pocket),
                > 0 for withdrawals/terminal value.
        
        Returns:
            Annualized IRR as decimal.
        """
        if not transactions:
            return 0.0
        
        dates = [t[0] for t in transactions]
        amounts = [t[1] for t in transactions]
        
        start_date = min(dates)
        years = np.array([(d - start_date).days / 365.0 for d in dates])
        amounts = np.array(amounts)
        
        def npv(rate: float) -> float:
            if rate <= -0.99:
                return float('inf')
            return np.sum(amounts / ((1 + rate) ** years))
        
        try:
            rate = 0.1
            for _ in range(50):
                f_val = npv(rate)
                if abs(f_val) < 1e-6:
                    return rate
                
                # Derivative of NPV with respect to rate
                df_val = np.sum(-years * amounts * ((1 + rate) ** (-years - 1)))
                if df_val == 0:
                    break
                
                new_rate = rate - f_val / df_val
                if abs(new_rate - rate) < 1e-6:
                    return new_rate
                rate = new_rate
            return rate
        except Exception:
            return 0.0
    
    def _get_drawdown_stats(
        self, nav_series: pd.Series
    ) -> Tuple[float, float, int, Any]:
        """
        Compute drawdown metrics on NAV series.
        
        Returns:
            Tuple of:
            1. MaxDD (float) - Maximum peak-to-trough decline
            2. Ulcer Index (float) - RMS of drawdowns (stress measure)
            3. Max Recovery Days (int) - Longest peak→trough→recovery cycle
            4. Trough Date (timestamp of max drawdown)
        """
        # Skip NaN values at the start (pre-exposure period)
        nav_clean = nav_series.dropna()
        if len(nav_clean) < 2:
            return 0.0, 0.0, 0, nav_series.index[0]
        
        cummax = nav_clean.cummax()
        dd_series = (nav_clean - cummax) / cummax
        
        max_dd = dd_series.min()
        ulcer = np.sqrt(np.mean(dd_series ** 2)) * 100
        
        trough_idx = dd_series.idxmin()
        
        # Recovery: Peak → Trough → Full Recovery
        in_drawdown = False
        cycle_start = None
        max_recovery_days = 0
        
        for i, (dt, nav_val) in enumerate(nav_clean.items()):
            peak_val = cummax.iloc[i]
            
            if not in_drawdown:
                if nav_val < peak_val:
                    in_drawdown = True
                    cycle_start = dt
            else:
                if nav_val >= peak_val:
                    in_drawdown = False
                    if cycle_start is not None:
                        recovery_days = (dt - cycle_start).days
                        max_recovery_days = max(max_recovery_days, recovery_days)
                    cycle_start = None
        
        # Check if still in drawdown at end
        if in_drawdown and cycle_start is not None:
            ongoing_days = (nav_clean.index[-1] - cycle_start).days
            max_recovery_days = max(max_recovery_days, ongoing_days)
        
        return max_dd, ulcer, max_recovery_days, trough_idx
    
    def _calculate_cvar(
        self, 
        returns: pd.Series, 
        confidence: float = 0.95
    ) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        
        Methodology: Mean of returns below the VaR threshold.
        CVaR_α = E[R | R < VaR_α] where VaR_α = (1-α) percentile of returns.
        
        Per MATH_SPEC.md: This measures the expected loss in the worst (1-α)% of cases.
        
        Args:
            returns: Daily returns series (log or simple).
            confidence: Confidence level (0.95 = 5% tail).
            
        Returns:
            CVaR as a negative decimal (e.g., -0.05 = 5% expected loss in tail).
        """
        if len(returns) < 10:
            return 0.0
        
        var_threshold = returns.quantile(1 - confidence)
        tail_losses = returns[returns <= var_threshold]
        
        return tail_losses.mean() if len(tail_losses) > 0 else 0.0
    
    def _compute_nav_series(self, df: pd.DataFrame, strat: str) -> pd.Series:
        """
        Calculate NAV (Net Asset Value) series for accurate risk measurement.
        
        TWR methodology:
        - Daily returns computed from PnL only (excluding external flows)
        - NAV starts from first valid exposure
        - Extreme returns are preserved (not clipped) to capture tail risk
        - Zero/NaN periods are marked as NaN (not filled with fictitious values)
        
        Args:
            df: DataFrame with equity and deposit columns.
            strat: Strategy name ('bench', 'hedged', 'call').
            
        Returns:
            NAV series starting at 1.0.
        """
        col_eq = f'{strat}_equity'
        equity = df[col_eq].copy()
        deposits = df['deposit_flow'].copy()
        
        # Pre-deposit equity = current equity - today's deposit
        equity_pre_deposit = equity - deposits
        
        # Daily PnL = equity_pre_deposit[t] - equity[t-1]
        equity_shifted = equity.shift(1)
        pnl = equity_pre_deposit - equity_shifted
        
        # Daily return = PnL / prior equity
        daily_return = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i == 0:
                daily_return.iloc[i] = 0.0  # NAV starts at 1.0
            else:
                prior_eq = equity_shifted.iloc[i]
                if prior_eq > 0:
                    ret = pnl.iloc[i] / prior_eq
                    daily_return.iloc[i] = ret
                    
                    # Log extreme returns (track tail events)
                    if abs(ret) > 0.5:  # >50% daily move
                        print(f"  [TAIL EVENT] {strat} @ {df.index[i].date()}: {ret:.1%} daily return")
                else:
                    daily_return.iloc[i] = 0.0  # No exposure yet
        
        # NAV = cumulative product of (1 + daily_return)
        nav = (1 + daily_return).cumprod()
        
        return nav
    
    def process_results(
        self, 
        df_results: pd.DataFrame, 
        hedge_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process backtest results and compute all metrics.
        
        Args:
            df_results: Raw backtest results DataFrame.
            hedge_stats: Optional dict with 'hedge_cost' and 'hedge_payoff'.
            
        Returns:
            Tuple of (processed DataFrame, metrics dict).
        """
        df = df_results.copy()
        df.set_index('timestamp', inplace=True)
        
        metrics: Dict[str, Any] = {}
        strategies = ['bench', 'hedged', 'call', 'collar']
        
        # --- 1. CASH FLOWS PREP ---
        df['flow'] = df['invested_total'].diff().fillna(0)
        df['flow'].iloc[0] = df['invested_total'].iloc[0]
        
        deposit_schedule = []
        for dt, flow in df['flow'].items():
            if flow > 0:
                deposit_schedule.append((dt, -flow))  # Negative for XIRR (Investment)
        
        # --- 2. MARKET CONTEXT (BENCHMARK STRESS) ---
        bench_cummax = df['bench_equity'].cummax()
        bench_dd = (df['bench_equity'] - bench_cummax) / bench_cummax
        market_stress_date = bench_dd.idxmin()
        
        invested_at_stress = df.loc[market_stress_date, 'invested_total']
        
        metrics['global'] = {
            'stress_date': market_stress_date,
            'bench_at_stress': df.loc[market_stress_date, 'bench_equity'],
            'invested_at_stress': invested_at_stress
        }
        
        # --- 3. COMPUTE NAV SERIES (for risk metrics) ---
        for strat in strategies:
            df[f'{strat}_nav'] = self._compute_nav_series(df, strat)
        
        # --- 4. STRATEGY METRICS ---
        for strat in strategies:
            col_eq = f'{strat}_equity'
            col_nav = f'{strat}_nav'
            
            # A. XIRR (uses investor equity with deposits - correct for MWR)
            flows = deposit_schedule.copy()
            flows.append((df.index[-1], df[col_eq].iloc[-1]))
            strat_xirr = self.xirr(flows)
            
            # B. Risk Metrics (uses NAV - removes deposit distortion)
            mdd, ulcer, recovery_days, strat_trough_date = self._get_drawdown_stats(df[col_nav])
            
            # C. CVaR (Expected Shortfall) at 95%
            # Calculate daily returns from NAV for CVaR
            nav_returns = df[col_nav].pct_change().dropna()
            cvar_95 = self._calculate_cvar(nav_returns, confidence=0.95)
            
            # D. Stress Test Performance
            equity_at_market_stress = df.loc[market_stress_date, col_eq]
            
            metrics[strat] = {
                'XIRR': strat_xirr,
                'MaxDD': mdd,
                'Ulcer': ulcer,
                'CVaR_95': cvar_95,
                'Recovery_Days': recovery_days,
                'Terminal': df[col_eq].iloc[-1],
                'Invested': df['invested_total'].iloc[-1],
                'Profit': df[col_eq].iloc[-1] - df['invested_total'].iloc[-1],
                'Equity_at_Market_Stress': equity_at_market_stress,
                'Invested_at_Market_Stress': invested_at_stress
            }
        
        # --- 5. ATTRIBUTION & COUNTERFACTUALS ---
        if hedge_stats:
            h_metrics = metrics['hedged']
            b_metrics = metrics['bench']
            
            cost = hedge_stats['hedge_cost']
            payoff = hedge_stats['hedge_payoff']
            net_hedge_pnl = payoff - cost
            
            # Hedge Efficiency (payoff / cost)
            hedge_efficiency = (payoff / cost) if cost > 0 else 0
            
            # Peak NAV before crash (hedged strategy)
            hedged_nav = df['hedged_nav']
            peak_nav_value = hedged_nav.loc[:market_stress_date].max()
            
            # Crisis Coverage: payoff as % of peak exposure
            crisis_coverage = (payoff / (peak_nav_value * df['invested_total'].iloc[-1])) if peak_nav_value > 0 else 0
            
            # Net cost ratio
            net_cost_ratio = (cost / h_metrics['Terminal']) if h_metrics['Terminal'] > 0 else 0
            
            # ATTRIBUTION: Counterfactual Analysis
            total_alpha = h_metrics['Profit'] - b_metrics['Profit']
            
            stress_price = df.loc[market_stress_date, 'price']
            end_price = df['price'].iloc[-1]
            market_recovery_factor = (end_price / stress_price) - 1.0 if stress_price > 0 else 0
            
            # Max Reinvestment Effect (Upper Bound)
            max_reinvestment_effect = payoff * market_recovery_factor
            
            # Structural / Convexity Alpha
            structural_alpha = total_alpha - net_hedge_pnl - max_reinvestment_effect
            
            metrics['attribution'] = {
                # Hedge (Put Spread)
                'hedge_cost': cost,
                'hedge_payoff': payoff,
                'Net_Hedge_PnL': net_hedge_pnl,
                'Hedge_Efficiency': hedge_efficiency,
                'Crisis_Coverage': crisis_coverage,
                'Net_Cost_Ratio': net_cost_ratio,
                'Total_Alpha': total_alpha,
                'Reinvestment_Effect': max_reinvestment_effect,
                'Residual_Path_Alpha': structural_alpha,
                # Call Strategy
                'call_premium': hedge_stats.get('call_premium', 0),
                'call_payoff': hedge_stats.get('call_payoff', 0),
                # Collar Strategy
                # premium_received: positive = net credit (we receive), negative = net debit (we pay)
                'collar_premium_received': hedge_stats.get('collar_premium_received', 0),
                'collar_payoff': hedge_stats.get('collar_payoff', 0)
            }
        
        return df, metrics
    
    def print_stats(self, metrics: Dict[str, Any]) -> None:
        """Print formatted statistics table to terminal."""
        print("\n" + "=" * 115)
        print(f"{'METRIC':<30} | {'BENCH':<12} | {'HEDGED':<12} | {'CALL':<12} | {'COLLAR':<12}")
        print("-" * 115)
        
        # Helpers
        def p(val: float) -> str:
            return f"{val:.2%}"
        
        def m(val: float) -> str:
            return f"${val:,.0f}"
        
        def f(val: float) -> str:
            return f"{val:.2f}"
        
        row = "{:<30} | {:<12} | {:<12} | {:<12} | {:<12}"
        
        print(row.format(
            "MWR / XIRR (Investor Yield)",
            p(metrics['bench']['XIRR']),
            p(metrics['hedged']['XIRR']),
            p(metrics['call']['XIRR']),
            p(metrics['collar']['XIRR'])
        ))
        print(row.format(
            "Total Net Profit",
            m(metrics['bench']['Profit']),
            m(metrics['hedged']['Profit']),
            m(metrics['call']['Profit']),
            m(metrics['collar']['Profit'])
        ))
        print("-" * 115)
        
        print(row.format(
            "Max Drawdown",
            p(metrics['bench']['MaxDD']),
            p(metrics['hedged']['MaxDD']),
            p(metrics['call']['MaxDD']),
            p(metrics['collar']['MaxDD'])
        ))
        print(row.format(
            "Ulcer Index (Stress Load)",
            f(metrics['bench']['Ulcer']),
            f(metrics['hedged']['Ulcer']),
            f(metrics['call']['Ulcer']),
            f(metrics['collar']['Ulcer'])
        ))
        print(row.format(
            "CVaR 95% (Expected Shortfall)",
            p(metrics['bench']['CVaR_95']),
            p(metrics['hedged']['CVaR_95']),
            p(metrics['call']['CVaR_95']),
            p(metrics['collar']['CVaR_95'])
        ))
        print("-" * 50 + " STRESS TEST " + "-" * 50)
        
        print(row.format(
            "Equity at Market Bottom",
            m(metrics['bench']['Equity_at_Market_Stress']),
            m(metrics['hedged']['Equity_at_Market_Stress']),
            m(metrics['call']['Equity_at_Market_Stress']),
            m(metrics['collar']['Equity_at_Market_Stress'])
        ))
        print(row.format(
            "Invested Cap at Market Bottom",
            m(metrics['bench']['Invested_at_Market_Stress']),
            m(metrics['hedged']['Invested_at_Market_Stress']),
            m(metrics['call']['Invested_at_Market_Stress']),
            m(metrics['collar']['Invested_at_Market_Stress'])
        ))
        
        # 3. STRATEGY COMPARISON (Insurance Mechanics)
        if 'attribution' in metrics:
            attr = metrics['attribution']
            print("\n" + "=" * 115)
            print(">>> STRATEGY COMPARISON (Insurance Mechanics)")
            print("-" * 115)
            print(f"{'METRIC':<30} | {'BENCH':<12} | {'HEDGED':<12} | {'CALL':<12} | {'COLLAR':<12}")
            print("-" * 115)
            
            # 1. Premium/Cost Section
            # Collar: positive premium_received = credit, negative = debit
            collar_premium = attr['collar_premium_received']
            collar_payoff = attr['collar_payoff']
            
            # For display: show actual cash flow (negative = paid, positive = received)
            print(row.format(
                "Premium Paid (Cost)",
                m(0),
                m(attr['hedge_cost']),
                m(attr['call_premium']),
                m(-collar_premium) if collar_premium > 0 else m(abs(collar_premium))
            ))
            print(row.format(
                "Premium Received (Credit)",
                m(0),
                m(0),
                m(0),
                m(collar_premium) if collar_premium > 0 else m(0)
            ))
            print(row.format(
                "Total Payoff Received",
                m(0),
                m(attr['hedge_payoff']),
                m(attr['call_payoff']),
                m(collar_payoff)
            ))
            # Net PnL = payoff + premium_received - cost
            # For collar: Net = payoff + premium_received (if credit) or payoff - abs(premium) (if debit)
            collar_net_pnl = collar_payoff + collar_premium
            print(row.format(
                "Net Insurance PnL",
                m(0),
                m(attr['hedge_payoff'] - attr['hedge_cost']),
                m(attr['call_payoff'] - attr['call_premium']),
                m(collar_net_pnl)
            ))
            print("-" * 115)
            
            # 2. Efficiency Section
            # Collar efficiency: (payoff + premium) / abs(cost if debit)
            if collar_premium > 0:
                collar_efficiency_str = f"+{collar_premium:,.0f}$"
            elif collar_premium < 0:
                collar_efficiency_str = p((collar_payoff + collar_premium) / abs(collar_premium)) if collar_premium != 0 else "N/A"
            else:
                collar_efficiency_str = "N/A"
            
            print(row.format(
                "Hedge Efficiency (Payoff/Cost)",
                "N/A",
                p(attr['Hedge_Efficiency']),
                p(attr['call_payoff'] / attr['call_premium']) if attr['call_premium'] > 0 else "N/A",
                collar_efficiency_str
            ))
            
            # Cost ratio (only for strategies that pay premium)
            collar_cost_ratio = "N/A" if collar_premium >= 0 else p(abs(collar_premium) / metrics['collar']['Terminal'])
            print(row.format(
                "Cost as % of Terminal Equity",
                "0.00%",
                p(attr['Net_Cost_Ratio']),
                p(attr['call_premium'] / metrics['call']['Terminal']) if metrics['call']['Terminal'] > 0 else "N/A",
                collar_cost_ratio
            ))
            print("-" * 115)
            
            # 3. Alpha Attribution
            print(row.format(
                "Total Alpha vs Benchmark",
                m(0),
                m(metrics['hedged']['Profit'] - metrics['bench']['Profit']),
                m(metrics['call']['Profit'] - metrics['bench']['Profit']),
                m(metrics['collar']['Profit'] - metrics['bench']['Profit'])
            ))
        
        print("=" * 115)
        
        # 4. STRATEGY DESCRIPTIONS
        print("\n>>> STRATEGY DESCRIPTIONS")
        print("-" * 80)
        print("BENCH:  Pure DCA spot buy. No hedging. Full upside, full downside.")
        print("HEDGED: Spot + Put Spread (95%/80%). Monthly premium for crash protection.")
        print("CALL:   Treasuries + ATM Calls. Synthetic BTC exposure, cash earns interest.")
        print("COLLAR: Spot + Collar (Sell 105% Call, Buy 95% Put, Sell 70% Put).")
        print("        Net credit structure: capped upside, limited downside protection.")
        print("=" * 80 + "\n")
    
    def plot_charts(self, df: pd.DataFrame) -> None:
        """Generate visualization charts."""
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # 1. Equity Curve
        ax1 = plt.subplot(gs[0])
        for strat, label in zip(
            ['bench', 'hedged', 'call', 'collar'], 
            ['DCA Spot', 'Put Spread', 'ATM Call', 'Collar']
        ):
            ax1.plot(df.index, df[f'{strat}_equity'], label=label, color=self.colors[strat], linewidth=2)
        
        ax1.plot(df.index, df['invested_total'], label='Invested Capital', color='black', linestyle='--', alpha=0.5)
        
        # Mark the Stress Point
        bench_max = df['bench_equity'].cummax()
        bench_dd = (df['bench_equity'] - bench_max) / bench_max
        stress_date = bench_dd.idxmin()
        ax1.axvline(stress_date, color='red', linestyle=':', alpha=0.5, label='Market Stress Point')
        
        ax1.set_title('Portfolio Value (USD)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('USD')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdowns (NAV-based - removes deposit distortion)
        ax2 = plt.subplot(gs[1], sharex=ax1)
        for strat in ['bench', 'hedged', 'call', 'collar']:
            nav = df[f'{strat}_nav']
            peak = nav.cummax()
            dd = (nav - peak) / peak.replace(0, 1)
            ax2.plot(df.index, dd, color=self.colors[strat], linewidth=1)
            ax2.fill_between(df.index, dd, 0, color=self.colors[strat], alpha=0.1)
        
        ax2.set_ylabel('Drawdown %')
        ax2.set_title('Risk Profile (NAV-Based, Deposit-Adjusted)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Strategy Alpha (NAV-based - clean comparison)
        ax3 = plt.subplot(gs[2], sharex=ax1)
        
        nav_ratio = df['hedged_nav'] / df['bench_nav'].replace(0, 1)
        ax3.plot(df.index, nav_ratio, color=self.colors['hedged'], label='Hedged / Bench NAV Ratio')
        ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        
        # Highlight alpha generation
        ax3.fill_between(df.index, nav_ratio, 1.0, where=(nav_ratio > 1.0), color=self.colors['hedged'], alpha=0.2)
        ax3.fill_between(df.index, nav_ratio, 1.0, where=(nav_ratio < 1.0), color='red', alpha=0.1)
        
        ax3.set_ylabel('NAV Alpha Ratio')
        ax3.set_title('Strategy Alpha (NAV-Based, Deposit-Neutral)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
