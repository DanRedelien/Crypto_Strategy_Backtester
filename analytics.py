import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

class InstitutionalAnalytics:
    def __init__(self, risk_free_rate=0.04):
        plt.style.use('seaborn-v0_8-whitegrid')
        self.rf = risk_free_rate
        self.colors = {
            'bench': '#95a5a6',   # Grey (Benchmark)
            'hedged': '#2c3e50',  # Dark Blue (Institutional)
            'call': '#e67e22'     # Orange (Aggressive)
        }

    def xirr(self, transactions):
        """
        Calculates XIRR (Extended Internal Rate of Return) for irregular flows.
        transactions: list of tuples (date, amount).
        amount < 0 for deposits (outflows from pocket), > 0 for withdrawals/terminal value.
        """
        if not transactions:
            return 0.0

        dates = [t[0] for t in transactions]
        amounts = [t[1] for t in transactions]
        
        start_date = min(dates)
        years = np.array([(d - start_date).days / 365.0 for d in dates])
        amounts = np.array(amounts)

        def npv(rate):
            if rate <= -0.99: return float('inf')
            # Vectorized NPV calculation
            return np.sum(amounts / ((1 + rate) ** years))

        try:
            rate = 0.1
            for _ in range(50):
                f_val = npv(rate)
                if abs(f_val) < 1e-6: return rate
                
                # Derivative of NPV with respect to rate
                df_val = np.sum(-years * amounts * ((1 + rate) ** (-years - 1)))
                if df_val == 0: break
                
                new_rate = rate - f_val / df_val
                if abs(new_rate - rate) < 1e-6: return new_rate
                rate = new_rate
            return rate
        except:
            return 0.0

    def _get_drawdown_stats(self, nav_series):
        """
        Institutional-grade drawdown metrics on NAV series.
        
        Returns:
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
        
        # Institutional Recovery: Peak → Trough → Full Recovery
        # Find all drawdown cycles properly
        in_drawdown = False
        cycle_start = None
        max_recovery_days = 0
        
        for i, (dt, nav_val) in enumerate(nav_clean.items()):
            peak_val = cummax.iloc[i]
            
            if not in_drawdown:
                # Check if we just entered drawdown
                if nav_val < peak_val:
                    in_drawdown = True
                    cycle_start = dt
            else:
                # Check if we recovered (new peak)
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

    def _compute_nav_series(self, df, strat):
        """
        Calculates NAV (Net Asset Value) series for accurate risk measurement.
        
        Institutional TWR methodology:
        - Daily returns computed from PnL only (excluding external flows)
        - NAV starts from first valid exposure
        - Extreme returns are preserved (not clipped) to capture tail risk
        - Zero/NaN periods are marked as NaN (not filled with fictitious values)
        
        WARNING: Extreme daily returns are logged for review.
        """
        col_eq = f'{strat}_equity'
        equity = df[col_eq].copy()
        deposits = df['deposit_flow'].copy()
        
        # Pre-deposit equity = current equity - today's deposit
        equity_pre_deposit = equity - deposits
        
        # Daily PnL = equity_pre_deposit[t] - equity[t-1]
        equity_shifted = equity.shift(1)
        pnl = equity_pre_deposit - equity_shifted
        
        # First period: no return (we start tracking from here)
        # Daily return = PnL / prior equity
        # NaN for zero/invalid prior equity (no fictitious capital)
        daily_return = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i == 0:
                daily_return.iloc[i] = 0.0  # NAV starts at 1.0
            else:
                prior_eq = equity_shifted.iloc[i]
                if prior_eq > 0:
                    ret = pnl.iloc[i] / prior_eq
                    daily_return.iloc[i] = ret
                    
                    # Log extreme returns (institutional: track tail events)
                    if abs(ret) > 0.5:  # >50% daily move
                        print(f"  [TAIL EVENT] {strat} @ {df.index[i].date()}: {ret:.1%} daily return")
                else:
                    daily_return.iloc[i] = 0.0  # No exposure yet
        
        # NAV = cumulative product of (1 + daily_return)
        # Extreme returns preserved - they are real risk
        nav = (1 + daily_return).cumprod()
        
        return nav

    def process_results(self, df_results: pd.DataFrame, hedge_stats=None):
        df = df_results.copy()
        df.set_index('timestamp', inplace=True)
        
        metrics = {}
        strategies = ['bench', 'hedged', 'call']
        
        # --- 1. CASH FLOWS PREP ---
        # XIRR here is STRATEGY-LEVEL IRR, not investor cash IRR.
        # Option premiums and payoffs are treated as internal self-financing flows.
        # This makes XIRR comparable across strategies, but it is NOT pocket-level IRR.
        
        df['flow'] = df['invested_total'].diff().fillna(0)
        df['flow'].iloc[0] = df['invested_total'].iloc[0]
        
        deposit_schedule = []
        for dt, flow in df['flow'].items():
            if flow > 0:
                deposit_schedule.append((dt, -flow)) # Negative for XIRR calculation (Investment)

        # --- 2. GLOBAL MARKET CONTEXT (BENCHMARK STRESS) ---
        # We identify the "Market Bottom" relative to the Benchmark.
        # This serves as the reference point for Liquidity Injection analysis.
        bench_cummax = df['bench_equity'].cummax()
        bench_dd = (df['bench_equity'] - bench_cummax) / bench_cummax
        market_stress_date = bench_dd.idxmin()
        
        # How much capital had been effectively deployed by the investor at the moment of the crash?
        invested_at_stress = df.loc[market_stress_date, 'invested_total']
        
        metrics['global'] = {
            'stress_date': market_stress_date,
            'bench_at_stress': df.loc[market_stress_date, 'bench_equity'],
            'invested_at_stress': invested_at_stress
        }

        # --- 3. COMPUTE NAV SERIES (for institutional-grade risk metrics) ---
        for strat in strategies:
            df[f'{strat}_nav'] = self._compute_nav_series(df, strat)

        # --- 4. STRATEGY METRICS ---
        for strat in strategies:
            col_eq = f'{strat}_equity'
            col_nav = f'{strat}_nav'
            
            # A. XIRR (uses investor equity with deposits - correct for MWR)
            flows = deposit_schedule.copy()
            # Terminal value treated as positive inflow
            flows.append((df.index[-1], df[col_eq].iloc[-1]))
            strat_xirr = self.xirr(flows)
            
            # B. Risk Metrics (uses NAV - removes deposit distortion)
            mdd, ulcer, recovery_days, strat_trough_date = self._get_drawdown_stats(df[col_nav])
            
            # C. Stress Test Performance
            # How much money did this strategy have when the MARKET was at its worst?
            equity_at_market_stress = df.loc[market_stress_date, col_eq]
            
            metrics[strat] = {
                'XIRR': strat_xirr,
                'MaxDD': mdd,
                'Ulcer': ulcer,
                'Recovery_Days': recovery_days,
                'Terminal': df[col_eq].iloc[-1],
                'Invested': df['invested_total'].iloc[-1],
                'Profit': df[col_eq].iloc[-1] - df['invested_total'].iloc[-1],
                'Equity_at_Market_Stress': equity_at_market_stress,
                'Invested_at_Market_Stress': invested_at_stress
            }

        # --- 4. ATTRIBUTION & COUNTERFACTUALS (The "Why") ---

        # TODO (CRITICAL):
        # Add counterfactual equity curve where hedge payoff is NOT reinvested,
        # but held as cash (0% return), to test robustness of timing alpha.

        if hedge_stats:
            h_metrics = metrics['hedged']
            b_metrics = metrics['bench']
            
            # Data points
            cost = hedge_stats['hedge_cost']
            payoff = hedge_stats['hedge_payoff']
            net_hedge_pnl = payoff - cost
            
            # A. Hedge Efficiency (payoff / cost)
            hedge_efficiency = (payoff / cost) if cost > 0 else 0
            
            # B. Insurance Metrics (institutionally correct)
            # 1. Payoff ROI = payoff / cost (how much did insurance pay per dollar spent)
            # 2. Crisis Coverage = payoff / peak_NAV (what % of peak exposure was recovered)
            
            # Peak NAV before crash (hedged strategy)
            hedged_nav = df['hedged_nav']
            peak_nav_value = hedged_nav.loc[:market_stress_date].max()
            
            # Crisis Coverage: payoff as % of peak exposure
            crisis_coverage = (payoff / (peak_nav_value * df['invested_total'].iloc[-1])) if peak_nav_value > 0 else 0
            
            # Net cost ratio: how much did insurance cost as % of terminal equity
            net_cost_ratio = (cost / h_metrics['Terminal']) if h_metrics['Terminal'] > 0 else 0
            
            # C. ATTRIBUTION: Counterfactual Analysis
            # Total Alpha
            total_alpha = h_metrics['Profit'] - b_metrics['Profit']
            
            # Counterfactual: What if we kept the payoff in cash (0% return) instead of reinvesting?
            # Approximation: We assume the bulk of Payoff happens near Market Stress Date.
            # Market Growth Factor since Stress = (End_Price / Stress_Price)
            
            stress_price = df.loc[market_stress_date, 'price'] # Need 'price' column from BacktestResult
            end_price = df['price'].iloc[-1]
            market_recovery_factor = (end_price / stress_price) - 1.0 if stress_price > 0 else 0
            
            # MAX Reinvestment Effect (Upper Bound)
            # Assumes payoff deployed fully at market stress date.
            # Used as a theoretical ceiling, not exact attribution.
            max_reinvestment_effect = payoff * market_recovery_factor
            
            # Structural / Convexity Alpha
            # Includes convexity effects, beta differences, volatility drag and DCA-path interaction.
            # Not noise. Not timing. Structural.
            structural_alpha = total_alpha - net_hedge_pnl - max_reinvestment_effect
            
            metrics['attribution'] = {
                'Cost': cost,
                'Payoff': payoff,
                'Net_Hedge_PnL': net_hedge_pnl,
                'Hedge_Efficiency': hedge_efficiency,        # payoff / cost
                'Crisis_Coverage': crisis_coverage,          # payoff / (peak_nav * capital)
                'Net_Cost_Ratio': net_cost_ratio,            # cost / terminal_equity
                'Total_Alpha': total_alpha,
                'Reinvestment_Effect': max_reinvestment_effect,
                'Residual_Path_Alpha': structural_alpha
            }

        return df, metrics

    def print_stats(self, metrics):
        print("\n" + "="*95)
        print(f"{'METRIC (INSTITUTIONAL)':<35} | {'BENCH (DCA)':<15} | {'HEDGED (Put)':<15} | {'CALL (ATM)':<15}")
        print("-" * 95)
        
        # Helpers
        def p(val): return f"{val:.2%}"
        def m(val): return f"${val:,.0f}"
        def f(val): return f"{val:.2f}"
        
        row = "{:<35} | {:<15} | {:<15} | {:<15}"
        
        # 1. RETURNS
        print(row.format("MWR / XIRR (Investor Yield)", p(metrics['bench']['XIRR']), p(metrics['hedged']['XIRR']), p(metrics['call']['XIRR'])))
        print(row.format("Total Net Profit", m(metrics['bench']['Profit']), m(metrics['hedged']['Profit']), m(metrics['call']['Profit'])))
        print("-" * 95)
        
        # 2. RISK & STRESS TEST
        # NOTE: With DCA, MaxDD is a path-shape metric, not an economic risk metric.
        print(row.format("Max Drawdown (Path Shape)", p(metrics['bench']['MaxDD']), p(metrics['hedged']['MaxDD']), p(metrics['call']['MaxDD'])))
        print(row.format("Ulcer Index (Stress Load)", f(metrics['bench']['Ulcer']), f(metrics['hedged']['Ulcer']), f(metrics['call']['Ulcer'])))
        print("-" * 40 + " STRESS TEST " + "-" * 40)
        
        # Emphasize Equity at Market Stress (The real test of a hedge)
        print(row.format("Equity at Market Bottom", m(metrics['bench']['Equity_at_Market_Stress']), m(metrics['hedged']['Equity_at_Market_Stress']), m(metrics['call']['Equity_at_Market_Stress'])))
        print(row.format("Invested Cap at Market Bottom", m(metrics['bench']['Invested_at_Market_Stress']), m(metrics['hedged']['Invested_at_Market_Stress']), m(metrics['call']['Invested_at_Market_Stress'])))
        
        # 3. ATTRIBUTION
        if 'attribution' in metrics:
            attr = metrics['attribution']
            print("\n" + "="*95)
            print(">>> HEDGE ALPHA DECOMPOSITION (Source of Returns)")
            print("-" * 95)
            
            print(f"1. Insurance Mechanics:")
            print(f"   Premium Paid (Cost):           {m(attr['Cost'])}")
            print(f"   Crash Payoff (Liquidity):      {m(attr['Payoff'])}")
            print(f"   Net Insurance PnL:             {m(attr['Net_Hedge_PnL'])} (Direct drag/boost)")
            
            print(f"\n2. Insurance Efficiency:")
            print(f"   Hedge Efficiency (Payoff/Cost): {p(attr['Hedge_Efficiency'])} (>100% = profitable insurance)")
            print(f"   Crisis Coverage:                {p(attr['Crisis_Coverage'])} of peak exposure recovered")
            print(f"   Net Cost Ratio:                 {p(attr['Net_Cost_Ratio'])} of terminal equity spent on premiums")
            
            print(f"\n3. Total Alpha Attribution:")
            print(f"   Total Outperformance:          {m(attr['Total_Alpha'])}")
            print(f"   = Net Insurance PnL:           {m(attr['Net_Hedge_PnL'])}")
            print(f"   + Reinvestment Bonus:          {m(attr['Reinvestment_Effect'])} (Timing)")
            print(f"   + Residual Path Alpha:         {m(attr['Residual_Path_Alpha'])} (Beta/Convexity/Noise)")
            
        print("="*95 + "\n")

    def plot_charts(self, df):
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # 1. Equity Curve
        ax1 = plt.subplot(gs[0])
        for strat, label in zip(['bench', 'hedged', 'call'], ['DCA Spot', 'Spot + Put', 'ATM Call']):
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
        for strat in ['bench', 'hedged', 'call']:
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
        
        # Use NAV ratio for strategy-level alpha (no deposit contamination)
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