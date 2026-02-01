"""
Entry point for BTC Backtest simulation.
Run with: python run.py
"""
import warnings
import sys
from pathlib import Path

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent))

from src.btc_backtest.settings import get_settings
from src.btc_backtest.data_lake import DataLake
from src.btc_backtest.engine import BacktestEngine
from src.btc_backtest.analytics import Analytics

warnings.filterwarnings('ignore')


def main() -> None:
    """Run the backtest simulation."""
    # 1. Config
    settings = get_settings()
    print(f"[Config] {settings.SYMBOL} | {settings.START_DATE} → {settings.END_DATE}")
    
    # 2. Data (with caching)
    lake = DataLake(settings)
    df_data = lake.load_or_fetch()
    
    if df_data.empty:
        print("No data found.")
        return
    
    # 3. Simulation
    bt = BacktestEngine(settings, df_data)
    df_results = bt.run()
    
    # 4. Analytics
    analytics = Analytics(risk_free_rate=settings.RISK_FREE_RATE)
    
    # Pass df_results AND hidden hedge statistics
    df_processed, metrics = analytics.process_results(df_results, hedge_stats=bt.final_stats)
    
    analytics.print_stats(metrics)
    analytics.plot_charts(df_processed)


if __name__ == "__main__":
    main()
