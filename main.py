import warnings
from config import Config
from data_fetcher import DataEngine
from backtest_engine import BacktestEngine
from analytics import InstitutionalAnalytics

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 1. Config
    cfg = Config()
    
    # 2. Data
    eng = DataEngine(cfg)
    df_data = eng.fetch_data()
    
    if df_data.empty:
        print("No data found.")
        exit()
        
    # 3. Simulation
    bt = BacktestEngine(cfg, df_data)
    df_results = bt.run()
    
    # 4. Analytics
    analytics = InstitutionalAnalytics(risk_free_rate=cfg.RISK_FREE_RATE)
    
    # Pass df_results AND hidden hedge statistics
    df_processed, metrics = analytics.process_results(df_results, hedge_stats=bt.final_stats)
    
    analytics.print_stats(metrics)
    analytics.plot_charts(df_processed)
