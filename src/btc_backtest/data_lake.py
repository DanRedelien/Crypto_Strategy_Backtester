"""
Data Lake with Parquet caching.
Smart-loader: fetch from API only if cache is missing or stale.
"""
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .settings import get_settings, Settings
from .data_fetcher import DataFetcher


class DataLake:
    """
    Smart data loader with local Parquet cache.
    
    1. Check Parquet cache.
    2. If missing/stale -> Fetch delta from API.
    3. Merge & Save.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.fetcher = DataFetcher(self.settings)
    
    def load_or_fetch(self) -> pd.DataFrame:
        """
        Load data from cache or fetch from exchange.
        
        Checks:
        1. Cache exists
        2. Cache covers START_DATE (fetch historical if not)
        3. Cache is fresh at END_DATE (fetch updates if stale)
        
        Returns:
            pd.DataFrame with columns: timestamp, open, high, low, close, volume
        """
        symbol = self.settings.SYMBOL
        file_path = self.settings.get_cache_path() / f"{symbol}.parquet"
        
        requested_start = pd.to_datetime(self.settings.START_DATE)
        requested_end = pd.to_datetime(self.settings.END_DATE)
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            
            if len(df) > 0:
                cache_start = pd.to_datetime(df['timestamp'].iloc[0])
                cache_end = pd.to_datetime(df['timestamp'].iloc[-1])
                
                needs_historical = requested_start < cache_start
                needs_update = requested_end > cache_end
                
                # 1. Fetch historical data if START_DATE is before cache start
                if needs_historical:
                    print(f"[DataLake] Cache missing historical data. Fetching from {requested_start.date()}...")
                    historical_data = self.fetcher.fetch_data(
                        start_override=requested_start, 
                        end_override=cache_start
                    )
                    if not historical_data.empty:
                        df = pd.concat([historical_data, df], ignore_index=True)
                        df = df.drop_duplicates(subset=['timestamp'], keep='last')
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        print(f"[DataLake] Added historical: {len(historical_data)} candles.")
                
                # 2. Fetch updates if cache is stale (end < requested_end)
                if needs_update:
                    print(f"[DataLake] Cache needs update. Fetching from {cache_end.date()}...")
                    new_data = self.fetcher.fetch_data(since=cache_end)
                    if not new_data.empty:
                        df = pd.concat([df, new_data], ignore_index=True)
                        df = df.drop_duplicates(subset=['timestamp'], keep='last')
                        print(f"[DataLake] Added updates: {len(new_data)} candles.")
                
                # Save updated cache
                if needs_historical or needs_update:
                    df.to_parquet(file_path, index=False)
                    print(f"[DataLake] Saved cache: {len(df)} candles total.")
                else:
                    print(f"[DataLake] Cache covers range ({len(df)} candles).")
                    
            return self._filter_by_date_range(df)
        
        else:
            # Cold start
            print(f"[DataLake] No cache found, fetching from API...")
            df = self.fetcher.fetch_data()
            if not df.empty:
                df.to_parquet(file_path, index=False)
                print(f"[DataLake] Saved cache: {len(df)} candles.")
            return self._filter_by_date_range(df)
    
    def _filter_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to configured date range."""
        if df.empty:
            return df
            
        start = pd.to_datetime(self.settings.START_DATE)
        end = pd.to_datetime(self.settings.END_DATE)
        
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        return df.loc[mask].reset_index(drop=True)
