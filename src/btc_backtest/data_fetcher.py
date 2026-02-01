"""
External API connector for exchange data.
Wraps CCXT for OHLCV fetching.
"""
import ccxt
import pandas as pd
import time
from datetime import datetime
from typing import Optional

from .settings import Settings


class DataFetcher:
    """
    CCXT-based data fetcher for cryptocurrency OHLCV data.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchange = getattr(ccxt, settings.EXCHANGE_ID)()
    
    def fetch_data(
        self, 
        since: Optional[datetime] = None,
        start_override: Optional[datetime] = None,
        end_override: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            since: Optional start datetime (for incremental updates).
            start_override: Override for START_DATE (for historical fetch).
            end_override: Override for END_DATE (for historical fetch).
            
        Returns:
            pd.DataFrame with columns: timestamp, open, high, low, close, volume
        """
        symbol = self.settings.SYMBOL
        
        # Determine start timestamp
        if since is not None:
            since_ts = int(since.timestamp() * 1000)
        elif start_override is not None:
            since_ts = int(start_override.timestamp() * 1000)
        else:
            since_ts = self.exchange.parse8601(self.settings.START_DATE)
        
        # Determine end timestamp
        if end_override is not None:
            end_ts = int(end_override.timestamp() * 1000)
        else:
            end_ts = self.exchange.parse8601(self.settings.END_DATE)
        
        print(f"[DataFetcher] Fetching {symbol} from {self.settings.EXCHANGE_ID}...")
        
        all_ohlcv = []
        
        while since_ts < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self.settings.TIMEFRAME,
                    since=since_ts,
                    limit=1000
                )
                if not ohlcv:
                    break
                
                since_ts = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[DataFetcher] Error: {e}")
                break
        
        if not all_ohlcv:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"[DataFetcher] Loaded {len(df)} candles.")
        return df
