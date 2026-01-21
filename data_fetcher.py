import ccxt
import pandas as pd
import time
from config import Config

class DataEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.exchange = getattr(ccxt, cfg.EXCHANGE_ID)()

    def fetch_data(self):
        print(f"[{self.cfg.SYMBOL}] Fetching OHLCV Data from {self.cfg.EXCHANGE_ID}...")
        
        # Конвертация дат в timestamp ms
        since = self.exchange.parse8601(self.cfg.START_DATE)
        end_ts = self.exchange.parse8601(self.cfg.END_DATE)
        
        all_ohlcv = []
        
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.cfg.SYMBOL, 
                    timeframe=self.cfg.TIMEFRAME, 
                    since=since, 
                    limit=1000
                )
                if not ohlcv:
                    break
                
                since = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)
                
                # Небольшая пауза, чтобы не дудосить API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break

        if not all_ohlcv:
            return pd.DataFrame()

        # Создаем DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Фильтруем по датам
        mask = (df['timestamp'] >= self.cfg.START_DATE) & (df['timestamp'] <= self.cfg.END_DATE)
        df = df.loc[mask].reset_index(drop=True)
        
        print(f"Data loaded: {len(df)} candles.")
        return df