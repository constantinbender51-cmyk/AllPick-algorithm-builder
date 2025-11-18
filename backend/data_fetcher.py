import yfinance as yf
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import time

class DataFetcher:
    """Handles data fetching from various APIs with rate limit management"""
    
    BINANCE_BAR_LIMIT = 1000
    
    def __init__(self):
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    
    def fetch_data(self, api: str, symbol: str, timeframe: str, 
                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch market data from specified API
        
        Args:
            api: 'yahoo' or 'binance'
            symbol: Trading pair symbol
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        if api == 'yahoo':
            return self._fetch_yahoo(symbol, timeframe, start_date, end_date)
        elif api == 'binance':
            return self._fetch_binance(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"Unsupported API: {api}")
    
    def _fetch_yahoo(self, symbol: str, timeframe: str, 
                     start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        # Map timeframes to yfinance intervals
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk'
        }
        
        interval = interval_map.get(timeframe, '1d')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Standardize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'timestamp'})
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            raise Exception(f"Yahoo Finance error: {str(e)}")
    
    def _fetch_binance(self, symbol: str, timeframe: str, 
                       start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data from Binance with chunking to respect 1000 bar limit
        """
        # Map timeframes to Binance format
        timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
        }
        
        tf = timeframe_map.get(timeframe, '1d')
        
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Calculate milliseconds per candle
            ms_per_candle = self._get_ms_per_candle(timeframe)
            
            all_data = []
            current_start = start_ts
            
            # Fetch data in chunks
            while current_start < end_ts:
                try:
                    ohlcv = self.binance.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=tf,
                        since=current_start,
                        limit=self.BINANCE_BAR_LIMIT
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    
                    # Move to next chunk
                    last_timestamp = ohlcv[-1][0]
                    current_start = last_timestamp + ms_per_candle
                    
                    # Respect rate limits
                    time.sleep(self.binance.rateLimit / 1000)
                    
                    # Break if we've reached the end date
                    if last_timestamp >= end_ts:
                        break
                        
                except Exception as e:
                    print(f"Error fetching chunk: {str(e)}")
                    break
            
            if not all_data:
                raise ValueError(f"No data returned for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filter to exact date range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            return df
        
        except Exception as e:
            raise Exception(f"Binance error: {str(e)}")
    
    def _get_ms_per_candle(self, timeframe: str) -> int:
        """Calculate milliseconds per candle for different timeframes"""
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }
        return timeframe_ms.get(timeframe, 24 * 60 * 60 * 1000)
    
    def get_preview(self, df: pd.DataFrame, rows: int = 5) -> Dict:
        """Get preview statistics of the data"""
        return {
            'total_rows': len(df),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'sample_data': df.head(rows).to_dict('records'),
            'statistics': {
                'close_mean': float(df['close'].mean()),
                'close_std': float(df['close'].std()),
                'volume_mean': float(df['volume'].mean())
            }
        }
