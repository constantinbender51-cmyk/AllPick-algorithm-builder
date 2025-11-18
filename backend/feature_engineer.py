import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List

class FeatureEngineer:
    """Handles all feature engineering tasks"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_columns = []
    
    def add_technical_indicator(self, feature_type: str, params: Dict) -> pd.DataFrame:
        """
        Add a technical indicator to the dataframe
        
        Args:
            feature_type: Type of indicator (SMA, EMA, RSI, etc.)
            params: Parameters for the indicator
        
        Returns:
            Updated dataframe with new feature
        """
        if feature_type == 'SMA':
            period = params.get('period', 20)
            indicator = SMAIndicator(close=self.df['close'], window=period)
            col_name = f'SMA_{period}'
            self.df[col_name] = indicator.sma_indicator()
            self.feature_columns.append(col_name)
        
        elif feature_type == 'EMA':
            period = params.get('period', 20)
            indicator = EMAIndicator(close=self.df['close'], window=period)
            col_name = f'EMA_{period}'
            self.df[col_name] = indicator.ema_indicator()
            self.feature_columns.append(col_name)
        
        elif feature_type == 'RSI':
            period = params.get('period', 14)
            indicator = RSIIndicator(close=self.df['close'], window=period)
            col_name = f'RSI_{period}'
            self.df[col_name] = indicator.rsi()
            self.feature_columns.append(col_name)
        
        elif feature_type == 'MACD':
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            indicator = MACD(close=self.df['close'], 
                           window_fast=fast, 
                           window_slow=slow, 
                           window_sign=signal)
            
            self.df[f'MACD_{fast}_{slow}'] = indicator.macd()
            self.df[f'MACD_signal_{signal}'] = indicator.macd_signal()
            self.df[f'MACD_diff'] = indicator.macd_diff()
            
            self.feature_columns.extend([
                f'MACD_{fast}_{slow}',
                f'MACD_signal_{signal}',
                f'MACD_diff'
            ])
        
        elif feature_type == 'BB':
            period = params.get('period', 20)
            std = params.get('std', 2)
            indicator = BollingerBands(close=self.df['close'], 
                                      window=period, 
                                      window_dev=std)
            
            self.df[f'BB_upper_{period}'] = indicator.bollinger_hband()
            self.df[f'BB_middle_{period}'] = indicator.bollinger_mavg()
            self.df[f'BB_lower_{period}'] = indicator.bollinger_lband()
            self.df[f'BB_width_{period}'] = indicator.bollinger_wband()
            
            self.feature_columns.extend([
                f'BB_upper_{period}',
                f'BB_middle_{period}',
                f'BB_lower_{period}',
                f'BB_width_{period}'
            ])
        
        elif feature_type == 'ATR':
            period = params.get('period', 14)
            indicator = AverageTrueRange(high=self.df['high'], 
                                        low=self.df['low'], 
                                        close=self.df['close'], 
                                        window=period)
            col_name = f'ATR_{period}'
            self.df[col_name] = indicator.average_true_range()
            self.feature_columns.append(col_name)
        
        elif feature_type == 'STOCH':
            period = params.get('period', 14)
            indicator = StochasticOscillator(high=self.df['high'], 
                                            low=self.df['low'], 
                                            close=self.df['close'], 
                                            window=period)
            
            self.df[f'STOCH_{period}'] = indicator.stoch()
            self.df[f'STOCH_signal_{period}'] = indicator.stoch_signal()
            
            self.feature_columns.extend([
                f'STOCH_{period}',
                f'STOCH_signal_{period}'
            ])
        
        return self.df
    
    def add_polynomial_features(self, degree: int, feature_cols: List[str] = None):
        """
        Add polynomial features
        
        Args:
            degree: Polynomial degree
            feature_cols: Columns to create polynomials from (default: all numeric)
        """
        if degree <= 1:
            return
        
        if feature_cols is None:
            feature_cols = self.feature_columns if self.feature_columns else ['close']
        
        # Get the feature data
        feature_data = self.df[feature_cols].fillna(0)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(feature_data)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(feature_cols)
        
        # Add new polynomial features (skip original features)
        for i, name in enumerate(feature_names):
            if name not in feature_cols:
                col_name = f'poly_{name}'
                self.df[col_name] = poly_features[:, i]
                self.feature_columns.append(col_name)
    
    def add_feature_multiplications(self, feature_pairs: List[tuple]):
        """
        Multiply specified feature pairs
        
        Args:
            feature_pairs: List of tuples containing feature names to multiply
        """
        for feat1, feat2 in feature_pairs:
            if feat1 in self.df.columns and feat2 in self.df.columns:
                col_name = f'{feat1}_x_{feat2}'
                self.df[col_name] = self.df[feat1] * self.df[feat2]
                self.feature_columns.append(col_name)
    
    def add_derivatives(self, first_order: bool = True, second_order: bool = False):
        """
        Add derivatives (rate of change) of features
        
        Args:
            first_order: Add first derivative
            second_order: Add second derivative
        """
        base_features = ['close', 'volume'] + self.feature_columns
        
        for feature in base_features:
            if feature in self.df.columns:
                if first_order:
                    col_name = f'{feature}_derivative_1'
                    self.df[col_name] = self.df[feature].diff()
                    self.feature_columns.append(col_name)
                
                if second_order:
                    col_name = f'{feature}_derivative_2'
                    self.df[col_name] = self.df[feature].diff().diff()
                    self.feature_columns.append(col_name)
    
    def create_target(self, target_type: str, horizon: int, threshold: float = 0.5):
        """
        Create target variable for prediction
        
        Args:
            target_type: 'binary' or 'regression'
            horizon: Number of bars to look ahead
            threshold: Threshold for binary classification (percentage)
        
        Returns:
            Dataframe with target column
        """
        # Calculate future returns
        future_price = self.df['close'].shift(-horizon)
        returns = ((future_price - self.df['close']) / self.df['close']) * 100
        
        if target_type == 'binary':
            # Binary classification: 1 for up, 0 for down
            self.df['target'] = (returns > threshold).astype(int)
        else:
            # Regression: actual return percentage
            self.df['target'] = returns
        
        # Remove rows with NaN targets (last 'horizon' rows)
        self.df = self.df[:-horizon]
        
        return self.df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns"""
        base_features = ['open', 'high', 'low', 'close', 'volume']
        return base_features + self.feature_columns
    
    def clean_data(self):
        """Remove NaN and infinite values"""
        # Forward fill NaN values
        self.df = self.df.fillna(method='ffill')
        # Backward fill any remaining NaN
        self.df = self.df.fillna(method='bfill')
        # Replace infinite values
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(0)
        
        return self.df
