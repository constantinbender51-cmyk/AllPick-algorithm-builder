import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Dict, List, Tuple

class DataNormalizer:
    """Handles data normalization with multiple methods"""
    
    def __init__(self):
        self.scalers = {}
        self.normalization_config = {}
    
    def normalize(self, df: pd.DataFrame, normalization_config: Dict[str, str], 
                  feature_columns: List[str]) -> pd.DataFrame:
        """
        Normalize specified features using configured methods
        
        Args:
            df: Input dataframe
            normalization_config: Dict mapping feature names to normalization methods
            feature_columns: List of feature column names
        
        Returns:
            Normalized dataframe
        """
        df_normalized = df.copy()
        self.normalization_config = normalization_config
        
        for feature in feature_columns:
            if feature not in df.columns:
                continue
            
            method = normalization_config.get(feature, 'none')
            
            if method == 'none' or method is None:
                continue
            
            # Get the data
            data = df[[feature]].values
            
            # Apply normalization
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                continue
            
            # Fit and transform
            normalized_data = scaler.fit_transform(data)
            df_normalized[feature] = normalized_data.flatten()
            
            # Store scaler for later use
            self.scalers[feature] = scaler
        
        return df_normalized
    
    def get_normalization_preview(self, df: pd.DataFrame, 
                                   normalization_config: Dict[str, str],
                                   feature_columns: List[str]) -> Dict:
        """
        Preview normalization statistics
        
        Args:
            df: Input dataframe
            normalization_config: Normalization configuration
            feature_columns: Features to normalize
        
        Returns:
            Dictionary with before/after statistics
        """
        stats = []
        
        for feature in feature_columns:
            if feature not in df.columns:
                continue
            
            method = normalization_config.get(feature, 'none')
            
            if method == 'none' or method is None:
                continue
            
            # Original statistics
            orig_data = df[feature].values
            orig_mean = float(np.nanmean(orig_data))
            orig_std = float(np.nanstd(orig_data))
            orig_min = float(np.nanmin(orig_data))
            orig_max = float(np.nanmax(orig_data))
            
            # Normalized statistics
            data_reshaped = orig_data.reshape(-1, 1)
            
            if method == 'minmax':
                scaler = MinMaxScaler()
                normalized_data = scaler.fit_transform(data_reshaped)
            elif method == 'standard':
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(data_reshaped)
            elif method == 'robust':
                scaler = RobustScaler()
                normalized_data = scaler.fit_transform(data_reshaped)
            else:
                continue
            
            norm_mean = float(np.nanmean(normalized_data))
            norm_std = float(np.nanstd(normalized_data))
            norm_min = float(np.nanmin(normalized_data))
            norm_max = float(np.nanmax(normalized_data))
            
            stats.append({
                'feature': feature,
                'method': method,
                'orig_mean': orig_mean,
                'orig_std': orig_std,
                'orig_min': orig_min,
                'orig_max': orig_max,
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'norm_min': norm_min,
                'norm_max': norm_max
            })
        
        return {'stats': stats}
    
    def inverse_transform(self, feature: str, normalized_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized values back to original scale
        
        Args:
            feature: Feature name
            normalized_values: Normalized values
        
        Returns:
            Original scale values
        """
        if feature in self.scalers:
            values_reshaped = normalized_values.reshape(-1, 1)
            return self.scalers[feature].inverse_transform(values_reshaped).flatten()
        return normalized_values
    
    def split_train_test(self, df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Input dataframe
            split_ratio: Ratio of training data
        
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
