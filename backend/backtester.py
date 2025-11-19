import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray, 
                     probabilities: np.ndarray, strategy_config: Dict) -> Dict:
        """
        Run backtest with given predictions and strategy
        
        Args:
            df: DataFrame with price data
            predictions: Model predictions
            probabilities: Prediction probabilities/confidence
            strategy_config: Strategy configuration
        
        Returns:
            Backtest results
        """
        print("len df: ", len(df))
        print("len prob: ", len(probabilities))
        print("len pred: ", len(predictions))
        print("len strategy_config: ", len(strategy_config))
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
        
        hold_period = strategy_config.get('holdPeriod', 'signal_change')
        stop_loss = strategy_config.get('stopLoss', 2.0) / 100  # Convert to decimal
        take_profit = strategy_config.get('takeProfit', 5.0) / 100
        position_sizing = strategy_config.get('positionSizing', 'fixed')
        confidence_multiplier = strategy_config.get('confidenceMultiplier', 1.0)
        
        bars_in_position = 0
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            timestamp = df.iloc[i]['timestamp']
            prediction = predictions[i]
            confidence = probabilities[i] if len(probabilities) > i else 0.5
            
            # Calculate position size
            if position_sizing == 'confidence':
                position_size = confidence * confidence_multiplier
                position_size = np.clip(position_size, 0.1, 1.0)
            else:
                position_size = 1.0
            
            # Check if in position
            if self.position != 0:
                # Calculate current P&L
                if self.position > 0:  # Long position
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                else:  # Short position
                    pnl_pct = (self.entry_price - current_price) / self.entry_price
                
                # Check stop loss
                if pnl_pct <= -stop_loss:
                    self._close_position(current_price, timestamp, 'stop_loss')
                    bars_in_position = 0
                
                # Check take profit
                elif pnl_pct >= take_profit:
                    self._close_position(current_price, timestamp, 'take_profit')
                    bars_in_position = 0
                
                # Check hold period
                elif hold_period == 'fixed':
                    bars_in_position += 1
                    if bars_in_position >= 1:
                        self._close_position(current_price, timestamp, 'hold_period')
                        bars_in_position = 0
                
                # Check signal change
                elif hold_period == 'signal_change':
                    # For binary: 1 = long, 0 = short/exit
                    if (self.position > 0 and prediction == 0) or (self.position < 0 and prediction == 1):
                        self._close_position(current_price, timestamp, 'signal_change')
                        bars_in_position = 0
            
            # Check for new entry
            if self.position == 0:
                if prediction == 1:  # Buy signal
                    self._open_position(current_price, timestamp, 'long', position_size)
                elif prediction == 0:  # Could implement short if desired
                    # For now, only trading long positions
                    pass
            
            # Record equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                'date': str(timestamp),
                'equity': equity,
                'price': current_price
            })
        
        # Close any remaining position
        if self.position != 0:
            self._close_position(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], 'end_of_data')
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        print("met: ", (metrics)
        return {
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'price_data': [{'date': str(row['timestamp']), 'price': row['close']} 
                          for _, row in df.iterrows()]
        }
    
    def _open_position(self, price: float, timestamp, direction: str, size: float):
        """Open a new position"""
        shares = int((self.capital * size) / price)
        if shares > 0:
            self.position = shares if direction == 'long' else -shares
            self.entry_price = price
            
            self.trades.append({
                'date': str(timestamp),
                'type': 'buy' if direction == 'long' else 'sell',
                'price': price,
                'shares': abs(self.position),
                'direction': direction
            })
    
    def _close_position(self, price: float, timestamp, reason: str):
        """Close current position"""
        if self.position == 0:
            return
        
        # Calculate P&L
        if self.position > 0:  # Closing long
            pnl = (price - self.entry_price) * self.position
        else:  # Closing short
            pnl = (self.entry_price - price) * abs(self.position)
        
        self.capital += pnl
        
        self.trades.append({
            'date': str(timestamp),
            'type': 'sell' if self.position > 0 else 'buy',
            'price': price,
            'shares': abs(self.position),
            'pnl': pnl,
            'reason': reason
        })
        
        self.position = 0
        self.entry_price = 0
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity"""
        if self.position == 0:
            return self.capital
        
        # Calculate unrealized P&L
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
        else:
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
        
        return self.capital + unrealized_pnl
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {}
        
        # Extract equity values
        equity_values = [e['equity'] for e in self.equity_curve]
        
        # Total return
        total_return = ((equity_values[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Sharpe ratio (assuming 252 trading days)
        if len(returns) > 0 and np.std(returns) != 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative = np.array(equity_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        winning_trades = [t for t in self.trades if 'pnl' in t and t['pnl'] > 0]
        losing_trades = [t for t in self.trades if 'pnl' in t and t['pnl'] < 0]
        
        total_trades = len([t for t in self.trades if 'pnl' in t])
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_trade_return = 0
        if total_trades > 0:
            total_pnl = sum([t['pnl'] for t in self.trades if 'pnl' in t])
            avg_trade_return = (total_pnl / total_trades / self.initial_capital) * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'avg_trade_return': avg_trade_return,
            'final_equity': equity_values[-1]
        }
