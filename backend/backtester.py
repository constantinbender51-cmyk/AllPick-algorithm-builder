import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import time 

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0  # Now fractional for crypto
        self.entry_price = 0.0
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
        print(f"DEBUG: Data validation")
        print(f"  DF length: {len(df)}")
        print(f"  Predictions length: {len(predictions)}")
        print(f"  Probabilities length: {len(probabilities)}")
        print(f"  First few predictions: {predictions[:10]}")
        
        # Check if we have any buy signals at all
        buy_signals = np.sum(predictions == 1)
        sell_signals = np.sum(predictions == 0)
        print(f"DEBUG: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
        
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        
        hold_period = strategy_config.get('holdPeriod', 'signal_change')
        stop_loss = strategy_config.get('stopLoss', 2.0) / 100  # Convert to decimal
        take_profit = strategy_config.get('takeProfit', 5.0) / 100
        position_sizing = strategy_config.get('positionSizing', 'fixed')
        confidence_multiplier = strategy_config.get('confidenceMultiplier', 1.0)
        min_position_amount = strategy_config.get('minPositionAmount', 10.0)  # Minimum $ amount to invest
        
        print(f"DEBUG: Strategy config - hold_period: {hold_period}, stop_loss: {stop_loss}, take_profit: {take_profit}")
        print(f"DEBUG: Position sizing: {position_sizing}, confidence_multiplier: {confidence_multiplier}")
        print(f"DEBUG: Initial capital: {self.initial_capital}")
        print(f"DEBUG: Min position amount: ${min_position_amount}")
        
        bars_in_position = 0
        
        # Add debug counters
        signals_processed = 0
        positions_opened = 0
        positions_closed = 0
        buy_signals_skipped = 0
        
        for i in range(len(df)):
            time.sleep(0.1)
            current_price = df.iloc[i]['close']
            timestamp = df.iloc[i]['timestamp']
            prediction = predictions[i]
            confidence = probabilities[i] if len(probabilities) > i else 0.5
            
            signals_processed += 1
            
            # Debug current state periodically
            if i % 100 == 0:  # Print every 100 bars
                print(f"DEBUG Bar {i}: Position: {self.position:.6f} BTC, Prediction: {prediction}, Price: ${current_price:.2f}, Capital: ${self.capital:.2f}")
            
            # Calculate position size
            if position_sizing == 'confidence':
                position_size = confidence * confidence_multiplier
                position_size = np.clip(position_size, 0.01, 1.0)  # At least 1% of capital
            else:
                position_size = 1.0  # Use all capital
            
            # Check if in position
            if abs(self.position) > 1e-8:  # Small epsilon to account for floating point
                # Calculate current P&L percentage
                if self.position > 0:  # Long position
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                else:  # Short position
                    pnl_pct = (self.entry_price - current_price) / self.entry_price
                
                # Check stop loss
                if pnl_pct <= -stop_loss:
                    print(f"DEBUG: Stop loss triggered at bar {i}, P&L: {pnl_pct:.4f}")
                    self._close_position(current_price, timestamp, 'stop_loss')
                    bars_in_position = 0
                    positions_closed += 1
                
                # Check take profit
                elif pnl_pct >= take_profit:
                    print(f"DEBUG: Take profit triggered at bar {i}, P&L: {pnl_pct:.4f}")
                    self._close_position(current_price, timestamp, 'take_profit')
                    bars_in_position = 0
                    positions_closed += 1
                
                # Check hold period
                elif hold_period == 'fixed':
                    bars_in_position += 1
                    if bars_in_position >= 1:
                        print(f"DEBUG: Hold period ended at bar {i}")
                        self._close_position(current_price, timestamp, 'hold_period')
                        bars_in_position = 0
                        positions_closed += 1
                
                # Check signal change
                elif hold_period == 'signal_change':
                    # For binary: 1 = long, 0 = short/exit
                    if (self.position > 0 and prediction == 0) or (self.position < 0 and prediction == 1):
                        print(f"DEBUG: Signal change at bar {i}")
                        self._close_position(current_price, timestamp, 'signal_change')
                        bars_in_position = 0
                        positions_closed += 1
            
            # Check for new entry
            if abs(self.position) < 1e-8:  # No position
                if prediction == 1:  # Buy signal
                    print(f"DEBUG: Buy signal at bar {i}, price: ${current_price:.2f}, position_size: {position_size:.3f}")
                    success = self._open_position(current_price, timestamp, 'long', position_size, min_position_amount)
                    if success:
                        positions_opened += 1
                        bars_in_position = 0
                    else:
                        buy_signals_skipped += 1
                        print(f"DEBUG: Buy signal skipped - insufficient capital or position size too small")
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
        if abs(self.position) > 1e-8:
            print(f"DEBUG: Closing remaining position at end of data")
            self._close_position(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], 'end_of_data')
            positions_closed += 1
        
        print(f"DEBUG: Summary - Signals processed: {signals_processed}, Positions opened: {positions_opened}, Positions closed: {positions_closed}")
        print(f"DEBUG: Buy signals skipped: {buy_signals_skipped}, Total trades: {len(self.trades)}")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        print("DEBUG: Calculated metrics:", metrics)
        return {
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'price_data': [{'date': str(row['timestamp']), 'price': row['close']} 
                          for _, row in df.iterrows()]
        }
    
    def _open_position(self, price: float, timestamp, direction: str, size: float, min_amount: float) -> bool:
        """Open a new fractional position. Returns True if successful, False otherwise."""
        # Calculate the dollar amount to invest
        investment_amount = self.capital * size
        
        print(f"DEBUG _open_position:")
        print(f"  Price: ${price:.2f}")
        print(f"  Capital: ${self.capital:.2f}")
        print(f"  Position size: {size:.3f}")
        print(f"  Investment amount: ${investment_amount:.2f}")
        print(f"  Minimum amount: ${min_amount:.2f}")
        
        # Check if investment meets minimum amount
        if investment_amount < min_amount:
            print(f"  ERROR: Investment amount ${investment_amount:.2f} below minimum ${min_amount:.2f}")
            return False
        
        # Check if we have enough capital
        if investment_amount > self.capital:
            print(f"  WARNING: Investment ${investment_amount:.2f} exceeds capital ${self.capital:.2f}, using available capital")
            investment_amount = self.capital
        
        # Calculate fractional units (e.g., Bitcoin amount)
        units = investment_amount / price
        
        print(f"  Final units: {units:.8f}")
        print(f"  Final investment: ${investment_amount:.2f}")
        
        if units > 1e-8:  # Small epsilon to avoid floating point issues
            self.position = units if direction == 'long' else -units
            self.entry_price = price
            # Deduct the investment from capital
            self.capital -= investment_amount
            
            self.trades.append({
                'date': str(timestamp),
                'type': 'buy' if direction == 'long' else 'sell',
                'price': price,
                'units': abs(self.position),
                'investment': investment_amount,
                'direction': direction
            })
            print(f"  SUCCESS: Position opened - {self.position:.8f} units at ${price:.2f}, remaining capital: ${self.capital:.2f}")
            return True
        else:
            print(f"  ERROR: Failed to open position - units too small: {units:.8f}")
            return False
    
    def _close_position(self, price: float, timestamp, reason: str):
        """Close current fractional position"""
        if abs(self.position) < 1e-8:
            print(f"DEBUG: No position to close")
            return
        
        # Calculate P&L
        if self.position > 0:  # Closing long
            pnl = (price - self.entry_price) * self.position
        else:  # Closing short
            pnl = (self.entry_price - price) * abs(self.position)
        
        # Calculate initial investment to return capital
        initial_investment = self.entry_price * abs(self.position)
        
        # Return capital + P&L
        self.capital += initial_investment + pnl
        
        return_pct = (pnl / initial_investment) * 100 if initial_investment > 0 else 0
        
        self.trades.append({
            'date': str(timestamp),
            'type': 'sell' if self.position > 0 else 'buy',
            'price': price,
            'units': abs(self.position),
            'pnl': pnl,
            'return_pct': return_pct,
            'reason': reason,
            'capital_after': self.capital
        })
        
        print(f"DEBUG: Position closed - {abs(self.position):.8f} units at ${price:.2f}, P&L: ${pnl:.2f} ({return_pct:.2f}%), Reason: {reason}")
        print(f"DEBUG: Capital after close: ${self.capital:.2f}")
        
        self.position = 0.0
        self.entry_price = 0.0
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized P&L with safety checks"""
        if abs(self.position) < 1e-8:
            return float(self.capital)
        
        # Ensure prices are valid
        if not np.isfinite(current_price) or not np.isfinite(self.entry_price):
            return float(self.capital)
        
        # Calculate unrealized P&L
        try:
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) * self.position
            else:
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
            
            equity = self.capital + unrealized_pnl
            
            # Ensure equity is finite
            if np.isfinite(equity) and equity >= 0:
                return float(equity)
            else:
                print(f"DEBUG: Invalid equity calculated: {equity}, using capital: {self.capital}")
                return float(self.capital)
                
        except Exception as e:
            print(f"DEBUG: Error calculating equity: {e}")
            return float(self.capital)
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics with NaN handling"""
        print(f"DEBUG: Calculating metrics - equity curve length: {len(self.equity_curve)}, trades: {len(self.trades)}")
        
        if not self.equity_curve:
            print("DEBUG: No equity curve data")
            return self._get_default_metrics()
        
        try:
            # Extract equity values and handle any NaN/inf
            equity_values = []
            for e in self.equity_curve:
                equity = e['equity']
                if np.isfinite(equity):  # Check if value is finite (not NaN or inf)
                    equity_values.append(equity)
                else:
                    print(f"DEBUG: Found non-finite equity value: {equity}, replacing with previous valid value")
                    # Use last valid value or initial capital if no previous
                    replacement = equity_values[-1] if equity_values else self.initial_capital
                    equity_values.append(replacement)
            
            if not equity_values:
                print("DEBUG: No valid equity values after cleaning")
                return self._get_default_metrics()
            
            print(f"DEBUG: Equity values range: ${min(equity_values):.2f} to ${max(equity_values):.2f}")
            
            # Total return with safety checks
            final_equity = equity_values[-1]
            if np.isfinite(final_equity) and self.initial_capital > 0:
                total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
            else:
                total_return = 0.0
                print(f"DEBUG: Invalid values for total return calculation - final_equity: {final_equity}, initial_capital: {self.initial_capital}")
            
            print(f"DEBUG: Total return: {total_return:.2f}%")
            
            # Calculate returns with safety
            returns = []
            if len(equity_values) > 1:
                for i in range(1, len(equity_values)):
                    if equity_values[i-1] > 0 and np.isfinite(equity_values[i-1]):
                        ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                        if np.isfinite(ret):
                            returns.append(ret)
                        else:
                            returns.append(0.0)
                    else:
                        returns.append(0.0)
            
            print(f"DEBUG: Returns calculated - length: {len(returns)}")
            
            # Sharpe ratio with safety checks
            sharpe_ratio = 0.0
            if len(returns) > 0:
                returns_array = np.array(returns)
                valid_returns = returns_array[np.isfinite(returns_array)]
                if len(valid_returns) > 0 and np.std(valid_returns) > 1e-8:
                    sharpe_ratio = (np.mean(valid_returns) / np.std(valid_returns)) * np.sqrt(252)
                    print(f"DEBUG: Sharpe ratio: {sharpe_ratio:.4f}")
                else:
                    print(f"DEBUG: Cannot calculate Sharpe - valid returns: {len(valid_returns)}, std: {np.std(valid_returns) if len(valid_returns) > 0 else 'N/A'}")
            
            # Max drawdown with safety
            max_drawdown = 0.0
            if len(equity_values) > 0:
                cumulative = np.array(equity_values)
                running_max = np.maximum.accumulate(cumulative)
                # Avoid division by zero in drawdown calculation
                with np.errstate(divide='ignore', invalid='ignore'):
                    drawdown = np.where(running_max > 0, (cumulative - running_max) / running_max * 100, 0.0)
                drawdown = drawdown[np.isfinite(drawdown)]
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            print(f"DEBUG: Max drawdown: {max_drawdown:.2f}%")
            
            # Trade statistics with safety
            closed_trades = [t for t in self.trades if 'pnl' in t and np.isfinite(t.get('pnl', 0))]
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] < 0]
            
            total_trades = len(closed_trades)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            
            print(f"DEBUG: Trade stats - total: {total_trades}, winning: {winning_count}, losing: {losing_count}")
            
            # Win rate with safety
            win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0.0
            
            # Average trade return with safety
            avg_trade_return = 0.0
            if total_trades > 0 and self.initial_capital > 0:
                total_pnl = sum([t['pnl'] for t in closed_trades])
                if np.isfinite(total_pnl):
                    avg_trade_return = (total_pnl / total_trades / self.initial_capital) * 100
                print(f"DEBUG: Avg trade return: {avg_trade_return:.2f}%, total P&L: ${total_pnl:.2f}")
            
            # Ensure all values are finite and convert to native Python types
            metrics = {
                'total_return': float(round(total_return, 2)) if np.isfinite(total_return) else 0.0,
                'sharpe_ratio': float(round(sharpe_ratio, 4)) if np.isfinite(sharpe_ratio) else 0.0,
                'max_drawdown': float(round(max_drawdown, 2)) if np.isfinite(max_drawdown) else 0.0,
                'win_rate': float(round(win_rate, 2)) if np.isfinite(win_rate) else 0.0,
                'total_trades': int(total_trades),
                'winning_trades': int(winning_count),
                'losing_trades': int(losing_count),
                'avg_trade_return': float(round(avg_trade_return, 2)) if np.isfinite(avg_trade_return) else 0.0,
                'final_equity': float(round(final_equity, 2)) if np.isfinite(final_equity) else float(self.initial_capital)
            }
            
            print(f"DEBUG: Final metrics: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"DEBUG: Error calculating metrics: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict:
        """Return default metrics when calculation fails"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_trade_return': 0.0,
            'final_equity': float(self.initial_capital)
        }
