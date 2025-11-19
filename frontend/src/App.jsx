import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { ChevronRight, ChevronLeft, Play, Plus, Trash2, TrendingUp, Activity, BarChart3, Database, Settings, Target, CheckCircle } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const AlgoBuilder = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [isMobile, setIsMobile] = useState(false);
  
  // Data Selection State
  const [dataSources, setDataSources] = useState([]);
  const [newDataSource, setNewDataSource] = useState({
    api: 'yahoo',
    symbol: '',
    timeframe: '1d',
    startDate: '',
    endDate: ''
  });
  const [dataLoading, setDataLoading] = useState(false);
  
  // Feature Selection State
  const [features, setFeatures] = useState([]);
  const [newFeature, setNewFeature] = useState({
    type: 'SMA',
    params: { period: 20 }
  });
  const [polynomialDegree, setPolynomialDegree] = useState(1);
  const [featureMultiplications, setFeatureMultiplications] = useState([]);
  const [derivatives, setDerivatives] = useState({ first: false, second: false });
  
  // Normalization State
  const [normalization, setNormalization] = useState({});
  const [normalizationPreview, setNormalizationPreview] = useState(null);
  
  // Target State
  const [target, setTarget] = useState({
    type: 'binary',
    horizon: 1,
    threshold: 0.5
  });
  
  // Algorithm State
  const [algorithm, setAlgorithm] = useState({
    type: 'random_forest',
    params: {
      n_estimators: 100,
      max_depth: 10,
      random_state: 42
    },
    trainTestSplit: 0.8
  });
  
  // Strategy State
  const [strategy, setStrategy] = useState({
    holdPeriod: 'signal_change',
    stopLoss: 2.0,
    takeProfit: 5.0,
    positionSizing: 'fixed',
    confidenceMultiplier: 1.0,
    initialCapital: 10000
  });
  
  // Evaluation State
  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const tabs = [
    { name: 'Data', icon: Database },
    { name: 'Features', icon: Activity },
    { name: 'Normalization', icon: BarChart3 },
    { name: 'Target', icon: Target },
    { name: 'Algorithm', icon: Settings },
    { name: 'Strategy', icon: TrendingUp },
    { name: 'Evaluation', icon: CheckCircle }
  ];

  const featureTypes = {
    'SMA': { label: 'Simple Moving Average', params: ['period'] },
    'EMA': { label: 'Exponential Moving Average', params: ['period'] },
    'RSI': { label: 'Relative Strength Index', params: ['period'] },
    'MACD': { label: 'MACD', params: ['fast', 'slow', 'signal'] },
    'BB': { label: 'Bollinger Bands', params: ['period', 'std'] },
    'ATR': { label: 'Average True Range', params: ['period'] },
    'STOCH': { label: 'Stochastic Oscillator', params: ['period'] }
  };

  const algorithmTypes = {
    'linear_regression': { label: 'Linear Regression', params: {} },
    'neural_network': { label: 'Neural Network', params: { layers: '[64, 32]', epochs: 50, learning_rate: 0.001 } },
    'rnn': { label: 'RNN/LSTM', params: { units: 64, epochs: 50, learning_rate: 0.001 } },
    'random_forest': { label: 'Random Forest', params: { n_estimators: 100, max_depth: 10 } },
    'knn': { label: 'K-Nearest Neighbors', params: { n_neighbors: 5 } },
    'naive_bayes': { label: 'Naive Bayes', params: {} },
    'transformer': { label: 'Transformer', params: { d_model: 64, n_heads: 4, n_layers: 2, epochs: 50 } },
    'lightgbm': { label: 'LightGBM', params: { n_estimators: 100, max_depth: 10, learning_rate: 0.1 } }
  };

  const addDataSource = async () => {
    if (!newDataSource.symbol || !newDataSource.startDate || !newDataSource.endDate) {
      alert('Please fill in all fields');
      return;
    }
    
    setDataLoading(true);
    try {
      const response = await fetch(`${API_URL}/data/fetch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newDataSource)
      });
      const data = await response.json();
      
      setDataSources([...dataSources, { ...newDataSource, id: Date.now(), preview: data.preview }]);
      setNewDataSource({ api: 'yahoo', symbol: '', timeframe: '1d', startDate: '', endDate: '' });
    } catch (error) {
      alert('Error fetching data: ' + error.message);
    }
    setDataLoading(false);
  };

  const addFeature = () => {
    const feature = {
      id: Date.now(),
      type: newFeature.type,
      params: { ...newFeature.params },
      label: `${newFeature.type}(${Object.values(newFeature.params).join(',')})`
    };
    setFeatures([...features, feature]);
  };

  const removeFeature = (id) => {
    setFeatures(features.filter(f => f.id !== id));
  };

  const updateNormalization = (featureName, method) => {
    setNormalization({ ...normalization, [featureName]: method });
  };

  const previewNormalization = async () => {
    try {
      const response = await fetch(`${API_URL}/features/normalize-preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ normalization, features })
      });
      const data = await response.json();
      setNormalizationPreview(data);
    } catch (error) {
      alert('Error previewing normalization: ' + error.message);
    }
  };

  const runBacktest = async () => {
    setBacktestLoading(true);
    try {
      const response = await fetch(`${API_URL}/backtest/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataSources,
          features,
          normalization,
          target,
          algorithm,
          strategy
        })
      });
      const data = await response.json();
      setBacktestResults(data);
      setCurrentTab(6); // Move to evaluation tab
    } catch (error) {
      alert('Error running backtest: ' + error.message);
    }
    setBacktestLoading(false);
  };

  const renderDataTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Data Selection</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">API Source</label>
            <select 
              className="w-full p-2 border border-gray-300 rounded-md"
              value={newDataSource.api}
              onChange={(e) => setNewDataSource({...newDataSource, api: e.target.value})}
            >
              <option value="yahoo">Yahoo Finance</option>
              <option value="binance">Binance</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Symbol</label>
            <input 
              type="text"
              className="w-full p-2 border border-gray-300 rounded-md"
              placeholder="AAPL or BTCUSDT"
              value={newDataSource.symbol}
              onChange={(e) => setNewDataSource({...newDataSource, symbol: e.target.value.toUpperCase()})}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Timeframe</label>
            <select 
              className="w-full p-2 border border-gray-300 rounded-md"
              value={newDataSource.timeframe}
              onChange={(e) => setNewDataSource({...newDataSource, timeframe: e.target.value})}
            >
              <option value="1m">1 Minute</option>
              <option value="5m">5 Minutes</option>
              <option value="15m">15 Minutes</option>
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
              <option value="1d">1 Day</option>
              <option value="1w">1 Week</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
            <input 
              type="date"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={newDataSource.startDate}
              onChange={(e) => setNewDataSource({...newDataSource, startDate: e.target.value})}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
            <input 
              type="date"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={newDataSource.endDate}
              onChange={(e) => setNewDataSource({...newDataSource, endDate: e.target.value})}
            />
          </div>
        </div>
        
        <button 
          onClick={addDataSource}
          disabled={dataLoading}
          className="w-full md:w-auto bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 transition flex items-center justify-center gap-2"
        >
          <Plus size={20} />
          {dataLoading ? 'Loading...' : 'Add Data Source'}
        </button>
      </div>
      
      {dataSources.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-4">Added Data Sources</h3>
          <div className="space-y-3">
            {dataSources.map((ds) => (
              <div key={ds.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                <div>
                  <span className="font-medium">{ds.symbol}</span>
                  <span className="text-gray-600 ml-2">({ds.api} - {ds.timeframe})</span>
                  <span className="text-gray-500 ml-2 text-sm">{ds.startDate} to {ds.endDate}</span>
                </div>
                <button 
                  onClick={() => setDataSources(dataSources.filter(d => d.id !== ds.id))}
                  className="text-red-600 hover:text-red-800"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderFeaturesTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Feature Engineering</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md space-y-4">
        <h3 className="text-lg font-semibold">Add Technical Indicator</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Indicator Type</label>
            <select 
              className="w-full p-2 border border-gray-300 rounded-md"
              value={newFeature.type}
              onChange={(e) => {
                const type = e.target.value;
                const defaultParams = type === 'MACD' 
                  ? { fast: 12, slow: 26, signal: 9 }
                  : type === 'BB'
                  ? { period: 20, std: 2 }
                  : { period: 14 };
                setNewFeature({ type, params: defaultParams });
              }}
            >
              {Object.entries(featureTypes).map(([key, val]) => (
                <option key={key} value={key}>{val.label}</option>
              ))}
            </select>
          </div>
          
          {Object.entries(newFeature.params).map(([param, value]) => (
            <div key={param}>
              <label className="block text-sm font-medium text-gray-700 mb-2 capitalize">{param}</label>
              <input 
                type="number"
                className="w-full p-2 border border-gray-300 rounded-md"
                value={value}
                onChange={(e) => setNewFeature({
                  ...newFeature,
                  params: { ...newFeature.params, [param]: parseInt(e.target.value) }
                })}
              />
            </div>
          ))}
        </div>
        
        <button 
          onClick={addFeature}
          className="w-full md:w-auto bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 transition flex items-center gap-2"
        >
          <Plus size={20} />
          Add Feature
        </button>
      </div>
      
      {features.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-4">Added Features</h3>
          <div className="space-y-2">
            {features.map((feature) => (
              <div key={feature.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                <span className="font-medium">{feature.label}</span>
                <button 
                  onClick={() => removeFeature(feature.id)}
                  className="text-red-600 hover:text-red-800"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="bg-white p-6 rounded-lg shadow-md space-y-4">
        <h3 className="text-lg font-semibold">Advanced Feature Engineering</h3>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Polynomial Degree</label>
          <input 
            type="number"
            min="1"
            max="5"
            className="w-full p-2 border border-gray-300 rounded-md"
            value={polynomialDegree}
            onChange={(e) => setPolynomialDegree(parseInt(e.target.value))}
          />
        </div>
        
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2">
            <input 
              type="checkbox"
              checked={derivatives.first}
              onChange={(e) => setDerivatives({...derivatives, first: e.target.checked})}
            />
            <span className="text-sm">First Derivative</span>
          </label>
          
          <label className="flex items-center gap-2">
            <input 
              type="checkbox"
              checked={derivatives.second}
              onChange={(e) => setDerivatives({...derivatives, second: e.target.checked})}
            />
            <span className="text-sm">Second Derivative</span>
          </label>
        </div>
      </div>
    </div>
  );

  const renderNormalizationTab = () => {
    const allFeatures = ['Open', 'High', 'Low', 'Close', 'Volume', ...features.map(f => f.label)];
    
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800">Normalization</h2>
        
        <div className="bg-white p-6 rounded-lg shadow-md space-y-4">
          <h3 className="text-lg font-semibold">Configure Normalization</h3>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-3">Feature</th>
                  <th className="text-left p-3">Normalization Method</th>
                </tr>
              </thead>
              <tbody>
                {allFeatures.map((feature) => (
                  <tr key={feature} className="border-b">
                    <td className="p-3 font-medium">{feature}</td>
                    <td className="p-3">
                      <select 
                        className="w-full p-2 border border-gray-300 rounded-md"
                        value={normalization[feature] || 'none'}
                        onChange={(e) => updateNormalization(feature, e.target.value)}
                      >
                        <option value="none">None</option>
                        <option value="minmax">Min-Max Scaling</option>
                        <option value="standard">Standard Scaling (Z-score)</option>
                        <option value="robust">Robust Scaling</option>
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <button 
            onClick={previewNormalization}
            className="w-full md:w-auto bg-purple-600 text-white px-6 py-2 rounded-md hover:bg-purple-700 transition"
          >
            Preview Normalization
          </button>
        </div>
        
        {normalizationPreview && (
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">Normalization Preview</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Feature</th>
                    <th className="text-left p-2">Original Mean</th>
                    <th className="text-left p-2">Original Std</th>
                    <th className="text-left p-2">Normalized Mean</th>
                    <th className="text-left p-2">Normalized Std</th>
                  </tr>
                </thead>
                <tbody>
                  {normalizationPreview.stats?.map((stat) => (
                    <tr key={stat.feature} className="border-b">
                      <td className="p-2">{stat.feature}</td>
                      <td className="p-2">{stat.orig_mean?.toFixed(4)}</td>
                      <td className="p-2">{stat.orig_std?.toFixed(4)}</td>
                      <td className="p-2">{stat.norm_mean?.toFixed(4)}</td>
                      <td className="p-2">{stat.norm_std?.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderTargetTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Target Definition</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Target Type</label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input 
                type="radio"
                checked={target.type === 'binary'}
                onChange={() => setTarget({...target, type: 'binary'})}
              />
              <span>Binary Classification (Up/Down)</span>
            </label>
            <label className="flex items-center gap-2">
              <input 
                type="radio"
                checked={target.type === 'regression'}
                onChange={() => setTarget({...target, type: 'regression'})}
              />
              <span>Regression (Return Value)</span>
            </label>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Prediction Horizon (bars ahead)
          </label>
          <input 
            type="number"
            min="1"
            className="w-full p-2 border border-gray-300 rounded-md"
            value={target.horizon}
            onChange={(e) => setTarget({...target, horizon: parseInt(e.target.value)})}
          />
        </div>
        
        {target.type === 'binary' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Classification Threshold (%)
            </label>
            <input 
              type="number"
              step="0.1"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={target.threshold}
              onChange={(e) => setTarget({...target, threshold: parseFloat(e.target.value)})}
            />
            <p className="text-sm text-gray-500 mt-1">
              Returns above {target.threshold}% are classified as "Up", below as "Down"
            </p>
          </div>
        )}
      </div>
    </div>
  );

  const renderAlgorithmTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Algorithm Selection</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">Select Algorithm</label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {Object.entries(algorithmTypes).map(([key, val]) => (
              <label key={key} className="flex items-center gap-2 p-3 border rounded-md cursor-pointer hover:bg-gray-50">
                <input 
                  type="radio"
                  checked={algorithm.type === key}
                  onChange={() => setAlgorithm({
                    type: key,
                    params: { ...val.params },
                    trainTestSplit: algorithm.trainTestSplit
                  })}
                />
                <span>{val.label}</span>
              </label>
            ))}
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Train/Test Split Ratio
          </label>
          <input 
            type="number"
            min="0.5"
            max="0.95"
            step="0.05"
            className="w-full p-2 border border-gray-300 rounded-md"
            value={algorithm.trainTestSplit}
            onChange={(e) => setAlgorithm({...algorithm, trainTestSplit: parseFloat(e.target.value)})}
          />
          <p className="text-sm text-gray-500 mt-1">
            {(algorithm.trainTestSplit * 100).toFixed(0)}% for training, {((1 - algorithm.trainTestSplit) * 100).toFixed(0)}% for testing
          </p>
        </div>
        
        {Object.keys(algorithm.params).length > 0 && (
          <div className="border-t pt-4">
            <h3 className="font-semibold mb-3">Hyperparameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(algorithm.params).map(([param, value]) => (
                <div key={param}>
                  <label className="block text-sm font-medium text-gray-700 mb-2 capitalize">
                    {param.replace(/_/g, ' ')}
                  </label>
                  <input 
                    type={typeof value === 'string' ? 'text' : 'number'}
                    step={typeof value === 'number' && value < 1 ? '0.001' : '1'}
                    className="w-full p-2 border border-gray-300 rounded-md"
                    value={value}
                    onChange={(e) => setAlgorithm({
                      ...algorithm,
                      params: {
                        ...algorithm.params,
                        [param]: typeof value === 'string' ? e.target.value : parseFloat(e.target.value)
                      }
                    })}
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderStrategyTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Strategy Configuration</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Hold Period</label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input 
                type="radio"
                checked={strategy.holdPeriod === 'fixed'}
                onChange={() => setStrategy({...strategy, holdPeriod: 'fixed'})}
              />
              <span>Fixed (1 bar)</span>
            </label>
            <label className="flex items-center gap-2">
              <input 
                type="radio"
                checked={strategy.holdPeriod === 'signal_change'}
                onChange={() => setStrategy({...strategy, holdPeriod: 'signal_change'})}
              />
              <span>Until Signal Change</span>
            </label>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Stop Loss (%)
            </label>
            <input 
              type="number"
              step="0.1"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={strategy.stopLoss}
              onChange={(e) => setStrategy({...strategy, stopLoss: parseFloat(e.target.value)})}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Take Profit (%)
            </label>
            <input 
              type="number"
              step="0.1"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={strategy.takeProfit}
              onChange={(e) => setStrategy({...strategy, takeProfit: parseFloat(e.target.value)})}
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Position Sizing</label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input 
                type="radio"
                checked={strategy.positionSizing === 'fixed'}
                onChange={() => setStrategy({...strategy, positionSizing: 'fixed'})}
              />
              <span>Fixed</span>
            </label>
            <label className="flex items-center gap-2">
              <input 
                type="radio"
                checked={strategy.positionSizing === 'confidence'}
                onChange={() => setStrategy({...strategy, positionSizing: 'confidence'})}
              />
              <span>Confidence-Based</span>
            </label>
          </div>
        </div>
        
        {strategy.positionSizing === 'confidence' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Multiplier
            </label>
            <input 
              type="number"
              step="0.1"
              min="0.1"
              max="5"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={strategy.confidenceMultiplier}
              onChange={(e) => setStrategy({...strategy, confidenceMultiplier: parseFloat(e.target.value)})}
            />
            <p className="text-sm text-gray-500 mt-1">
              Position size = Base size × Prediction confidence × Multiplier
            </p>
          </div>
        )}
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Initial Capital ($)
          </label>
          <input 
            type="number"
            step="1000"
            className="w-full p-2 border border-gray-300 rounded-md"
            value={strategy.initialCapital}
            onChange={(e) => setStrategy({...strategy, initialCapital: parseFloat(e.target.value)})}
          />
        </div>
      </div>
    </div>
  );

  const renderEvaluationTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Evaluation</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md">
        <button 
          onClick={runBacktest}
          disabled={backtestLoading || dataSources.length === 0}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-lg hover:from-blue-700 hover:to-purple-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 text-lg font-semibold"
        >
          <Play size={24} />
          {backtestLoading ? 'Running Backtest...' : 'RUN BACKTEST'}
        </button>
      </div>
      
      {backtestLoading && (
        <div className="bg-white p-8 rounded-lg shadow-md text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Training model and running backtest...</p>
        </div>
      )}
      
      {backtestResults && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-sm text-gray-600 mb-1">Total Return</div>
              <div className={`text-3xl font-bold ${backtestResults.metrics.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {backtestResults.metrics.total_return?.toFixed(2)}%
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-sm text-gray-600 mb-1">Sharpe Ratio</div>
              <div className="text-3xl font-bold text-blue-600">
                {backtestResults.metrics.sharpe_ratio?.toFixed(2)}
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-sm text-gray-600 mb-1">Max Drawdown</div>
              <div className="text-3xl font-bold text-red-600">
                {backtestResults.metrics.max_drawdown?.toFixed(2)}%
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-sm text-gray-600 mb-1">Win Rate</div>
              <div className="text-3xl font-bold text-purple-600">
                {backtestResults.metrics.win_rate?.toFixed(1)}%
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">Equity Curve</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={backtestResults.equity_curve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="equity" stroke="#2563eb" strokeWidth={2} dot={false} name="Portfolio Value" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">Trading Signals</h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis dataKey="price" />
                <Tooltip />
                <Legend />
                <Scatter 
                  data={backtestResults.trades?.filter(t => t.type === 'buy')} 
                  fill="#10b981" 
                  name="Buy Signal"
                  shape="circle"
                />
                <Scatter 
                  data={backtestResults.trades?.filter(t => t.type === 'sell')} 
                  fill="#ef4444" 
                  name="Sell Signal"
                  shape="circle"
                />
                <Line 
                  data={backtestResults.price_data} 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#6b7280" 
                  dot={false}
                  name="Price"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">Additional Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-gray-600">Total Trades</div>
                <div className="text-xl font-semibold">{backtestResults.metrics.total_trades}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Winning Trades</div>
                <div className="text-xl font-semibold text-green-600">{backtestResults.metrics.winning_trades}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Losing Trades</div>
                <div className="text-xl font-semibold text-red-600">{backtestResults.metrics.losing_trades}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Avg Trade Return</div>
                <div className="text-xl font-semibold">{backtestResults.metrics.avg_trade_return?.toFixed(2)}%</div>
              </div>
            </div>
          </div>
          
          {backtestResults.feature_importance && (
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
              <div className="space-y-2">
                {backtestResults.feature_importance.map((item, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="w-32 text-sm text-gray-700">{item.feature}</div>
                    <div className="flex-1 bg-gray-200 rounded-full h-6">
                      <div 
                        className="bg-blue-600 h-6 rounded-full flex items-center justify-end pr-2"
                        style={{ width: `${(item.importance * 100)}%` }}
                      >
                        <span className="text-xs text-white font-medium">
                          {(item.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );

  const renderTabContent = () => {
    switch(currentTab) {
      case 0: return renderDataTab();
      case 1: return renderFeaturesTab();
      case 2: return renderNormalizationTab();
      case 3: return renderTargetTab();
      case 4: return renderAlgorithmTab();
      case 5: return renderStrategyTab();
      case 6: return renderEvaluationTab();
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">AlgoBuilder AllPick</h1>
          <p className="text-gray-600">Build, test, and optimize your trading algorithms</p>
        </div>
        
        {isMobile ? (
          <div className="mb-8">
            <div className="flex flex-col space-y-2">
              {tabs.map((tab, idx) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => setCurrentTab(idx)}
                    className={`flex items-center gap-3 p-4 rounded-lg transition ${
                      currentTab === idx
                        ? 'bg-white shadow-md border-l-4 border-blue-600'
                        : 'bg-white/50 hover:bg-white/80'
                    }`}
                  >
                    <Icon size={20} className={currentTab === idx ? 'text-blue-600' : 'text-gray-600'} />
                    <span className={`font-medium ${currentTab === idx ? 'text-blue-600' : 'text-gray-700'}`}>
                      {tab.name}
                    </span>
                    {currentTab === idx && <ChevronRight className="ml-auto text-blue-600" size={20} />}
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="mb-8">
            <div className="flex flex-wrap gap-2 justify-center">
              {tabs.map((tab, idx) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => setCurrentTab(idx)}
                    className={`flex items-center gap-2 px-6 py-3 rounded-lg transition ${
                      currentTab === idx
                        ? 'bg-white shadow-lg border-2 border-blue-600 text-blue-600'
                        : 'bg-white hover:shadow-md text-gray-700'
                    }`}
                  >
                    <Icon size={20} />
                    <span className="font-medium">{tab.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
        
        <div className="bg-white/80 backdrop-blur rounded-2xl shadow-xl p-6 md:p-8 min-h-[600px]">
          {renderTabContent()}
        </div>
        
        <div className="mt-6 flex justify-between">
          <button
            onClick={() => setCurrentTab(Math.max(0, currentTab - 1))}
            disabled={currentTab === 0}
            className="flex items-center gap-2 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft size={20} />
            Previous
          </button>
          
          <button
            onClick={() => setCurrentTab(Math.min(tabs.length - 1, currentTab + 1))}
            disabled={currentTab === tabs.length - 1}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default AlgoBuilder;
