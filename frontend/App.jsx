import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { ChevronRight, ChevronLeft, Play, Plus, Trash2, TrendingUp, Activity, BarChart3, Database, Settings, Target, CheckCircle } from 'lucide-react';

const API_URL = 'http://localhost:8000/api';

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
              className="w-full p-2 border border-gray-300 
