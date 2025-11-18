# AlgoBuilder AllPick

A comprehensive web-based platform for building, testing, and optimizing trading algorithms using machine learning.

## Features

- **Multi-Source Data Fetching**: Support for Yahoo Finance and Binance APIs with automatic chunking for rate limit compliance
- **Advanced Feature Engineering**: 
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic)
  - Polynomial features
  - Feature-feature multiplication
  - Derivatives (1st and 2nd order)
- **Flexible Normalization**: Min-Max, Standard (Z-score), and Robust scaling with preview
- **Multiple ML Algorithms**:
  - Linear Regression / Logistic Regression
  - Neural Networks (customizable architecture)
  - RNN/LSTM
  - Random Forest
  - K-Nearest Neighbors
  - Naive Bayes
  - Transformer
  - LightGBM
- **Strategy Configuration**:
  - Hold period (fixed or signal-based)
  - Stop loss and take profit
  - Position sizing (fixed or confidence-based)
- **Comprehensive Evaluation**:
  - Equity curve visualization
  - Performance metrics (Total Return, Sharpe Ratio, Max Drawdown, Win Rate)
  - Trade log with entry/exit markers
  - Feature importance analysis

## Project Structure

```
algo-builder/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   └── ...
│   ├── package.json
│   └── ...
├── backend/                  # Python FastAPI backend
│   ├── main.py              # FastAPI application
│   ├── data_fetcher.py      # Data fetching with rate limit handling
│   ├── feature_engineer.py  # Feature engineering service
│   ├── normalizer.py        # Data normalization
│   ├── model_trainer.py     # ML model training
│   ├── backtester.py        # Backtesting engine
│   └── requirements.txt     # Python dependencies
├── railway.json             # Railway deployment config
└── README.md
```

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000`

## Deployment to Railway

### Option 1: Monorepo Deployment

1. Push your entire project to GitHub
2. Connect Railway to your GitHub repository
3. Railway will automatically detect both frontend and backend
4. Configure environment variables if needed

### Option 2: Separate Services

1. Create two Railway projects: one for frontend, one for backend
2. Deploy backend:
   - Select the backend folder
   - Railway will detect Python and install dependencies
   - Set start command: `python main.py`
3. Deploy frontend:
   - Select the frontend folder
   - Railway will detect Node.js
   - Update `API_URL` in frontend to point to backend Railway URL

### Environment Variables

Backend:
- No special environment variables needed for basic operation
- For production, consider adding API keys for data sources if required

Frontend:
- `REACT_APP_API_URL`: Backend API URL (set to Railway backend URL in production)

## Usage Guide

### 1. Data Selection
- Choose between Yahoo Finance or Binance
- Enter symbol (e.g., AAPL, BTCUSDT)
- Select timeframe (1m to 1w)
- Set date range
- Click "Add Data Source"

### 2. Feature Selection
- Add technical indicators with custom parameters
- Configure polynomial degree for feature expansion
- Enable derivatives for rate of change analysis
- Selected features will be used for model training

### 3. Normalization
- Select normalization method for each feature
- Preview normalization effects before proceeding
- Different features can use different methods

### 4. Target Definition
- Choose Binary Classification (Up/Down) or Regression (Return value)
- Set prediction horizon (how many bars ahead)
- For binary: set threshold percentage

### 5. Algorithm Selection
- Choose one algorithm from the list
- Configure hyperparameters
- Set train/test split ratio

### 6. Strategy Configuration
- Set hold period (fixed 1 bar or until signal change)
- Configure stop loss and take profit percentages
- Choose position sizing method
- Set initial capital

### 7. Evaluation
- Click "RUN BACKTEST" to execute the complete pipeline
- View performance metrics
- Analyze equity curve and trading signals
- Review feature importance (if applicable)

## API Endpoints

### Data Endpoints
- `POST /api/data/fetch` - Fetch market data
- `POST /api/features/calculate` - Calculate features
- `POST /api/features/normalize-preview` - Preview normalization

### Backtest Endpoint
- `POST /api/backtest/run` - Run complete backtest pipeline

### Health Check
- `GET /api/health` - Check API health status

## Technical Details

### Rate Limit Handling
The system automatically handles Binance's 1000 bar limit per request by:
- Calculating milliseconds per candle based on timeframe
- Fetching data in chunks
- Respecting rate limits between requests
- Combining chunks into complete dataset

### Model Training
All models are trained using train/test split to prevent overfitting. Deep learning models use early stopping and dropout for regularization.

### Backtesting Engine
The backtesting engine simulates realistic trading with:
- Stop loss and take profit execution
- Position sizing based on capital
- Confidence-based position sizing option
- Comprehensive trade tracking

## Performance Considerations

- Data fetching may take time for large date ranges (especially with Binance chunking)
- Model training duration varies by algorithm complexity
- Transformer and deep learning models are most computationally intensive
- Consider using simpler models for initial exploration

## Future Enhancements

- [ ] User authentication and saved configurations
- [ ] Database for persistent storage
- [ ] Real-time trading integration
- [ ] Advanced portfolio optimization
- [ ] Custom indicator creation UI
- [ ] Multi-asset portfolio backtesting
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - feel free to use this project for your trading research and development.

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and never risk more than you can afford to lose.
