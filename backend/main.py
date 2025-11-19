from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from normalizer import DataNormalizer
from model_trainer import ModelTrainer
from backtester import Backtester

app = FastAPI(title="AlgoBuilder AllPick API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for session data
session_data = {}

# Pydantic models for request/response
class DataSourceRequest(BaseModel):
    api: str
    symbol: str
    timeframe: str
    startDate: str
    endDate: str

class FeatureRequest(BaseModel):
    type: str
    params: Dict[str, Any]

class NormalizationPreviewRequest(BaseModel):
    normalization: Dict[str, str]
    features: List[Dict[str, Any]]

class FeaturePreviewRequest(BaseModel):
    dataSource: Dict[str, Any]
    features: List[Dict[str, Any]]
    baseFeatures: Optional[Dict[str, bool]] = None

class BacktestRequest(BaseModel):
    dataSources: List[Dict[str, Any]]
    features: List[Dict[str, Any]]
    baseFeatures: Optional[Dict[str, bool]] = None
    normalization: Dict[str, str]
    target: Dict[str, Any]
    algorithm: Dict[str, Any]
    strategy: Dict[str, Any]

@app.get("/")
async def root():
    return {"message": "AlgoBuilder AllPick API", "version": "1.0"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/data/fetch")
async def fetch_data(request: DataSourceRequest):
    """Fetch market data from specified API"""
    try:
        fetcher = DataFetcher()
        
        df = fetcher.fetch_data(
            api=request.api,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.startDate,
            end_date=request.endDate
        )
        
        # Store data in session (in production, use proper session management)
        session_key = f"{request.symbol}_{request.timeframe}_{request.startDate}"
        session_data[session_key] = df
        
        # Get preview
        preview = fetcher.get_preview(df)
        
        return {
            "success": True,
            "preview": preview,
            "session_key": session_key
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/features/calculate")
async def calculate_features(features: List[Dict[str, Any]], session_key: str):
    """Calculate technical indicators and features"""
    try:
        if session_key not in session_data:
            raise ValueError("Data not found. Please fetch data first.")
        
        df = session_data[session_key].copy()
        engineer = FeatureEngineer(df)
        
        # Add each feature
        for feature in features:
            engineer.add_technical_indicator(
                feature_type=feature['type'],
                params=feature['params']
            )
        
        # Store updated dataframe
        session_data[session_key] = engineer.df
        
        return {
            "success": True,
            "feature_columns": engineer.get_feature_columns()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/features/preview")
async def preview_features(request: FeaturePreviewRequest):
    """Preview feature values and statistics"""
    try:
        # Fetch data
        fetcher = DataFetcher()
        df = fetcher.fetch_data(
            api=request.dataSource['api'],
            symbol=request.dataSource['symbol'],
            timeframe=request.dataSource['timeframe'],
            start_date=request.dataSource['startDate'],
            end_date=request.dataSource['endDate']
        )
        
        # Get selected base features
        selected_base = None
        if request.baseFeatures:
            selected_base = [k for k, v in request.baseFeatures.items() if v]
        
        # Create feature engineer with selected base features
        engineer = FeatureEngineer(df, selected_base_features=selected_base)
        
        # Add technical indicators
        for feature in request.features:
            engineer.add_technical_indicator(
                feature_type=feature['type'],
                params=feature['params']
            )
        
        # Get feature statistics
        stats = engineer.get_feature_stats()
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/features/normalize-preview")
async def normalize_preview(request: NormalizationPreviewRequest):
    """Preview normalization effects"""
    try:
        # Create sample dataframe for preview
        # In production, would use actual session data
        sample_data = {
            'Open': np.random.randn(100) * 10 + 100,
            'High': np.random.randn(100) * 10 + 105,
            'Low': np.random.randn(100) * 10 + 95,
            'Close': np.random.randn(100) * 10 + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }
        
        # Add feature columns
        for feature in request.features:
            sample_data[feature['label']] = np.random.randn(100) * 5 + 50
        
        df = pd.DataFrame(sample_data)
        
        normalizer = DataNormalizer()
        feature_columns = list(df.columns)
        
        preview = normalizer.get_normalization_preview(
            df=df,
            normalization_config=request.normalization,
            feature_columns=feature_columns
        )
        
        return preview
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run complete backtest pipeline"""
    try:
        # Step 1: Fetch and combine data
        fetcher = DataFetcher()
        dfs = []
        
        for source in request.dataSources:
            df = fetcher.fetch_data(
                api=source['api'],
                symbol=source['symbol'],
                timeframe=source['timeframe'],
                start_date=source['startDate'],
                end_date=source['endDate']
            )
            dfs.append(df)
        
        # Use first dataframe (in production, might merge multiple sources)
        df = dfs[0]
        
        # Step 2: Feature engineering
        # Get selected base features
        selected_base = None
        if request.baseFeatures:
            selected_base = [k for k, v in request.baseFeatures.items() if v]
        
        engineer = FeatureEngineer(df, selected_base_features=selected_base)
        
        for feature in request.features:
            engineer.add_technical_indicator(
                feature_type=feature['type'],
                params=feature['params']
            )
        
        # Add polynomial features if specified
        # Add derivatives if specified
        # Add feature multiplications if specified
        
        # Create target
        engineer.create_target(
            target_type=request.target['type'],
            horizon=request.target['horizon'],
            threshold=request.target.get('threshold', 0.5)
        )
        
        # Clean data
        engineer.clean_data()
        
        # Step 3: Normalization
        normalizer = DataNormalizer()
        feature_columns = engineer.get_feature_columns()
        
        df_normalized = normalizer.normalize(
            df=engineer.df,
            normalization_config=request.normalization,
            feature_columns=feature_columns
        )
        
        # Step 4: Split data
        train_df, test_df = normalizer.split_train_test(
            df_normalized,
            split_ratio=request.algorithm['trainTestSplit']
        )
        
        # Prepare features and target
        X_train = train_df[feature_columns].values
        y_train = train_df['target'].values
        X_test = test_df[feature_columns].values
        y_test = test_df['target'].values
        
        # Step 5: Train model
        trainer = ModelTrainer()
        training_results = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            algorithm_type=request.algorithm['type'],
            params=request.algorithm['params'],
            target_type=request.target['type']
        )
        
        # Step 6: Make predictions
        predictions = trainer.predict(X_test)
        probabilities = trainer.predict_proba(X_test)
        
        # Convert predictions to binary if needed
        if request.target['type'] == 'binary':
            predictions = (predictions > 0.5).astype(int)
        else:
            # For regression, convert to binary signals
            predictions = (predictions > 0).astype(int)
        
        # Step 7: Run backtest
        backtester = Backtester(
            initial_capital=request.strategy['initialCapital']
        )
        
        backtest_results = backtester.run_backtest(
            df=test_df,
            predictions=predictions,
            probabilities=probabilities,
            strategy_config=request.strategy,
            train_df=train_df  # Pass training data for equity curve
        )
        
        # Add feature importance if available
        if 'feature_importance' in training_results:
            importance = training_results['feature_importance']
            feature_importance = [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in zip(feature_columns, importance)
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            backtest_results['feature_importance'] = feature_importance[:10]  # Top 10
        
        return backtest_results
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
