import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Any

class ModelTrainer:
    """Handles training of various ML algorithms"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.is_classification = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray,
              algorithm_type: str, params: Dict, 
              target_type: str) -> Dict:
        """
        Train the specified algorithm
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            algorithm_type: Type of algorithm
            params: Hyperparameters
            target_type: 'binary' or 'regression'
        
        Returns:
            Dictionary with model and training results
        """
        print(f"DEBUG ModelTrainer.train(): Starting training")
        print(f"  Algorithm: {algorithm_type}")
        print(f"  Target type: {target_type}")
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"  Params: {params}")
        
        # Debug data content
        print(f"  X_train stats - min: {X_train.min():.4f}, max: {X_train.max():.4f}, mean: {X_train.mean():.4f}")
        print(f"  y_train stats - min: {y_train.min():.4f}, max: {y_train.max():.4f}, mean: {y_train.mean():.4f}")
        print(f"  y_train unique values: {np.unique(y_train)}")
        print(f"  y_train value counts: {np.bincount(y_train.astype(int)) if len(np.unique(y_train)) <= 10 else 'Too many unique values'}")
        
        self.model_type = algorithm_type
        self.is_classification = (target_type == 'binary')
        
        if algorithm_type == 'linear_regression':
            return self._train_linear_regression(X_train, y_train, X_test, y_test)
        
        elif algorithm_type == 'neural_network':
            return self._train_neural_network(X_train, y_train, X_test, y_test, params)
        
        elif algorithm_type == 'rnn':
            return self._train_rnn(X_train, y_train, X_test, y_test, params)
        
        elif algorithm_type == 'random_forest':
            return self._train_random_forest(X_train, y_train, X_test, y_test, params)
        
        elif algorithm_type == 'knn':
            return self._train_knn(X_train, y_train, X_test, y_test, params)
        
        elif algorithm_type == 'naive_bayes':
            return self._train_naive_bayes(X_train, y_train, X_test, y_test)
        
        elif algorithm_type == 'transformer':
            return self._train_transformer(X_train, y_train, X_test, y_test, params)
        
        elif algorithm_type == 'lightgbm':
            return self._train_lightgbm(X_train, y_train, X_test, y_test, params)
        
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    def _train_linear_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Linear/Logistic Regression"""
        print(f"DEBUG: Training Linear/Logistic Regression")
        print(f"  Classification: {self.is_classification}")
        
        if self.is_classification:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Debug predictions
        train_pred = self.model.predict(X_train[:5])
        test_pred = self.model.predict(X_test[:5])
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        return {
            'train_score': float(train_score),
            'test_score': float(test_score)
        }
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test, params) -> Dict:
        """Train Neural Network"""
        print(f"DEBUG: Training Neural Network")
        
        # Parse layer configuration
        layers_config = eval(params.get('layers', '[64, 32]'))
        epochs = int(params.get('epochs', 50))
        learning_rate = float(params.get('learning_rate', 0.001))
        
        print(f"  Layers: {layers_config}, Epochs: {epochs}, Learning rate: {learning_rate}")
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Dense(layers_config[0], activation='relu', input_shape=(X_train.shape[1],)))
        
        for units in layers_config[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.2))
        
        if self.is_classification:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(layers.Dense(1))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='mse',
                         metrics=['mae'])
        
        print(f"  Model architecture:")
        model.summary()
        
        # Train
        history = model.fit(X_train, y_train, 
                          epochs=epochs, 
                          batch_size=32,
                          validation_data=(X_test, y_test),
                          verbose=0)
        
        self.model = model
        
        # Debug predictions
        train_pred = self.model.predict(X_train[:5], verbose=0).flatten()
        test_pred = self.model.predict(X_test[:5], verbose=0).flatten()
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Final train loss: {history.history['loss'][-1]:.4f}, test loss: {history.history['val_loss'][-1]:.4f}")
        
        return {
            'train_score': float(history.history['loss'][-1]),
            'test_score': float(history.history['val_loss'][-1])
        }
    
    def _train_rnn(self, X_train, y_train, X_test, y_test, params) -> Dict:
        """Train RNN/LSTM"""
        print(f"DEBUG: Training RNN/LSTM")
        
        units = int(params.get('units', 64))
        epochs = int(params.get('epochs', 50))
        learning_rate = float(params.get('learning_rate', 0.001))
        
        print(f"  Units: {units}, Epochs: {epochs}, Learning rate: {learning_rate}")
        
        # Reshape for LSTM (samples, timesteps, features)
        # Using single timestep for now
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        print(f"  Reshaped X_train: {X_train_reshaped.shape}, X_test: {X_test_reshaped.shape}")
        
        # Build model
        model = keras.Sequential([
            layers.LSTM(units, input_shape=(1, X_train.shape[1]), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units // 2),
            layers.Dropout(0.2)
        ])
        
        if self.is_classification:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(layers.Dense(1))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='mse',
                         metrics=['mae'])
        
        print(f"  Model architecture:")
        model.summary()
        
        # Train
        history = model.fit(X_train_reshaped, y_train,
                          epochs=epochs,
                          batch_size=32,
                          validation_data=(X_test_reshaped, y_test),
                          verbose=0)
        
        self.model = model
        self.is_sequence = True
        
        # Debug predictions
        train_pred = self.model.predict(X_train_reshaped[:5], verbose=0).flatten()
        test_pred = self.model.predict(X_test_reshaped[:5], verbose=0).flatten()
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Final train loss: {history.history['loss'][-1]:.4f}, test loss: {history.history['val_loss'][-1]:.4f}")
        
        return {
            'train_score': float(history.history['loss'][-1]),
            'test_score': float(history.history['val_loss'][-1])
        }
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test, params) -> Dict:
        """Train Random Forest"""
        print(f"DEBUG: Training Random Forest")
        
        n_estimators = int(params.get('n_estimators', 100))
        max_depth = int(params.get('max_depth', 10))
        
        print(f"  n_estimators: {n_estimators}, max_depth: {max_depth}")
        
        if self.is_classification:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Debug predictions
        train_pred = self.model.predict(X_train[:5])
        test_pred = self.model.predict(X_test[:5])
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        result = {
            'train_score': float(train_score),
            'test_score': float(test_score)
        }
        
        if hasattr(self.model, 'feature_importances_'):
            result['feature_importance'] = self.model.feature_importances_.tolist()
            print(f"  Feature importance range: {min(self.model.feature_importances_):.4f} to {max(self.model.feature_importances_):.4f}")
        
        return result
    
    def _train_knn(self, X_train, y_train, X_test, y_test, params) -> Dict:
        """Train K-Nearest Neighbors"""
        print(f"DEBUG: Training KNN")
        
        n_neighbors = int(params.get('n_neighbors', 5))
        print(f"  n_neighbors: {n_neighbors}")
        
        if self.is_classification:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        else:
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Debug predictions
        train_pred = self.model.predict(X_train[:5])
        test_pred = self.model.predict(X_test[:5])
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        return {
            'train_score': float(train_score),
            'test_score': float(test_score)
        }
    
    def _train_naive_bayes(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Naive Bayes (classification only)"""
        print(f"DEBUG: Training Naive Bayes")
        
        if not self.is_classification:
            raise ValueError("Naive Bayes only supports classification")
        
        self.model = GaussianNB()
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Debug predictions
        train_pred = self.model.predict(X_train[:5])
        test_pred = self.model.predict(X_test[:5])
        train_proba = self.model.predict_proba(X_train[:5])
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Sample train probabilities: {train_proba}")
        print(f"  Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        return {
            'train_score': float(train_score),
            'test_score': float(test_score)
        }
    
    def _train_transformer(self, X_train, y_train, X_test, y_test, params) -> Dict:
        """Train Transformer model"""
        print(f"DEBUG: Training Transformer")
        
        d_model = int(params.get('d_model', 64))
        n_heads = int(params.get('n_heads', 4))
        n_layers = int(params.get('n_layers', 2))
        epochs = int(params.get('epochs', 50))
        
        print(f"  d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}, epochs: {epochs}")
        
        # Reshape for transformer
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        print(f"  Reshaped X_train: {X_train_reshaped.shape}, X_test: {X_test_reshaped.shape}")
        
        # Build transformer model
        inputs = layers.Input(shape=(1, X_train.shape[1]))
        x = layers.Dense(d_model)(inputs)
        
        for _ in range(n_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model // n_heads
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)
            
            # Feed forward
            ff = layers.Dense(d_model * 2, activation='relu')(x)
            ff = layers.Dense(d_model)(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        
        if self.is_classification:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            outputs = layers.Dense(1)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"  Model architecture:")
        model.summary()
        
        history = model.fit(X_train_reshaped, y_train,
                          epochs=epochs,
                          batch_size=32,
                          validation_data=(X_test_reshaped, y_test),
                          verbose=0)
        
        self.model = model
        self.is_sequence = True
        
        # Debug predictions
        train_pred = self.model.predict(X_train_reshaped[:5], verbose=0).flatten()
        test_pred = self.model.predict(X_test_reshaped[:5], verbose=0).flatten()
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Final train loss: {history.history['loss'][-1]:.4f}, test loss: {history.history['val_loss'][-1]:.4f}")
        
        return {
            'train_score': float(history.history['loss'][-1]),
            'test_score': float(history.history['val_loss'][-1])
        }
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test, params) -> Dict:
        """Train LightGBM"""
        print(f"DEBUG: Training LightGBM")
        
        n_estimators = int(params.get('n_estimators', 100))
        max_depth = int(params.get('max_depth', 10))
        learning_rate = float(params.get('learning_rate', 0.1))
        
        print(f"  n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}")
        
        if self.is_classification:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                verbose=-1
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                verbose=-1
            )
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Debug predictions
        train_pred = self.model.predict(X_train[:5])
        test_pred = self.model.predict(X_test[:5])
        print(f"  Sample train predictions: {train_pred}")
        print(f"  Sample test predictions: {test_pred}")
        print(f"  Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        result = {
            'train_score': float(train_score),
            'test_score': float(test_score)
        }
        
        if hasattr(self.model, 'feature_importances_'):
            result['feature_importance'] = self.model.feature_importances_.tolist()
            print(f"  Feature importance range: {min(self.model.feature_importances_):.4f} to {max(self.model.feature_importances_):.4f}")
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        print(f"DEBUG ModelTrainer.predict(): Making predictions")
        print(f"  Input X shape: {X.shape}")
        
        if hasattr(self, 'is_sequence') and self.is_sequence:
            X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
            print(f"  Reshaped for sequence model: {X_reshaped.shape}")
        else:
            X_reshaped = X
        
        if isinstance(self.model, keras.Model):
            predictions = self.model.predict(X_reshaped, verbose=0)
            predictions = predictions.flatten()
        else:
            predictions = self.model.predict(X_reshaped)
        
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Predictions stats - min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}")
        print(f"  First 10 predictions: {predictions[:10]}")
        print(f"  Unique prediction values: {np.unique(predictions)}")
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (for classification)"""
        print(f"DEBUG ModelTrainer.predict_proba(): Getting probabilities")
        print(f"  Input X shape: {X.shape}")
        
        if not self.is_classification:
            print("  Not a classification model, using predict()")
            return self.predict(X)
        
        if hasattr(self, 'is_sequence') and self.is_sequence:
            X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
            print(f"  Reshaped for sequence model: {X_reshaped.shape}")
        else:
            X_reshaped = X
        
        if isinstance(self.model, keras.Model):
            probabilities = self.model.predict(X_reshaped, verbose=0).flatten()
        elif hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_reshaped)
            probabilities = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
        else:
            probabilities = self.predict(X_reshaped)
        
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Probabilities stats - min: {probabilities.min():.4f}, max: {probabilities.max():.4f}, mean: {probabilities.mean():.4f}")
        print(f"  First 10 probabilities: {probabilities[:10]}")
        
        return probabilities
