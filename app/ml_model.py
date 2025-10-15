import asyncio
import pickle
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import structlog
from pathlib import Path
import joblib

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .config import settings, Collections
from .database import get_collection
from .models import TrainingDataPoint, RiskSummary
from .security import log_to_loki

logger = structlog.get_logger()

class LiquidationPredictor:
    """XGBoost-based liquidation probability predictor"""
    
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = [
            'health_ratio',
            'total_supplied_usd',
            'total_borrowed_usd',
            'volatility_score',
            'liquidity_score',
            'concentration_score',
            'gas_price_gwei',
            'market_fear_greed_index',
            'btc_price_change_24h',
            'eth_price_change_24h'
        ]
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        self.is_trained = False
        self.last_training = None
        self.model_metrics = {}
    
    async def load_or_train_model(self) -> bool:
        """Load existing model or train new one if needed"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, skipping model training")
            return False
        
        try:
            # Try to load existing model
            if await self._load_model():
                logger.info("Loaded existing ML model")
                return True
            
            # Train new model if no existing model or it's too old
            logger.info("Training new ML model")
            return await self.train_model()
            
        except Exception as e:
            logger.error("Error loading or training model", error=str(e))
            return False
    
    async def _load_model(self) -> bool:
        """Load model from disk"""
        try:
            model_file = self.model_path / "liquidation_predictor.pkl"
            scaler_file = self.model_path / "feature_scaler.pkl"
            metadata_file = self.model_path / "model_metadata.pkl"
            
            if not all(f.exists() for f in [model_file, scaler_file, metadata_file]):
                return False
            
            # Check if model is too old
            model_age = datetime.utcnow() - datetime.fromtimestamp(model_file.stat().st_mtime)
            if model_age > timedelta(hours=settings.MODEL_RETRAIN_HOURS):
                logger.info(f"Model is {model_age.total_seconds()/3600:.1f} hours old, needs retraining")
                return False
            
            # Load model components
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.model_metrics = metadata.get('metrics', {})
                self.last_training = metadata.get('last_training')
            
            self.is_trained = True
            logger.info("Model loaded successfully", 
                       last_training=self.last_training,
                       metrics=self.model_metrics)
            
            return True
            
        except Exception as e:
            logger.error("Error loading model", error=str(e))
            return False
    
    async def _save_model(self):
        """Save model to disk"""
        try:
            model_file = self.model_path / "liquidation_predictor.pkl"
            scaler_file = self.model_path / "feature_scaler.pkl"
            metadata_file = self.model_path / "model_metadata.pkl"
            
            # Save model components
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            # Save metadata
            metadata = {
                'metrics': self.model_metrics,
                'last_training': self.last_training,
                'feature_columns': self.feature_columns
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error("Error saving model", error=str(e))
    
    async def prepare_training_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Prepare training data from database"""
        try:
            training_collection = get_collection(Collections.TRAINING_DATA)
            
            # Get training data from last 30 days
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            
            cursor = training_collection.find({
                "timestamp": {"$gte": cutoff_time}
            })
            
            training_docs = await cursor.to_list(length=None)
            
            if len(training_docs) < settings.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient training data: {len(training_docs)} samples, need {settings.MIN_TRAINING_SAMPLES}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(training_docs)
            
            # Fill missing values
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0  # Default value for missing features
                else:
                    df[col] = df[col].fillna(df[col].median())
            
            # Prepare features and target
            X = df[self.feature_columns]
            y = df['liquidated_within_24h'].astype(int)
            
            logger.info(f"Prepared training data: {len(X)} samples, {len(self.feature_columns)} features")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            logger.error("Error preparing training data", error=str(e))
            return None
    
    async def train_model(self) -> bool:
        """Train XGBoost model"""
        if not ML_AVAILABLE:
            return False
        
        try:
            # Prepare training data
            data = await self.prepare_training_data()
            if data is None:
                return False
            
            X, y = data
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
            # Fit model
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc = 0.0  # Handle case where all samples are one class
            
            self.model_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.last_training = datetime.utcnow()
            self.is_trained = True
            
            logger.info("Model training completed", metrics=self.model_metrics)
            
            # Save model
            await self._save_model()
            
            # Log to LokiAI
            await log_to_loki("ml_model_trained", {
                "metrics": self.model_metrics,
                "training_samples": len(X_train),
                "features": self.feature_columns
            })
            
            return True
            
        except Exception as e:
            logger.error("Error training model", error=str(e))
            return False
    
    def predict_liquidation_probability(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict liquidation probability for given features
        Returns: (24h_probability, 7d_probability)
        """
        if not self.is_trained or not ML_AVAILABLE:
            return 0.0, 0.0
        
        try:
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            # Scale features
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_array)
            
            # Predict probability
            prob_24h = self.model.predict_proba(feature_scaled)[0, 1]
            
            # Approximate 7-day probability (simplified approach)
            # In reality, this would need a separate model or more sophisticated approach
            prob_7d = min(1.0, prob_24h * 3.5)  # Rough approximation
            
            return float(prob_24h), float(prob_7d)
            
        except Exception as e:
            logger.error("Error predicting liquidation probability", error=str(e))
            return 0.0, 0.0
    
    async def enhance_risk_summary(self, risk_summary: RiskSummary) -> RiskSummary:
        """Enhance risk summary with ML predictions"""
        if not self.is_trained:
            return risk_summary
        
        try:
            # Extract features for prediction
            features = {
                'health_ratio': risk_summary.health_ratio or 10.0,
                'total_supplied_usd': risk_summary.total_portfolio_value_usd,
                'total_borrowed_usd': risk_summary.total_debt_usd,
                'volatility_score': risk_summary.volatility_risk / 100.0,
                'liquidity_score': 1.0 - (risk_summary.liquidity_risk / 100.0),
                'concentration_score': risk_summary.concentration_risk / 100.0,
                'gas_price_gwei': 25.0,  # Would fetch current gas price
                'market_fear_greed_index': 50.0,  # Would fetch from Fear & Greed API
                'btc_price_change_24h': 0.0,  # Would fetch from price API
                'eth_price_change_24h': 0.0,  # Would fetch from price API
            }
            
            # Get predictions
            prob_24h, prob_7d = self.predict_liquidation_probability(features)
            
            # Update risk summary
            risk_summary.predicted_liquidation_prob_24h = prob_24h
            risk_summary.predicted_liquidation_prob_7d = prob_7d
            
            logger.debug(f"ML predictions for {risk_summary.wallet_address}: 24h={prob_24h:.3f}, 7d={prob_7d:.3f}")
            
            return risk_summary
            
        except Exception as e:
            logger.error("Error enhancing risk summary with ML", error=str(e))
            return risk_summary
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {}
        for i, col in enumerate(self.feature_columns):
            importance_dict[col] = float(self.model.feature_importances_[i])
        
        return importance_dict
    
    async def update_training_data_with_liquidations(self):
        """Update training data with actual liquidation events"""
        try:
            # This would be called periodically to update training data
            # with actual liquidation events observed from the blockchain
            
            # For now, this is a placeholder for the logic that would:
            # 1. Monitor for liquidation events on supported protocols
            # 2. Update existing training data points with actual outcomes
            # 3. Create new training samples from recent liquidations
            
            training_collection = get_collection(Collections.TRAINING_DATA)
            
            # Example: mark liquidations based on health ratio thresholds
            # In production, this would use actual liquidation event data
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            await training_collection.update_many(
                {
                    "timestamp": {"$gte": cutoff_time},
                    "health_ratio": {"$lt": 1.0},
                    "liquidated_within_24h": False
                },
                {
                    "$set": {
                        "liquidated_within_24h": True,
                        "liquidation_occurred_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info("Updated training data with liquidation events")
            
        except Exception as e:
            logger.error("Error updating training data", error=str(e))

# Global ML predictor instance
ml_predictor = LiquidationPredictor()

# Helper function for easy access
async def get_liquidation_predictions(features: Dict[str, float]) -> Tuple[float, float]:
    """Get liquidation probability predictions"""
    return ml_predictor.predict_liquidation_probability(features)

async def enhance_risk_with_ml(risk_summary: RiskSummary) -> RiskSummary:
    """Enhance risk summary with ML predictions"""
    return await ml_predictor.enhance_risk_summary(risk_summary)