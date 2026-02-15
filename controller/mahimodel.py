import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import os

# Setup logging
logger = logging.getLogger(__name__)


class LiquidityPredictor:
    

    def __init__(self, 
                model_path: str = 'liquidity_model.pkl',
                scaler_path: str = 'scaler.pkl', 
                features_path: str = 'feature_columns.pkl'):
        
        logger.info("Loading liquidity stress prediction models...")

        try:
            # Get absolute directory of THIS file
            base_dir = os.path.dirname(os.path.abspath(__file__))

            # Build absolute paths
            model_path = os.path.join(base_dir, model_path)
            scaler_path = os.path.join(base_dir, scaler_path)
            features_path = os.path.join(base_dir, features_path)

            # Load models
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.required_features = joblib.load(features_path)

            logger.info(f"Model loaded: {self.model.n_estimators} trees")
            logger.info(f"Required features: {len(self.required_features)}")

            self.model_path = model_path
            self.loaded_at = datetime.now()

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Could not load models: {e}")

    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        
        start_time = datetime.now()
        
        try:
            if not isinstance(features, dict):
                raise ValueError(f"Features must be dict, got {type(features)}")
            
            if "peg_deviation_mean_3h" in features:
                features["peg_deviation_mean_180m"] = features["peg_deviation_mean_3h"]

            if "peg_deviation_std_3h" in features:
                features["peg_deviation_std_180m"] = features["peg_deviation_std_3h"]

            if "peg_deviation_mean_12h" in features:
                features["peg_deviation_mean_720m"] = features["peg_deviation_mean_12h"]

            if "peg_deviation_std_12h" in features:
                features["peg_deviation_std_720m"] = features["peg_deviation_std_12h"]
                
            missing = [f for f in self.required_features if f not in features]

            if missing:
                logger.warning(f"Missing {len(missing)} features, filling with 0: {missing[:5]}...")
                for feat in missing:
                    features[feat] = 0.0
           
            X = pd.DataFrame([features])[self.required_features]
           
            if X.isna().any().any():
                logger.warning("NaN values detected, filling with 0")
                X = X.fillna(0)
           
            X_scaled = self.scaler.transform(X)
          
            prediction = float(self.model.predict(X_scaled)[0])
           
            prediction_pct = prediction * 100
           
            if prediction < 0.005:
                risk_level = "MINIMAL"
                risk_score = 1
                alert = False
            elif prediction < 0.01:
                risk_level = "LOW"
                risk_score = 2
                alert = False
            elif prediction < 0.015:
                risk_level = "MODERATE"
                risk_score = 3
                alert = False
            elif prediction < 0.02:
                risk_level = "HIGH"
                risk_score = 4
                alert = True
            else:
                risk_level = "CRITICAL"
                risk_score = 5
                alert = True
           
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
           
            result = {
                'max_peg_deviation': round(prediction, 6),
                'max_peg_deviation_pct': round(prediction_pct, 4),
                'risk_level': risk_level,
                'risk_score': risk_score,
                'alert_required': alert,
                'timestamp': datetime.now().isoformat(),
                'inference_time_ms': round(inference_time, 2)
            }
            
            logger.debug(f"Prediction: {prediction_pct:.3f}% ({risk_level}) in {inference_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, features_list: list) -> list:
        
        results = []
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        
        return results
    
    def get_required_features(self) -> list:
       
        return self.required_features.copy()
    
    def get_info(self) -> dict:
       
        return {
            'model_type': 'XGBoost Regressor',
            'n_trees': self.model.n_estimators,
            'required_features': len(self.required_features),
            'model_path': self.model_path,
            'loaded_at': self.loaded_at.isoformat()
        }

_global_predictor = None

def get_predictor() -> LiquidityPredictor:
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = LiquidityPredictor()
    return _global_predictor




def predict_liquidity_stress(features: Dict[str, float]) -> Dict[str, Any]:
 
    predictor = get_predictor()
    return predictor.predict(features)

if __name__ == "__main__":
    
    print("="*80)
    print("LIQUIDITY STRESS PREDICTOR - DEMO")
    print("="*80)
    
    # Method 1: Direct instantiation
    print("\n1. Direct instantiation:")
    predictor = LiquidityPredictor()
    print(f"   Model info: {predictor.get_info()}")
    
    # Create sample features
    sample_features = {
        # Required 21 features
        "percent_change_1h": -0.0005,
        "percent_change_24h": -0.0012,
        "percent_change_7d": 0.0003,
        "volume_percent_change_24h": 15.2,
        "volume_vs_24h": 1.15,
        "volume_vs_7d": 0.98,
        "extreme_volume_spike": 0,
        "taker_buy_ratio": 0.52,
        "volume_imbalance": 0.04,
        "large_trade_anomaly": 0,
        "peg_deviation_mean_180m": 0.0008,
        "peg_deviation_std_180m": 0.0003,
        "peg_deviation_mean_720m": 0.0006,
        "peg_deviation_std_720m": 0.0002,
        "peg_stress_1pct_3h": 0,
        "peg_stress_2pct_3h": 0,
        "price_accel_1h_24h": 0.0007,
        "price_accel_24h_7d": -0.0015,
        "market_cap_percent_change_1d": -0.0008,
        "circulating_supply_percent_change_1d": 0.0001,
        "mcap_to_volume_24h": 125.3,
        
        # Additional 150 dummy features (will be ignored)
        **{f"feature_{i}": np.random.randn() for i in range(150)}
    }
    
    # Predict
    result = predictor.predict(sample_features)
    
    print("\n2. Prediction result:")
    print(f"   Max Deviation: {result['max_peg_deviation_pct']:.2f}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Alert Required: {result['alert_required']}")
    print(f"   Inference Time: {result['inference_time_ms']:.1f}ms")
    
    # Method 2: Convenience function
    print("\n3. Using convenience function:")
    result2 = predict_liquidity_stress(sample_features)
    print(f"   Risk: {result2['risk_level']}")
    
    # Method 3: Batch prediction
    print("\n4. Batch prediction:")
    batch_features = [sample_features] * 5
    batch_results = predictor.predict_batch(batch_features)
    print(f"   Predicted {len(batch_results)} samples")
    
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE")
    print("="*80)
    print("""
# In your existing backend server:

from liquidity_predictor import LiquidityPredictor

# Initialize once at server startup
predictor = LiquidityPredictor()

# In your data processing function (called every minute):
def process_new_data():
    # 1. Fetch raw data from API
    raw_data = fetch_from_binance_gecko()
    
    # 2. Compute your 171 features (you already have this)
    features_171 = compute_all_features(raw_data)
    
    # 3. Predict (fast! ~10ms)
    result = predictor.predict(features_171)
    
    # 4. Use result
    if result['alert_required']:
        send_alert(f"Risk: {result['risk_level']}")
    
    # 5. Store to database
    db.save({
        'timestamp': result['timestamp'],
        'risk_level': result['risk_level'],
        'deviation': result['max_peg_deviation_pct']
    })
    
    return result
    """)