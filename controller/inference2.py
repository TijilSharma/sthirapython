
"""
DEPEG CLASSIFIER - INFERENCE MODULE (UPDATED)
==============================================

Hybrid stacked ensemble for predicting depeg events in next 24h.
Updated to handle JSON input format from real-time data pipeline.

Usage:
------
from depeg_predictor_updated import DepegPredictor

# Initialize once (loads models)
predictor = DepegPredictor()

# Use with JSON data
result = predictor.predict_from_json(json_data)
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logger = logging.getLogger(__name__)


import os

class DepegPredictor:

    def __init__(self,
                 rf_model_path: str = None,
                 xgb_model_path: str = None,
                 meta_model_path: str = None,
                 config_path: str = None):

        base_dir = os.path.dirname(os.path.abspath(__file__))

        rf_model_path = rf_model_path or os.path.join(base_dir, 'final_depeg_model_rf.pkl')
        xgb_model_path = xgb_model_path or os.path.join(base_dir, 'final_depeg_model_xgb.pkl')
        meta_model_path = meta_model_path or os.path.join(base_dir, 'final_depeg_model_meta.pkl')
        config_path = config_path or os.path.join(base_dir, 'final_depeg_model_config.pkl')
        """
        Initialize predictor (loads all models)
        
        Parameters:
        -----------
        rf_model_path : str
            Path to Random Forest model
        xgb_model_path : str
            Path to XGBoost model
        meta_model_path : str
            Path to meta-learner (Logistic Regression)
        config_path : str
            Path to configuration (features, threshold, etc.)
        """
        
        logger.info("Loading depeg prediction models...")
        
        try:
            # Load models
            self.rf_model = joblib.load(rf_model_path)
            self.xgb_model = joblib.load(xgb_model_path)
            self.meta_model = joblib.load(meta_model_path)
            
            # Load configuration
            config = joblib.load(config_path)
            self.feature_cols = config['feature_cols']
            self.optimal_threshold = config['optimal_threshold']
            self.target_precision = config.get('target_precision', 0.90)
            self.min_recall = config.get('min_recall', 0.30)
            self.feature_name_map = config.get('feature_name_map', {})
            
            logger.info(f"✅ RF Model loaded: {self.rf_model.n_estimators} trees")
            logger.info(f"✅ XGB Model loaded: {self.xgb_model.n_estimators} trees")
            logger.info(f"✅ Meta Model loaded (threshold: {self.optimal_threshold:.4f})")
            logger.info(f"✅ Required features: {len(self.feature_cols)}")
            
            # Store for metadata
            self.loaded_at = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise RuntimeError(f"Could not load models: {e}")
    
    def _map_hour_to_minute_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map hour-based feature names (3h, 6h, 12h, 24h) to minute-based names
        that the model expects (180m, 360m, 720m, 1440m)
        """
        mapped = features.copy()
        
        for hour_name, minute_name in self.feature_name_map.items():
            if hour_name in features:
                mapped[minute_name] = features[hour_name]
                # Remove the hour-based name to avoid confusion
                if hour_name != minute_name:
                    mapped.pop(hour_name, None)
        
        return mapped
    
    def _clean_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Clean and convert features to proper numeric types
        Handles: None, NaN, empty strings, null
        """
        cleaned = {}
        for key, value in features.items():
            # Handle various null/empty representations
            if value is None or value == "" or value == "null":
                cleaned[key] = 0.0
            elif isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = float(value)
            else:
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert '{key}': '{value}' to float, using 0.0")
                    cleaned[key] = 0.0
        return cleaned
    
    def _extract_features_from_json(self, json_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from the JSON format provided
        
        Expected JSON structure:
        {
            'timestamp': '...',
            'features': {
                'percent_change_1h': -0.009995,
                'close_mean_3h': 1.0004,
                ...
            }
        }
        """
        # Extract the features dict
        if 'features' in json_data:
            features = json_data['features']
        else:
            # Assume the whole dict is features
            features = json_data
        
        # Map hour-based to minute-based names
        features = self._map_hour_to_minute_features(features)
        
        # Clean the features
        features = self._clean_features(features)
        
        return features
    
    def _create_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base model predictions"""
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        # Meta-features (same as training)
        meta_features = np.column_stack([
            rf_proba,                           # RF confidence
            xgb_proba,                          # XGB confidence
            rf_proba * xgb_proba,               # Agreement signal
            np.abs(rf_proba - xgb_proba),       # Disagreement
            np.minimum(rf_proba, xgb_proba),    # Conservative view
            (rf_proba + xgb_proba) / 2,         # Average confidence
            (rf_proba > 0.5).astype(int) & (xgb_proba > 0.5).astype(int),  # Both predict positive
        ])
        
        return meta_features
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict depeg event probability for next 24h
        
        Parameters:
        -----------
        features : dict
            Feature dictionary (must contain all required features)
            Can use either hour-based (3h, 6h) or minute-based (180m, 360m) names
        
        Returns:
        --------
        dict : Prediction result
        """
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not isinstance(features, dict):
                raise ValueError(f"Features must be dict, got {type(features)}")
            
            # Map hour-based to minute-based names if needed
            features = self._map_hour_to_minute_features(features)
            
            # Clean features (handle empty strings, None, etc.)
            features = self._clean_features(features)
            
            # Check for missing required features
            missing = [f for f in self.feature_cols if f not in features]
            if missing:
                logger.warning(f"Missing {len(missing)} features, filling with 0: {missing[:5]}...")
                for feat in missing:
                    features[feat] = 0.0
            
            # Select and order features
            X = pd.DataFrame([features])[self.feature_cols]
            
            # Handle NaN (should be rare after cleaning, but just in case)
            if X.isna().any().any():
                logger.warning("NaN values detected after cleaning, filling with 0")
                X = X.fillna(0)
            
            # Handle inf values
            X = X.replace([np.inf, -np.inf], 0)
            
            # Create meta-features
            meta_features = self._create_meta_features(X)
            
            # Get probability from meta-model
            probability = float(self.meta_model.predict_proba(meta_features)[:, 1][0])
            
            # Binary prediction using optimal threshold
            prediction = int(probability >= self.optimal_threshold)
            
            # Confidence level based on probability
            if probability >= 0.90:
                confidence = "VERY_HIGH"
            elif probability >= 0.70:
                confidence = "HIGH"
            elif probability >= 0.50:
                confidence = "MEDIUM"
            elif probability >= 0.30:
                confidence = "LOW"
            else:
                confidence = "VERY_LOW"
            
            # Risk classification
            if prediction == 1:  # Depeg predicted
                if probability >= 0.95:
                    risk_class = "CRITICAL"
                    alert = True
                elif probability >= 0.85:
                    risk_class = "HIGH"
                    alert = True
                else:
                    risk_class = "MODERATE"
                    alert = True
            else:  # No depeg predicted
                if probability >= 0.50:
                    risk_class = "ELEVATED"
                    alert = False
                else:
                    risk_class = "NORMAL"
                    alert = False
            
            # Calculate additional metrics from input features
            max_peg_deviation = features.get('peg_deviation_max_1440m', 
                                            features.get('peg_deviation_max_24h', 0.0))
            max_peg_deviation_pct = max_peg_deviation * 100
            
            # Risk score (0-10 scale)
            risk_score = min(10, int(probability * 10))
            
            # Inference time
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build result matching your expected format
            result = {
                'success': True,
                'prediction': {
                    'depeg_predicted': bool(prediction),
                    'depeg_probability': round(probability, 6),
                    'confidence_level': confidence,
                    'threshold_used': round(self.optimal_threshold, 6),
                    'risk_classification': risk_class,
                    'alert_required': alert,
                    'max_peg_deviation': round(max_peg_deviation, 5),
                    'max_peg_deviation_pct': round(max_peg_deviation_pct, 3),
                    'risk_level': risk_class,
                    'risk_score': risk_score,
                    'timestamp': datetime.now().isoformat(),
                    'inference_time_ms': round(inference_time, 2)
                }
            }
            
            logger.debug(
                f"Prediction: {'DEPEG' if prediction else 'SAFE'} "
                f"(p={probability:.4f}, {confidence}) in {inference_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_from_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction method for JSON input format
        
        Parameters:
        -----------
        json_data : dict
            JSON data with structure:
            {
                'timestamp': '...',
                'features': {
                    'percent_change_1h': ...,
                    'close_mean_3h': ...,
                    ...
                }
            }
        
        Returns:
        --------
        dict : Prediction result with structure:
            {
                'success': True,
                'prediction': {
                    'depeg_predicted': bool,
                    'depeg_probability': float,
                    'confidence_level': str,
                    'risk_classification': str,
                    'alert_required': bool,
                    'max_peg_deviation': float,
                    'max_peg_deviation_pct': float,
                    'risk_level': str,
                    'risk_score': int,
                    'timestamp': str,
                    'inference_time_ms': float
                }
            }
        """
        try:
            # Extract features from JSON
            features = self._extract_features_from_json(json_data)
            
            # Make prediction
            result = self.predict(features)
            
            # Add input timestamp if available
            if 'timestamp' in json_data:
                result['input_timestamp'] = json_data['timestamp']
            
            return result
            
        except Exception as e:
            logger.error(f"JSON prediction failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, json_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict for multiple JSON samples
        
        Parameters:
        -----------
        json_data_list : list of dict
            List of JSON data objects
        
        Returns:
        --------
        list of dict : Predictions for each sample
        """
        
        results = []
        for json_data in json_data_list:
            result = self.predict_from_json(json_data)
            results.append(result)
        
        return results
    
    def get_required_features(self) -> List[str]:
        """Get list of required feature names (minute-based)"""
        return self.feature_cols.copy()
    
    def get_hour_based_feature_names(self) -> List[str]:
        """Get list of hour-based feature names that can be used as input"""
        hour_names = list(self.feature_name_map.keys())
        return hour_names
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'Hybrid Stacked Ensemble',
            'base_models': ['RandomForest', 'XGBoost'],
            'meta_model': 'LogisticRegression',
            'rf_trees': self.rf_model.n_estimators,
            'xgb_trees': self.xgb_model.n_estimators,
            'required_features': len(self.feature_cols),
            'optimal_threshold': self.optimal_threshold,
            'target_precision': self.target_precision,
            'min_recall': self.min_recall,
            'loaded_at': self.loaded_at.isoformat(),
            'accepts_hour_based_names': True,
            'feature_mapping_available': len(self.feature_name_map) > 0
        }


# ============================================================================
# SINGLETON PATTERN (Optional - for even faster repeated use)
# ============================================================================

_global_predictor = None

def get_predictor() -> DepegPredictor:
    """
    Get global predictor instance (loads once, reuses many times)
    
    Usage:
    ------
    predictor = get_predictor()
    result = predictor.predict_from_json(json_data)
    """
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = DepegPredictor()
    return _global_predictor


# ============================================================================
# CONVENIENCE FUNCTION (Simplest usage)
# ============================================================================

def predict_depeg_from_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for one-line predictions from JSON
    
    Usage:
    ------
    from depeg_predictor_updated import predict_depeg_from_json
    
    result = predict_depeg_from_json(your_json_data)
    if result['success']:
        print(f"Depeg predicted: {result['prediction']['depeg_predicted']}")
    
    Parameters:
    -----------
    json_data : dict
        Your JSON data with features
    
    Returns:
    --------
    dict : Prediction result
    """
    predictor = get_predictor()
    return predictor.predict_from_json(json_data)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("DEPEG CLASSIFIER - UPDATED JSON DEMO")
    print("="*80)
    
    # Sample JSON input (matching your format)
    sample_json = {
        'timestamp': '1771108380000',
        'features': {
            'percent_change_1h': -0.009995002498749524,
            'percent_change_24h': -0.01998800719568039,
            'volume_percent_change_24h': -100,
            'market_cap_percent_change_1d': -0.041189086459913635,
            'circulating_supply_percent_change_1d': -0.04559389532495715,
            'daily_range': 0,
            'body_size': 0,
            'upper_shadow': 0,
            'lower_shadow': 0,
            'close_position': 0,
            'trade_size': 0,
            'taker_buy_ratio': 0,
            'taker_sell_ratio': 0,
            'volume_imbalance': 0,
            'mcap_to_supply_ratio': 0.9996008970574389,
            'price_to_mcap_billion': 0.005444543426752472,
            # Hour-based names (will be auto-mapped to minute-based)
            'close_mean_3h': 1.000422222222222,
            'close_std_3h': 0.00006477770803849398,
            'close_min_3h': 1.0003,
            'close_max_3h': 1.0005,
            'volume_mean_3h': 360813.0833333333,
            'volume_std_3h': 2903878.5531664365,
            'volume_sum_3h': 64946355,
            'trades_mean_3h': 217.7888888888889,
            'trade_size_mean_3h': 867.9097108055663,
            'market_cap_mean_3h': 183750694308.58612,
            'market_cap_std_3h': 13432229.339502184,
            'circulating_supply_mean_3h': 183814259616.87305,
            'close_mean_6h': 1.0003830555555555,
            'close_std_6h': 0.00007249318263766084,
            'close_min_6h': 1.0002,
            'close_max_6h': 1.0005,
            'volume_mean_6h': 376268.6194444444,
            'volume_std_6h': 2368411.3492028033,
            'volume_sum_6h': 135456703,
            'trades_mean_6h': 240.79444444444445,
            'trade_size_mean_6h': 961.1688333097898,
            'market_cap_mean_6h': 183756744161.82718,
            'market_cap_std_6h': 14361991.110911762,
            'circulating_supply_mean_6h': 183815936504.91888,
            'close_mean_12h': 1.0003665277777776,
            'close_std_12h': 0.00009183130997419099,
            'close_min_12h': 1.0002,
            'close_max_12h': 1.0005,
            'volume_mean_12h': 576203.4388888889,
            'volume_std_12h': 2072227.3355952878,
            'volume_sum_12h': 414866476,
            'trades_mean_12h': 267.1125,
            'trade_size_mean_12h': 1525.81912061672,
            'market_cap_mean_12h': 183760638862.88715,
            'market_cap_std_12h': 15493245.173919138,
            'circulating_supply_mean_12h': 183818167196.6133,
            'close_mean_24h': 1.0004170138888888,
            'close_std_24h': 0.00009280270314523648,
            'close_min_24h': 1.0002,
            'close_max_24h': 1.0006,
            'volume_mean_24h': 462975.97222222225,
            'volume_std_24h': 1832062.1301180248,
            'volume_sum_24h': 666685400,
            'trades_mean_24h': 230.6888888888889,
            'trade_size_mean_24h': 1445.5374816188985,
            'market_cap_mean_24h': 183758865437.07022,
            'market_cap_std_24h': 20792576.109523162,
            'circulating_supply_mean_24h': 183822586427.1406,
            'peg_deviation': 0.00039999999999995595,
            'peg_deviation_pct': 0.039999999999995595,
            'peg_direction': 0.00039999999999995595,
            'above_peg': 1,
            'below_peg': 0,
            'peg_deviation_mean_3h': 0.0004222222222221757,
            'peg_deviation_max_3h': 0.0004999999999999449,
            'peg_deviation_std_3h': 0.00006477770803849398,
            'peg_deviation_mean_6h': 0.00038305555555551336,
            'peg_deviation_max_6h': 0.0004999999999999449,
            'peg_deviation_std_6h': 0.00007249318263766085,
            'peg_deviation_mean_12h': 0.0003665277777777374,
            'peg_deviation_max_12h': 0.0004999999999999449,
            'peg_deviation_std_12h': 0.000091831309974191,
            'peg_deviation_mean_24h': 0.00041701388888884294,
            'peg_deviation_max_24h': 0.0005999999999999339,
            'peg_deviation_std_24h': 0.00009280270314523649,
            'peg_stress_1pct_3h': 0,
            'peg_stress_1pct_24h': 0,
            'peg_stress_2pct_3h': 0,
            'peg_stress_2pct_24h': 0,
            'volume_vs_3h': 0,
            'volume_vs_24h': 0,
            'trade_size_vs_24h': 0,
            'mcap_to_volume_24h': 275.6076527834568,
            'price_accel_1h_24h': 0.009993004696930867,
            'volume_spike_peg_stress': 0,
            'extreme_volume_spike': 0,
            'large_trade_anomaly': 0,
            'peg_deviation_mean_180m': 0,
            'peg_deviation_std_180m': 0,
            'peg_deviation_mean_720m': 0,
            'peg_deviation_std_720m': 0
        }
    }
    
    # Method 1: Direct instantiation
    print("\n1. Initializing predictor...")
    predictor = DepegPredictor()
    print(f"   ✓ Loaded successfully")
    
    # Method 2: Predict from JSON
    print("\n2. Making prediction from JSON...")
    result = predictor.predict_from_json(sample_json)
    
    if result['success']:
        pred = result['prediction']
        print(f"\n   {'='*60}")
        print(f"   PREDICTION RESULTS")
        print(f"   {'='*60}")
        print(f"   Depeg Predicted:    {pred['depeg_predicted']}")
        print(f"   Probability:        {pred['depeg_probability']:.6f}")
        print(f"   Confidence Level:   {pred['confidence_level']}")
        print(f"   Risk Classification: {pred['risk_classification']}")
        print(f"   Risk Score:         {pred['risk_score']}/10")
        print(f"   Alert Required:     {pred['alert_required']}")
        print(f"   Max Peg Deviation:  {pred['max_peg_deviation']:.5f} ({pred['max_peg_deviation_pct']:.3f}%)")
        print(f"   Threshold Used:     {pred['threshold_used']:.6f}")
        print(f"   Inference Time:     {pred['inference_time_ms']:.2f}ms")
        print(f"   {'='*60}\n")
    else:
        print(f"   ❌ Prediction failed: {result['error']}")
    
    # Method 3: Convenience function
    print("\n3. Using convenience function:")
    result2 = predict_depeg_from_json(sample_json)
    if result2['success']:
        print(f"   ✓ Depeg: {result2['prediction']['depeg_predicted']} "
              f"(p={result2['prediction']['depeg_probability']:.4f})")
    
    print("\n" + "="*80)
    print("INTEGRATION GUIDE")
    print("="*80)
    print("""
# In your backend server:

from depeg_predictor_updated import DepegPredictor

# Initialize once at server startup
depeg_predictor = DepegPredictor()

# In your API endpoint or data processing function:
def process_live_data(json_data):
    '''
    json_data format:
    {
        'timestamp': '1771108380000',
        'features': {
            'percent_change_1h': -0.009995,
            'close_mean_3h': 1.0004,  # Can use hour-based names!
            ...
        }
    }
    '''
    
    # Predict (fast! ~5-15ms)
    result = depeg_predictor.predict_from_json(json_data)
    
    if result['success']:
        prediction = result['prediction']
        
        # Check if alert needed
        if prediction['alert_required']:
            send_alert({
                'type': 'DEPEG_WARNING',
                'risk': prediction['risk_classification'],
                'probability': prediction['depeg_probability'],
                'timestamp': prediction['timestamp']
            })
        
        # Store to database
        db.save_prediction({
            'input_timestamp': json_data['timestamp'],
            'depeg_predicted': prediction['depeg_predicted'],
            'probability': prediction['depeg_probability'],
            'risk_level': prediction['risk_level'],
            'risk_score': prediction['risk_score'],
            'prediction_timestamp': prediction['timestamp']
        })
        
        return result
    else:
        logger.error(f"Prediction error: {result['error']}")
        return result

# Flask/FastAPI example:
@app.post("/predict")
async def predict_endpoint(data: dict):
    result = depeg_predictor.predict_from_json(data)
    return result
    """)