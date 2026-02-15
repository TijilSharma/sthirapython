import logging
from typing import Dict, Any
from controller.mahimodel import LiquidityPredictor

logger = logging.getLogger(__name__)

# Global singleton instance
_predictor_instance: LiquidityPredictor | None = None


def load_model_once() -> LiquidityPredictor:
    """
    Loads the LiquidityPredictor only once (singleton).
    Safe to call multiple times.
    """
    global _predictor_instance

    if _predictor_instance is None:
        logger.info("Initializing LiquidityPredictor (first load)...")
        _predictor_instance = LiquidityPredictor()
        logger.info("LiquidityPredictor loaded successfully.")

    return _predictor_instance


def predict_from_websocket(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper used by websocket controller.

    1. Ensures model is loaded once
    2. Extracts features from payload
    3. Runs prediction
    4. Returns result
    """

    predictor = load_model_once()

    try:
        # If your websocket already sends clean features dict:
        features = payload

        # If transformation is required, do it here:
        # features = transform_payload_to_features(payload)

        result = predictor.predict(features)

        return {
            "success": True,
            "prediction": result
        }

    except Exception as e:
        logger.exception("Prediction failed from websocket payload")
        return {
            "success": False,
            "error": str(e)
        }
