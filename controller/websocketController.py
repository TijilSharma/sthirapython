from flask_socketio import emit
from controller.sliding_window import process_live_data
import pandas as pd
from controller.ml_service import predict_from_websocket
from controller.inference2 import DepegPredictor

REQUIRED_COLUMNS = [
    "open_time",
    "open", "high", "low", "close",
    "volume", "quote_volume",
    "trades", "taker_base", "taker_quote",
    "ignore", "price",
    "market_cap", "circulating_supply"
]

NUMERIC_COLUMNS = [
    "open", "high", "low", "close",
    "volume", "quote_volume",
    "taker_base", "taker_quote",
    "market_cap", "circulating_supply"
]

def normalize_tick(tick: dict) -> dict:
    """
    Normalize incoming JSON tick to match 30-day historical engine schema.
    """
    df = pd.DataFrame([tick])

    # 1️⃣ Convert open_time ms -> seconds
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")


    # 2️⃣ Convert numeric fields from strings to floats
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")

    # 3️⃣ Handle trades = None
    df["trades"] = pd.to_numeric(df.get("trades", 0), errors="coerce").fillna(0.0)

    # 4️⃣ Add missing required columns
    df["price"] = df["close"]          # derived
    df["ignore"] = 0.0                 # placeholder

    # 5️⃣ Keep only required columns, drop extras
    df = df[REQUIRED_COLUMNS]

    return df.iloc[0].to_dict()


def register_socket_events(socketio):
    @socketio.on("connect")
    def handle_connect():
        print("Express connected to ML server ✅")

    @socketio.on("ml_data")
    def handle_ml_data(data):
        """
        Expected data format: JSON tick from Binance / other API.
        Fully normalize before feeding engine.
        """
        try:
            # --- Normalize incoming tick ---
            tick = normalize_tick(data)
            print(tick)

            # --- Process tick ---
            features_payload = process_live_data(tick)
            result_1 = predict_from_websocket(features_payload["features"])
            predictor = DepegPredictor()
            result_2 = predictor.predict_from_json(features_payload["features"])

            # --- Send back features ---
            emit("ml_response", {
                "features": features_payload,
                "prediction_1": result_1,
                "prediction_2": result_2
            })

        except Exception as e:
            print("[Error] Failed to process live data:", e)
            emit("ml_response", {"error": str(e)})
