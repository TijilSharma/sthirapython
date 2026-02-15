import pandas as pd
from controller.featurecalc import StablecoinRealtimeFeatureEngine, EngineConfig
from controller.merge_raw_data import merge_raw_data  # your merged CSV function

# ----------------------------
# Initialize engine
# ----------------------------
cfg = EngineConfig()  # default configuration
engine = StablecoinRealtimeFeatureEngine(cfg)

# ----------------------------
# Load historical 30-day data at startup
# ----------------------------
def load_history():
    try:
        df = merge_raw_data()
        print(df.head)  # should return a DataFrame with all required columns
        engine.load_history(df)
        print(f"[SlidingWindow] Loaded {len(df)} historical rows")
    except Exception as e:
        print(f"[SlidingWindow] Failed to load history: {e}")

# ----------------------------
# Update sliding window & compute features
# ----------------------------
def process_live_data(new_row: dict) -> dict:
    """
    Append new row to sliding window and compute features for latest tick.
    Returns JSON-ready dict of features.
    """
    return engine.update_and_compute(new_row)
