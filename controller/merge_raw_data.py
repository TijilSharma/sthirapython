import pandas as pd
from controller.fetch_coingekko import fetch_coingecko_supply
from controller.fetchohclv import initialize_ohlcv

RAW_OUTPUT = "raw_data.csv"

def merge_raw_data() -> pd.DataFrame:
    """
    Merge OHLCV and CoinGecko supply data.
    Returns a DataFrame with all numeric columns ready for engine.
    Saves CSV once.
    """
    print("[merge_raw_data] Starting merge...")

    # Load OHLCV
    initialize_ohlcv()  # creates ohclvNew.csv
    df_ohlcv = pd.read_csv("ohclvNew.csv")
    df_ohlcv["open_time"] = pd.to_datetime(df_ohlcv["open_time"])
    df_ohlcv["open_time"] = (df_ohlcv["open_time"].astype('int64') // 10**6)
    df_ohlcv["close_time"] = pd.to_datetime(df_ohlcv["close_time"], errors='coerce')
    df_ohlcv["close_time"] = (df_ohlcv["close_time"].astype('int64') // 10**6)
    df_ohlcv = df_ohlcv.sort_values("open_time")

    # Load CoinGecko supply
    df_cg = fetch_coingecko_supply()
    df_cg["timestamp"] = pd.to_datetime(df_cg["timestamp"])
    df_cg["epoch_ms"] = (df_cg["timestamp"].astype('int64') // 10**6)
    df_cg = df_cg.sort_values("epoch_ms")

    # Merge
    df_merged = pd.merge_asof(
        df_ohlcv,
        df_cg,
        left_on="open_time",
        right_on="epoch_ms",
        direction="backward"
    )

    # Drop helper columns
    df_merged = df_merged.drop(columns=["epoch_ms", "timestamp", "close_time", "ignore"], errors='ignore')

    # Ensure engine numeric columns exist
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "trades", "taker_base", "taker_quote",
                    "quote_volume", "market_cap", "circulating_supply"]
    for c in numeric_cols:
        df_merged[c] = pd.to_numeric(df_merged[c], errors="coerce")

    # Fill missing numeric values with 0 (or drop if you prefer)
    df_merged = df_merged.fillna(0).reset_index(drop=True)

    # Save CSV
    df_merged.to_csv(RAW_OUTPUT, index=False)
    print(f"[merge_raw_data] Merged CSV saved to {RAW_OUTPUT}, rows: {len(df_merged)}")

    return df_merged
