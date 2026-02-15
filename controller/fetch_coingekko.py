# controllers/fetch_coingecko.py

import requests
import pandas as pd
import time
from datetime import datetime, timezone

OUTPUT_FILE = "usdt_supply.csv"

def fetch_coingecko_supply():
    """
    Fetch USDT supply and market cap from CoinGecko, resampled to 1-min.
    Saves CSV as usdt_supply.csv
    """
    print("Fetching CoinGecko supply data...")
    to_timestamp = int(time.time())
    from_timestamp = to_timestamp - (30 * 24 * 60 * 60)  # last 30 days

    url = f"https://api.coingecko.com/api/v3/coins/tether/market_chart/range?vs_currency=usd&from={from_timestamp}&to={to_timestamp}"
    headers = {"x-cg-demo-api-key": "CG-fXhdQPJdcFMW13DrWaQ6D5NQ"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    df_price = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df_mc = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
    df = df_price.merge(df_mc, on="timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["circulating_supply"] = df["market_cap"] / df["price"]

    # Resample to 1-minute frequency
    df = df.set_index("timestamp").resample("1min").ffill().reset_index()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"CoinGecko supply data saved to {OUTPUT_FILE}, rows: {len(df)}")
    return df
