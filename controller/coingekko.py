import requests
import pandas as pd
import time
from datetime import datetime, timezone

# Current time in seconds (Epoch)
to_timestamp = int(time.time())

# Subtract 30 days (30 days * 24 hours * 60 mins * 60 secs)
from_timestamp = to_timestamp - (30 * 24 * 60 * 60)

url = f"https://api.coingecko.com/api/v3/coins/tether/market_chart/range?vs_currency=usd&from={from_timestamp}&to={to_timestamp}"

print(url)

headers = {"x-cg-demo-api-key": "CG-fXhdQPJdcFMW13DrWaQ6D5NQ"}
OUTPUT_FILE = "usdt_supply_2025_2026.csv"

response = requests.get(url, headers=headers)

all_prices = []
all_market_caps = []

response.raise_for_status()
data = response.json()

all_prices.extend(data["prices"])
all_market_caps.extend(data["market_caps"])

df_price = pd.DataFrame(all_prices, columns=["timestamp", "price"])
df_mc = pd.DataFrame(all_market_caps, columns=["timestamp", "market_cap"])

df = df_price.merge(df_mc, on="timestamp")

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.drop_duplicates("timestamp")
df = df.sort_values("timestamp")
df = df.set_index("timestamp")

# ==============================
# CALCULATE CIRCULATING SUPPLY
# ==============================

df["circulating_supply"] = df["market_cap"] / df["price"]

print("Rows fetched:", len(df))
print("Start:", df.index.min())
print("End:", df.index.max())

# ==============================
# OPTIONAL: RESAMPLE TO MINUTE
# ==============================

df_minute = df.resample("1min").ffill()

print("Minute rows:", len(df_minute))

# ==============================
# SAVE
# ==============================

df_minute.reset_index().to_csv(OUTPUT_FILE, index=False)

print("Saved to:", OUTPUT_FILE)

print(response.text)