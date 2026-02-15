import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

SYMBOL = "USDCUSDT"
INTERVAL = "1m"
OUTPUT_FILE = "ohclvNew.csv"
BASE_URL = "https://api.binance.com"
SESSION = requests.Session()
WORKERS = 4  # Adjust depending on your system

# -----------------------------
# SAFE REQUEST WITH RETRIES
# -----------------------------
def safe_request(url, params, retries=5):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Retry {attempt+1}/{retries} due to error: {e}")
            time.sleep(2 ** attempt)
    return None

# -----------------------------
# FETCH DATA FOR A GIVEN RANGE
# -----------------------------
def fetch_range(start_ts, end_ts):
    url = f"{BASE_URL}/api/v3/klines"
    all_data = []
    current = start_ts

    while current < end_ts:
        print("Fetching:", datetime.fromtimestamp(current / 1000))
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": current,
            "endTime": end_ts,
            "limit": 1000
        }

        data = safe_request(url, params)
        if not data:
            break

        all_data.extend(data)
        current = data[-1][6] + 1  # next close_time + 1ms
        time.sleep(0.1)

    return all_data

# -----------------------------
# SPLIT LAST 30 DAYS INTO WEEKS
# -----------------------------
def split_into_weeks():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=30)
    ranges = []

    while start < now:
        end = start + timedelta(days=7)
        if end > now:
            end = now
        ranges.append((int(start.timestamp() * 1000), int(end.timestamp() * 1000)))
        start = end

    return ranges

# -----------------------------
# SAVE DATA TO CSV
# -----------------------------
def save_to_csv(df, file_name=OUTPUT_FILE):
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}. Rows: {len(df)}")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def run_fetch():
    try:
        print("Splitting 30-day range into weekly jobs...")
        week_ranges = split_into_weeks()
        all_results = []

        # Parallel fetch
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(fetch_range, start, end) for start, end in week_ranges]

            for future in as_completed(futures):
                data_chunk = future.result()
                if data_chunk:
                    all_results.extend(data_chunk)
                    print("Week chunk fetched. Candles:", len(data_chunk))

        if not all_results:
            raise Exception("No data fetched.")

        # Convert to DataFrame
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore"
        ]
        df = pd.DataFrame(all_results, columns=columns)

        df = df.drop_duplicates(subset=["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)

        # Convert open_time to datetime and numeric columns to float
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "taker_base", "taker_quote"]
        df[numeric_cols] = df[numeric_cols].astype(float)

        save_to_csv(df)
        print("DONE fetching 1 month of OHLCV data in parallel.")

    except Exception as e:
        print("ERROR OCCURRED:", e)
        if 'df' in locals():
            print("Saving partial data...")
            save_to_csv(df, "partial_ohclv.csv")
        elif 'all_results' in locals() and all_results:
            print("Saving raw partial data...")
            partial_df = pd.DataFrame(all_results)
            partial_df.to_csv("partial_raw_data.csv", index=False)
        print("Process stopped safely.")
