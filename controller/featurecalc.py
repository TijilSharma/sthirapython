from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineConfig:
    """
    Configuration for minute-resolution rolling features.

    windows_hours:
      These are expressed in HOURS, but internally converted to rows using rows_per_hour.
      Example: 3h => 180 rows if rows_per_hour=60.
    """
    rows_per_hour: int = 60
    max_days: int = 30
    windows_hours: List[int] = (3, 6, 12, 24, 48, 168, 720)  # 3h ... 30d
    peg_price: float = 1.0
    eps: float = 1e-12


class StablecoinRealtimeFeatureEngine:
    """
    Production-grade realtime feature engineering engine.

    - Keeps last `max_days` of minute bars in memory
    - Accepts one new minute bar as JSON
    - Computes feature vector for the latest row only
    - Returns JSON with engineered features

    Required base fields per minute:
      open_time, open, high, low, close, volume, trades,
      taker_base, taker_quote, quote_volume, market_cap, circulating_supply
    """

    def __init__(self, config: EngineConfig = EngineConfig()):
        self.cfg = config
        self.max_rows = int(self.cfg.max_days * 24 * self.cfg.rows_per_hour)  # 30 days of minutes
        self.df: pd.DataFrame = pd.DataFrame()

    # ----------------------------
    # Initialization / Buffer Mgmt
    # ----------------------------

    def load_history(self, history_df: pd.DataFrame) -> None:
        """
        Load last 30 days (or more) of minute data ONCE at service startup.
        Keeps only the most recent max_rows.
        """
        required = self._required_base_columns()
        missing = [c for c in required if c not in history_df.columns]
        if missing:
            raise ValueError(f"History is missing required columns: {missing}")

        df = history_df.copy()
        df = self._standardize_types(df)
        df = df.sort_values("open_time").reset_index(drop=True)

        if len(df) > self.max_rows:
            df = df.iloc[-self.max_rows:].reset_index(drop=True)

        self.df = df

    def update_and_compute(self, minute_bar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append one minute JSON bar, keep rolling 30d memory, compute features for last row,
        return JSON payload containing engineered features.
        """

        total_rows = len(self.df)

        if total_rows < 43200:
            print(f"Warmup phase: {total_rows}/43200 rows")
            return {}
    
        if self.df.empty:
            raise RuntimeError("Engine has no history loaded. Call load_history() first.")

        # Append new bar
        new_row = pd.DataFrame([minute_bar])
        missing = [c for c in self._required_base_columns() if c not in new_row.columns]
        if missing:
            raise ValueError(f"Incoming bar missing required columns: {missing}")

        new_row = self._standardize_types(new_row)

        # Ensure monotonic time (optional but recommended)
        if float(new_row.iloc[0]["open_time"]) <= float(self.df.iloc[-1]["open_time"]):
            raise ValueError("Incoming open_time must be strictly greater than last stored open_time.")

        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # Trim to 30 days
        cutoff_time = self.df.iloc[-1]["open_time"] - (30 * 24 * 60 * 60*1000)
        self.df = self.df[self.df["open_time"] >= cutoff_time]

        print(self.df)
        # Compute features for latest row only
        features = self._compute_latest_features()

        # Return JSON
        ts = self.df.iloc[-1]["open_time"]
        return {
            "timestamp": self._format_timestamp(ts),
            "features": self._serialize_features(features),
        }

    # ----------------------------
    # Feature Computation
    # ----------------------------

    def _compute_latest_features(self) -> Dict[str, float]:
        """
        Compute full engineered feature set for the latest row, using rolling history.
        """
        df = self.df
        i = len(df) - 1
        row = df.iloc[i]

        eps = self.cfg.eps
        peg = self.cfg.peg_price
        rph = self.cfg.rows_per_hour

        feats: Dict[str, float] = {}

        # Ensure 'price' exists conceptually (use close)
        price_t = float(row["close"])

        # ----------------------------
        # A) Derived "input-like" fields computed from history
        # (So you don't need them from API)
        # ----------------------------

        # Percent changes for price (based on close)
        feats["percent_change_1h"] = self._pct_change(df["close"], i, 1 * rph)
        feats["percent_change_24h"] = self._pct_change(df["close"], i, 24 * rph)
        feats["percent_change_7d"] = self._pct_change(df["close"], i, 168 * rph)
        feats["percent_change_30d"] = self._pct_change(df["close"], i, 720 * rph)

        # Volume percent changes
        feats["volume_percent_change_24h"] = self._pct_change(df["volume"], i, 24 * rph)
        feats["volume_percent_change_7d"] = self._pct_change(df["volume"], i, 168 * rph)
        feats["volume_percent_change_30d"] = self._pct_change(df["volume"], i, 720 * rph)

        # Market cap percent changes
        feats["market_cap_percent_change_1d"] = self._pct_change(df["market_cap"], i, 24 * rph)
        feats["market_cap_percent_change_7d"] = self._pct_change(df["market_cap"], i, 168 * rph)
        feats["market_cap_percent_change_30d"] = self._pct_change(df["market_cap"], i, 720 * rph)

        # Supply percent changes
        feats["circulating_supply_percent_change_1d"] = self._pct_change(df["circulating_supply"], i, 24 * rph)
        feats["circulating_supply_percent_change_7d"] = self._pct_change(df["circulating_supply"], i, 168 * rph)
        feats["circulating_supply_percent_change_30d"] = self._pct_change(df["circulating_supply"], i, 720 * rph)

        # ----------------------------
        # B) Step 1: Basic features (as in your training script, corrected)
        # ----------------------------

        open_t = float(row["open"])
        high_t = float(row["high"])
        low_t = float(row["low"])
        close_t = float(row["close"])
        vol_t = float(row["volume"])
        qvol_t = float(row["quote_volume"])
        trades_t = float(row["trades"])
        taker_base_t = float(row["taker_base"])
        market_cap_t = float(row["market_cap"])
        supply_t = float(row["circulating_supply"])

        # Price action
        feats["daily_range"] = (high_t - low_t) / (open_t + eps)
        feats["body_size"] = abs(close_t - open_t) / (open_t + eps)

        # Shadow fractions (stable even when high==low)
        denom_hl = (high_t - low_t) + eps
        feats["upper_shadow"] = (high_t - max(open_t, close_t)) / denom_hl
        feats["lower_shadow"] = (min(open_t, close_t) - low_t) / denom_hl
        feats["close_position"] = (close_t - low_t) / denom_hl

        # Volume / order flow
        feats["trade_size"] = vol_t / (trades_t + 1.0)
        feats["taker_buy_ratio"] = taker_base_t / (vol_t + eps)

        # Correct taker sell ratio (not the incorrect taker_quote/quote_volume)
        feats["taker_sell_ratio"] = (vol_t - taker_base_t) / (vol_t + eps)

        # Symmetric imbalance in [-1, +1]
        feats["volume_imbalance"] = (2.0 * taker_base_t - vol_t) / (vol_t + eps)

        # Market
        feats["mcap_to_supply_ratio"] = market_cap_t / (supply_t + eps)

        # This existed in your script: price / (mcap in billions)
        feats["price_to_mcap_billion"] = price_t / ((market_cap_t / 1e9) + eps)

        # ----------------------------
        # C) Step 2: Rolling features for specified windows (hours -> rows)
        # ----------------------------

        for h in self.cfg.windows_hours:
            w = h * rph
            if len(df) < w:
                # Not enough history; keep NaN-like defaults (or 0). Here we set NaN.
                feats[f"close_mean_{h}h"] = np.nan
                feats[f"close_std_{h}h"] = np.nan
                feats[f"close_min_{h}h"] = np.nan
                feats[f"close_max_{h}h"] = np.nan

                feats[f"volume_mean_{h}h"] = np.nan
                feats[f"volume_std_{h}h"] = np.nan
                feats[f"volume_sum_{h}h"] = np.nan

                feats[f"trades_mean_{h}h"] = np.nan
                feats[f"trade_size_mean_{h}h"] = np.nan

                feats[f"market_cap_mean_{h}h"] = np.nan
                feats[f"market_cap_std_{h}h"] = np.nan
                feats[f"circulating_supply_mean_{h}h"] = np.nan
                continue

            window_df = df.iloc[-w:]

            feats[f"close_mean_{h}h"] = float(window_df["close"].mean())
            feats[f"close_std_{h}h"] = float(window_df["close"].std(ddof=1))
            feats[f"close_min_{h}h"] = float(window_df["close"].min())
            feats[f"close_max_{h}h"] = float(window_df["close"].max())

            feats[f"volume_mean_{h}h"] = float(window_df["volume"].mean())
            feats[f"volume_std_{h}h"] = float(window_df["volume"].std(ddof=1))
            feats[f"volume_sum_{h}h"] = float(window_df["volume"].sum())

            feats[f"trades_mean_{h}h"] = float(window_df["trades"].mean())
            # trade_size is derived; compute mean of derived
            trade_size_series = window_df["volume"] / (window_df["trades"] + 1.0)
            feats[f"trade_size_mean_{h}h"] = float(trade_size_series.mean())

            feats[f"market_cap_mean_{h}h"] = float(window_df["market_cap"].mean())
            feats[f"market_cap_std_{h}h"] = float(window_df["market_cap"].std(ddof=1))
            feats[f"circulating_supply_mean_{h}h"] = float(window_df["circulating_supply"].mean())

        # ----------------------------
        # D) Step 3: Peg deviation features
        # ----------------------------

        peg_dev_t = abs(close_t - peg)
        feats["peg_deviation"] = peg_dev_t
        feats["peg_deviation_pct"] = peg_dev_t * 100.0
        feats["peg_direction"] = close_t - peg
        feats["above_peg"] = 1.0 if close_t > peg else 0.0
        feats["below_peg"] = 1.0 if close_t < peg else 0.0

        for h in self.cfg.windows_hours:
            w = h * rph
            if len(df) < w:
                feats[f"peg_deviation_mean_{h}h"] = np.nan
                feats[f"peg_deviation_max_{h}h"] = np.nan
                feats[f"peg_deviation_std_{h}h"] = np.nan
                continue

            peg_window = (df.iloc[-w:]["close"] - peg).abs()
            feats[f"peg_deviation_mean_{h}h"] = float(peg_window.mean())
            feats[f"peg_deviation_max_{h}h"] = float(peg_window.max())
            feats[f"peg_deviation_std_{h}h"] = float(peg_window.std(ddof=1))

        # Stress indicators: your script used rolling(3) and rolling(24) but those were "hours".
        # For minute data, use 3h=180 rows and 24h=1440 rows.
        rows_3h = 3 * rph
        rows_24h = 24 * rph

        peg_series = (df["close"] - peg).abs()
        feats["peg_stress_1pct_3h"] = float((peg_series > 0.01).iloc[-rows_3h:].sum()) if len(df) >= rows_3h else np.nan
        feats["peg_stress_1pct_24h"] = float((peg_series > 0.01).iloc[-rows_24h:].sum()) if len(df) >= rows_24h else np.nan
        feats["peg_stress_2pct_3h"] = float((peg_series > 0.02).iloc[-rows_3h:].sum()) if len(df) >= rows_3h else np.nan
        feats["peg_stress_2pct_24h"] = float((peg_series > 0.02).iloc[-rows_24h:].sum()) if len(df) >= rows_24h else np.nan

        # Recovery metrics based on 24h close min/max
        if len(df) >= rows_24h:
            close_24h = df.iloc[-rows_24h:]["close"]
            feats["price_range_24h"] = float(close_24h.max() - close_24h.min())
            feats["worst_deviation_24h"] = float((close_24h - peg).abs().max())
        else:
            feats["price_range_24h"] = np.nan
            feats["worst_deviation_24h"] = np.nan

        # ----------------------------
        # E) Step 4: Interaction features
        # ----------------------------

        # volume_vs_* uses volume_mean windows
        feats["volume_vs_3h"] = vol_t / (feats.get("volume_mean_3h", np.nan) + eps) if not math.isnan(feats.get("volume_mean_3h", np.nan)) else np.nan
        feats["volume_vs_24h"] = vol_t / (feats.get("volume_mean_24h", np.nan) + eps) if not math.isnan(feats.get("volume_mean_24h", np.nan)) else np.nan
        feats["volume_vs_7d"] = vol_t / (feats.get("volume_mean_168h", np.nan) + eps) if not math.isnan(feats.get("volume_mean_168h", np.nan)) else np.nan
        feats["volume_vs_30d"] = vol_t / (feats.get("volume_mean_720h", np.nan) + eps) if not math.isnan(feats.get("volume_mean_720h", np.nan)) else np.nan

        # trade_size_vs_* uses trade_size_mean windows
        feats["trade_size_vs_24h"] = feats["trade_size"] / (feats.get("trade_size_mean_24h", np.nan) + eps) if not math.isnan(feats.get("trade_size_mean_24h", np.nan)) else np.nan
        feats["trade_size_vs_7d"] = feats["trade_size"] / (feats.get("trade_size_mean_168h", np.nan) + eps) if not math.isnan(feats.get("trade_size_mean_168h", np.nan)) else np.nan

        # Liquidity: market_cap / volume_sum
        feats["mcap_to_volume_24h"] = market_cap_t / (feats.get("volume_sum_24h", np.nan) + eps) if not math.isnan(feats.get("volume_sum_24h", np.nan)) else np.nan
        feats["mcap_to_volume_7d"] = market_cap_t / (feats.get("volume_sum_168h", np.nan) + eps) if not math.isnan(feats.get("volume_sum_168h", np.nan)) else np.nan

        # Supply: supply relative to rolling mean
        feats["supply_to_mean_24h"] = supply_t / (feats.get("circulating_supply_mean_24h", np.nan) + eps) if not math.isnan(feats.get("circulating_supply_mean_24h", np.nan)) else np.nan
        feats["supply_to_mean_7d"] = supply_t / (feats.get("circulating_supply_mean_168h", np.nan) + eps) if not math.isnan(feats.get("circulating_supply_mean_168h", np.nan)) else np.nan

        # Divergence features (require percent changes computed above)
        feats["mcap_supply_div_1d"] = feats["market_cap_percent_change_1d"] - feats["circulating_supply_percent_change_1d"]
        feats["mcap_supply_div_7d"] = feats["market_cap_percent_change_7d"] - feats["circulating_supply_percent_change_7d"]

        # ----------------------------
        # F) Step 5: Momentum features
        # ----------------------------

        feats["price_accel_1h_24h"] = feats["percent_change_1h"] - feats["percent_change_24h"]
        feats["price_accel_24h_7d"] = feats["percent_change_24h"] - feats["percent_change_7d"]
        feats["price_accel_7d_30d"] = feats["percent_change_7d"] - feats["percent_change_30d"]

        feats["volume_accel_24h_7d"] = feats["volume_percent_change_24h"] - feats["volume_percent_change_7d"]
        feats["volume_accel_7d_30d"] = feats["volume_percent_change_7d"] - feats["volume_percent_change_30d"]

        feats["mcap_accel_1d_7d"] = feats["market_cap_percent_change_1d"] - feats["market_cap_percent_change_7d"]
        feats["mcap_accel_7d_30d"] = feats["market_cap_percent_change_7d"] - feats["market_cap_percent_change_30d"]

        feats["supply_accel_1d_7d"] = feats["circulating_supply_percent_change_1d"] - feats["circulating_supply_percent_change_7d"]
        feats["redemption_pressure_1d"] = -feats["circulating_supply_percent_change_1d"]
        feats["redemption_pressure_7d"] = -feats["circulating_supply_percent_change_7d"]

        # Trend strength vs moving averages (use close_mean windows)
        feats["price_vs_ma24h"] = (close_t - feats.get("close_mean_24h", np.nan)) / (feats.get("close_mean_24h", np.nan) + eps) if not math.isnan(feats.get("close_mean_24h", np.nan)) else np.nan
        feats["price_vs_ma7d"] = (close_t - feats.get("close_mean_168h", np.nan)) / (feats.get("close_mean_168h", np.nan) + eps) if not math.isnan(feats.get("close_mean_168h", np.nan)) else np.nan
        feats["price_vs_ma30d"] = (close_t - feats.get("close_mean_720h", np.nan)) / (feats.get("close_mean_720h", np.nan) + eps) if not math.isnan(feats.get("close_mean_720h", np.nan)) else np.nan

        # ----------------------------
        # G) Step 6: Market dynamics features
        # ----------------------------

        # volume_spike_peg_stress: volume_vs_24h > 2 and peg_deviation > 1%
        volume_vs_24h = feats.get("volume_vs_24h", np.nan)
        feats["volume_spike_peg_stress"] = 1.0 if (not math.isnan(volume_vs_24h) and volume_vs_24h > 2.0 and peg_dev_t > 0.01) else 0.0

        # extreme_volume_spike: volume_vs_24h > 3
        feats["extreme_volume_spike"] = 1.0 if (not math.isnan(volume_vs_24h) and volume_vs_24h > 3.0) else 0.0

        # large_trade_anomaly: trade_size_vs_24h > 2
        trade_size_vs_24h = feats.get("trade_size_vs_24h", np.nan)
        feats["large_trade_anomaly"] = 1.0 if (not math.isnan(trade_size_vs_24h) and trade_size_vs_24h > 2.0) else 0.0

        return feats

    # ----------------------------
    # Helpers
    # ----------------------------

    @staticmethod
    def _required_base_columns() -> List[str]:
        return [
            "open_time",
            "open", "high", "low", "close",
            "volume", "quote_volume", "trades",
            "taker_base", "taker_quote",
            "market_cap", "circulating_supply",
        ]

    def _standardize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce numeric types; keeps open_time as float/int compatible.
        """
        df = df.copy()

        # open_time can be int ms or seconds; keep numeric
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")

        numeric_cols = [c for c in df.columns if c != "open_time"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # quote_volume might arrive named differently; ensure it exists if provided
        # (Your incoming data uses quote_volume; keep as-is.)

        # Drop completely invalid rows (optional)
        if df[numeric_cols].isna().any(axis=1).any():
            # In production you might log and reject; here we keep it strict:
            df = df.dropna(subset=numeric_cols)

        return df

    def _pct_change(self, series: pd.Series, idx: int, lag: int) -> float:
        """
        Percent change in (0..100) using minute index lag.
        If insufficient history, return NaN.
        """
        if idx - lag < 0:
            return np.nan
        prev = float(series.iloc[idx - lag])
        cur = float(series.iloc[idx])
        if abs(prev) < self.cfg.eps:
            return np.nan
        return 100.0 * (cur - prev) / prev

    @staticmethod
    def _format_timestamp(open_time_value: Any) -> str:
        """
        Leaves timestamp formatting policy to you:
        - If you store open_time in epoch ms, returning as string preserves exact value.
        - Frontend/backend can convert to ISO if needed.
        """
        return str(int(open_time_value)) if pd.notna(open_time_value) else ""

    @staticmethod
    def _serialize_features(features: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert numpy NaNs to None for JSON safety.
        """
        out: Dict[str, Any] = {}
        for k, v in features.items():
            if v is None:
                out[k] = None
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                out[k] = None
            else:
                out[k] = float(v)
        return out