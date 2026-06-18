from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "date",
    "channel",
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "revenue",
    "spend",
    "clicks",
    "impressions",
    "conversions",
    "budget",
]


@dataclass
class LoadedData:
    raw: pd.DataFrame
    daily_features: pd.DataFrame
    feature_metadata: Dict[str, float]


def load_all_data(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    frames: List[pd.DataFrame] = []
    for path in sorted(data_dir.rglob("*.csv")):
        frame = pd.read_csv(path)
        normalized = normalize_source_frame(frame, path.name)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined = validate_and_clean(combined)
    return combined


def normalize_source_frame(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    columns = {column: _normalize_name(column) for column in frame.columns}
    lowered = frame.rename(columns=columns)
    if {"campaignid", "timeperiod"}.issubset(lowered.columns):
        return _normalize_bing(lowered)
    if {"campaign_id", "segments_date"}.issubset(lowered.columns):
        return _normalize_google(lowered)
    if {"campaign_id", "date_start"}.issubset(lowered.columns):
        return _normalize_meta(lowered)
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def validate_and_clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    numeric_columns = ["revenue", "spend", "clicks", "impressions", "conversions", "budget"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "channel", "campaign_id"])
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    frame[numeric_columns] = frame[numeric_columns].fillna(0.0)
    frame["revenue"] = frame["revenue"].clip(lower=0.0)
    frame["spend"] = frame["spend"].clip(lower=0.0)
    frame["budget"] = frame["budget"].clip(lower=0.0)
    frame["campaign_name"] = frame["campaign_name"].fillna("unknown_campaign")
    frame["campaign_type"] = frame["campaign_type"].fillna("UNKNOWN")
    return frame.loc[:, CANONICAL_COLUMNS].drop_duplicates().reset_index(drop=True)


def build_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "revenue", "spend", "clicks", "impressions", "conversions", "roas", "ctr", "cvr", "cpc", "cpm"])
    daily = frame.groupby("date", as_index=False).agg(
        {
            "revenue": "sum",
            "spend": "sum",
            "clicks": "sum",
            "impressions": "sum",
            "conversions": "sum",
            "budget": "sum",
        }
    ).sort_values("date")
    daily["roas"] = _ratio(daily["revenue"], daily["spend"])
    daily["ctr"] = _ratio(daily["clicks"], daily["impressions"])
    daily["cvr"] = _ratio(daily["conversions"], daily["clicks"])
    daily["cpc"] = _ratio(daily["spend"], daily["clicks"])
    daily["cpm"] = _ratio(daily["spend"] * 1000.0, daily["impressions"])

    for lag in (1, 3, 7, 14, 30, 60):
        daily[f"revenue_lag_{lag}"] = daily["revenue"].shift(lag)
        daily[f"spend_lag_{lag}"] = daily["spend"].shift(lag)

    for window in (3, 7, 14, 30, 60):
        daily[f"revenue_roll_mean_{window}"] = daily["revenue"].rolling(window, min_periods=1).mean()
    for window in (7, 14, 30):
        daily[f"revenue_roll_std_{window}"] = daily["revenue"].rolling(window, min_periods=1).std().fillna(0.0)
    for window in (7, 14, 30):
        daily[f"spend_roll_mean_{window}"] = daily["spend"].rolling(window, min_periods=1).mean()

    daily["revenue_growth"] = daily["revenue"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily["spend_growth"] = daily["spend"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily["conversion_growth"] = daily["conversions"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily["click_growth"] = daily["clicks"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    daily["day_of_week"] = daily["date"].dt.weekday
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["month"] = daily["date"].dt.month
    daily["quarter"] = daily["date"].dt.quarter
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    daily["campaign_age"] = np.arange(len(daily)) + 1
    daily["channel_share"] = _ratio(daily["revenue"], daily["revenue"].sum())
    daily["budget_utilization"] = _ratio(daily["spend"], daily["budget"])
    daily["channel_revenue_share"] = daily["channel_share"]
    daily["spend_x_clicks"] = daily["spend"] * daily["clicks"]
    daily["spend_x_conversions"] = daily["spend"] * daily["conversions"]
    daily["ctr_x_cvr"] = daily["ctr"] * daily["cvr"]
    daily["roas_x_budget"] = daily["roas"] * daily["budget"]
    return daily.fillna(0.0)


def summarize_feature_metadata(frame: pd.DataFrame) -> Dict[str, float]:
    revenue = frame.get("revenue", pd.Series(dtype=float)).astype(float)
    revenue = revenue.replace([np.inf, -np.inf], np.nan).dropna()
    skew = float(revenue.skew()) if len(revenue) else 0.0
    target_transform = "log1p" if abs(skew) > 1.0 else "identity"
    return {
        "target_transform": target_transform,
        "target_mean": float(revenue.mean()) if len(revenue) else 0.0,
        "target_std": float(revenue.std(ddof=0)) if len(revenue) else 0.0,
        "row_count": float(len(frame)),
        "revenue_skew": skew,
    }


def _normalize_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum() or character == "_")


def _normalize_bing(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame.get("timeperiod"), errors="coerce")
    result["channel"] = "bing"
    result["campaign_id"] = frame.get("campaignid")
    result["campaign_name"] = frame.get("campaignname", "unknown_campaign")
    result["campaign_type"] = frame.get("campaigntype", "bing")
    result["revenue"] = pd.to_numeric(frame.get("revenue"), errors="coerce")
    result["spend"] = pd.to_numeric(frame.get("spend"), errors="coerce")
    result["clicks"] = pd.to_numeric(frame.get("clicks"), errors="coerce")
    result["impressions"] = pd.to_numeric(frame.get("impressions"), errors="coerce")
    result["conversions"] = pd.to_numeric(frame.get("conversions"), errors="coerce")
    result["budget"] = pd.to_numeric(frame.get("dailybudget"), errors="coerce")
    return result


def _normalize_google(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame.get("segments_date"), errors="coerce")
    result["channel"] = "google"
    result["campaign_id"] = frame.get("campaign_id")
    result["campaign_name"] = frame.get("campaign_name", "unknown_campaign")
    result["campaign_type"] = frame.get("campaign_advertising_channel_type", "google")
    result["spend"] = pd.to_numeric(frame.get("metrics_cost_micros"), errors="coerce") / 1_000_000.0
    result["clicks"] = pd.to_numeric(frame.get("metrics_clicks"), errors="coerce")
    result["impressions"] = pd.to_numeric(frame.get("metrics_impressions"), errors="coerce")
    result["conversions"] = pd.to_numeric(frame.get("metrics_conversions"), errors="coerce")
    result["revenue"] = pd.to_numeric(frame.get("metrics_conversions_value"), errors="coerce")
    result["budget"] = pd.to_numeric(frame.get("campaign_budget_amount"), errors="coerce")
    return result


def _normalize_meta(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame.get("date_start"), errors="coerce")
    result["channel"] = "meta"
    result["campaign_id"] = frame.get("campaign_id")
    result["campaign_name"] = frame.get("campaign_name", "unknown_campaign")
    result["campaign_type"] = "PAID_SOCIAL"
    result["spend"] = pd.to_numeric(frame.get("spend"), errors="coerce")
    result["clicks"] = pd.to_numeric(frame.get("clicks"), errors="coerce")
    result["impressions"] = pd.to_numeric(frame.get("impressions"), errors="coerce")
    result["conversions"] = pd.to_numeric(frame.get("conversion"), errors="coerce")
    result["budget"] = pd.to_numeric(frame.get("daily_budget"), errors="coerce")
    result["revenue"] = np.nan
    return result


def estimate_missing_revenue(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    known = result[(result["revenue"] > 0) & (result["spend"] > 0)].copy()
    if known.empty:
        result["revenue"] = result["revenue"].fillna(0.0).clip(lower=0.0)
        return result

    known_roas = (known["revenue"] / known["spend"]).replace([np.inf, -np.inf], np.nan).dropna()
    if known_roas.empty:
        global_roas = 5.0
    else:
        global_roas = float(known_roas.median())
    global_roas = float(np.clip(global_roas, 1.0, 15.0))

    channel_roas = (
        result.loc[(result["revenue"] > 0) & (result["spend"] > 0)]
        .assign(roas=lambda frame_: frame_["revenue"] / frame_["spend"])
        .groupby("channel", dropna=False)["roas"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_dict()
    )

    missing_mask = ~np.isfinite(result["revenue"]) | (result["revenue"] <= 0)
    if missing_mask.any():
        fallback = result.loc[missing_mask, "channel"].map(channel_roas).astype(float)
        fallback = fallback.fillna(global_roas)
        fallback = fallback.clip(lower=1.0, upper=15.0)
        imputed = result.loc[missing_mask, "spend"].fillna(0.0) * fallback
        result.loc[missing_mask, "revenue"] = imputed

    result["revenue"] = result["revenue"].fillna(0.0).clip(lower=0.0)
    return result


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    if np.isscalar(denominator):
        denominator_value = float(denominator)
        if not np.isfinite(denominator_value) or denominator_value == 0:
            return pd.Series(np.zeros(len(numerator), dtype=float), index=numerator.index)
        values = numerator / denominator_value
    else:
        denominator = denominator.replace(0, np.nan)
        values = numerator / denominator
    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
