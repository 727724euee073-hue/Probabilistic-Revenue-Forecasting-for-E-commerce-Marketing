from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast
import warnings

import numpy as np
import pandas as pd


def _safe_div(numerator: pd.Series | np.ndarray | float, denominator: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        numerator_series = pd.Series(numerator)
        denominator_series = pd.Series(denominator, index=numerator_series.index)
        denominator_series = denominator_series.replace(0, np.nan)
        result = numerator_series / denominator_series
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
    if isinstance(result, np.ndarray):
        result[~np.isfinite(result)] = 0.0
    elif not np.isfinite(result):
        result = 0.0
    return result


def _clip_positive(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = floor
    return np.maximum(values, floor)


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(_clip_positive(values))


@dataclass
class ForecastResult:
    forecast_period: str
    predicted_revenue: float
    p10: float
    p50: float
    p90: float
    predicted_roas: float
    confidence_score: float


@dataclass
class RevenueForecastModel:
    seed: int = 42
    bootstrap_simulations: int = 1000
    min_roas: float = 3.0
    max_roas: float = 8.0
    warning_roas_threshold: float = 15.0
    daily_revenue_spike_cap: float = 3.0
    target_transform: str = "log1p"
    inverse_transform: str = "expm1"
    base_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 0.30,
            "seasonal": 0.25,
            "spend_response": 0.25,
            "reversion": 0.20,
        }
    )

    def load_features(self, features_path: str | Path) -> pd.DataFrame:
        frame = pd.read_csv(features_path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        history = self._prepare_history(features)
        daily_revenue_path, daily_spend_path, residuals, diagnostics = self._forecast_daily_paths(history)

        historical_daily_avg = diagnostics["historical_daily_revenue"]
        forecast_daily_avg = float(np.mean(daily_revenue_path)) if len(daily_revenue_path) else 0.0
        if historical_daily_avg > 0 and forecast_daily_avg > 3.0 * historical_daily_avg:
            scale = (3.0 * historical_daily_avg) / forecast_daily_avg
            daily_revenue_path = daily_revenue_path * scale
            warnings.warn(
                f"Forecast daily revenue exceeded 3x historical average; scaling revenue path by {scale:.4f}",
                RuntimeWarning,
            )

        rows: List[ForecastResult] = []
        for horizon in (30, 60, 90):
            slice_revenue = daily_revenue_path[:horizon]
            slice_spend = daily_spend_path[:horizon]
            samples = self._bootstrap_horizon_samples(slice_revenue, slice_spend, residuals, horizon)

            predicted_revenue = float(np.sum(slice_revenue))
            predicted_spend = float(np.sum(slice_spend))
            p10, p50, p90 = self._percentile_triplet(samples["revenue"])
            p10, p50, p90 = self._apply_revenue_bounds(p10, p50, p90, predicted_revenue, historical_daily_avg, horizon)
            predicted_revenue = float(np.clip(predicted_revenue, p10, p90))
            predicted_spend = self._apply_spend_bounds(predicted_spend, diagnostics["historical_daily_spend"], horizon)
            predicted_roas = self._safe_roas(predicted_revenue, predicted_spend)
            if predicted_roas > self.warning_roas_threshold:
                warnings.warn(
                    f"Predicted ROAS {predicted_roas:.2f} exceeds warning threshold {self.warning_roas_threshold:.2f}.",
                    RuntimeWarning,
                )
                predicted_roas = self.warning_roas_threshold
                predicted_revenue = predicted_roas * predicted_spend

            confidence = self._confidence_score(samples["revenue"], samples["spend"], diagnostics)

            rows.append(
                ForecastResult(
                    forecast_period=f"{horizon}d",
                    predicted_revenue=predicted_revenue,
                    p10=p10,
                    p50=p50,
                    p90=p90,
                    predicted_roas=predicted_roas,
                    confidence_score=confidence,
                )
            )

        self._validate_forecast_rows(rows, diagnostics)
        return pd.DataFrame([row.__dict__ for row in rows])

    def _prepare_history(self, features: pd.DataFrame) -> pd.DataFrame:
        required = ["date", "revenue", "spend", "clicks", "impressions", "conversions"]
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        history = features.loc[:, required].copy()
        history = history.groupby("date", as_index=False).sum(numeric_only=True).sort_values("date")
        for column in ["revenue", "spend", "clicks", "impressions", "conversions"]:
            history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)

        history["roas"] = pd.Series(_safe_div(history["revenue"], history["spend"].replace(0, np.nan)), index=history.index).fillna(0.0)
        history["ctr"] = pd.Series(_safe_div(history["clicks"], history["impressions"].replace(0, np.nan)), index=history.index).fillna(0.0)
        history["cvr"] = pd.Series(_safe_div(history["conversions"], history["clicks"].replace(0, np.nan)), index=history.index).fillna(0.0)
        return history.reset_index(drop=True)

    def _forecast_daily_paths(self, history: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        revenue = _clip_positive(history["revenue"].to_numpy(dtype=float))
        spend = _clip_positive(history["spend"].to_numpy(dtype=float))
        dates = pd.to_datetime(history["date"]).dt.normalize()

        if len(history) == 0:
            zeros = np.zeros(90, dtype=float)
            return zeros, zeros, np.zeros(1, dtype=float), {
                "historical_daily_revenue": 1.0,
                "historical_daily_spend": 1.0,
                "historical_roas": 5.0,
            }

        historical_daily_revenue = float(np.mean(revenue))
        historical_daily_spend = float(np.mean(spend))
        historical_roas = float(np.sum(revenue) / max(np.sum(spend), 1e-6))
        historical_roas = float(np.clip(historical_roas, self.min_roas, self.max_roas))

        recent_revenue = float(np.median(revenue[-min(28, len(revenue)) :]))
        recent_spend = float(np.median(spend[-min(28, len(spend)) :]))
        recent_revenue = max(recent_revenue if np.isfinite(recent_revenue) else historical_daily_revenue, 1.0)
        recent_spend = max(recent_spend if np.isfinite(recent_spend) else historical_daily_spend, 1.0)

        revenue_growth = self._bounded_growth_rate(revenue)
        spend_growth = self._bounded_growth_rate(spend)
        weekday_rev = self._weekday_profile(dates, revenue)
        weekday_spend = self._weekday_profile(dates, spend)

        blended_roas = self._bounded_roas(float(np.median(history["roas"])), historical_roas)
        residuals = self._fit_residuals(_safe_log1p(revenue))

        revenue_path = np.zeros(90, dtype=float)
        spend_path = np.zeros(90, dtype=float)
        current_revenue = recent_revenue
        current_spend = recent_spend

        for step in range(90):
            forecast_date = dates.iloc[-1] + pd.Timedelta(days=step + 1)
            weekday = int(forecast_date.weekday())
            seasonal_revenue = float(np.clip(weekday_rev.get(weekday, 1.0), 0.82, 1.18))
            seasonal_spend = float(np.clip(weekday_spend.get(weekday, 1.0), 0.85, 1.15))

            trend_revenue = current_revenue * (1.0 + revenue_growth / 30.0)
            trend_spend = current_spend * (1.0 + spend_growth / 30.0)
            trend_revenue = float(np.clip(trend_revenue, historical_daily_revenue * 0.70, historical_daily_revenue * 1.25))
            trend_spend = float(np.clip(trend_spend, historical_daily_spend * 0.80, historical_daily_spend * 1.20))

            spend_forecast = self._blend_values(
                [trend_spend, historical_daily_spend, recent_spend],
                [0.55, 0.25, 0.20],
            ) * seasonal_spend
            spend_forecast = float(np.clip(spend_forecast, historical_daily_spend * 0.50, historical_daily_spend * 1.50))

            revenue_from_spend = spend_forecast * blended_roas
            revenue_forecast = self._blend_values(
                [trend_revenue, revenue_from_spend, historical_daily_revenue],
                [0.40, 0.45, 0.15],
            ) * seasonal_revenue

            revenue_forecast = float(np.clip(revenue_forecast, historical_daily_revenue * 0.50, historical_daily_revenue * 1.80))
            if step > 0:
                revenue_forecast = min(revenue_forecast, revenue_path[step - 1] * self.daily_revenue_spike_cap)

            revenue_path[step] = max(revenue_forecast, 1e-6)
            spend_path[step] = max(spend_forecast, 1e-6)

            current_revenue = 0.85 * current_revenue + 0.15 * revenue_path[step]
            current_spend = 0.85 * current_spend + 0.15 * spend_path[step]

        diagnostics = {
            "historical_daily_revenue": historical_daily_revenue,
            "historical_daily_spend": historical_daily_spend,
            "historical_roas": historical_roas,
            "baseline_roas": blended_roas,
        }
        return revenue_path, spend_path, residuals, diagnostics

    def _fit_residuals(self, log_revenue: np.ndarray) -> np.ndarray:
        values = np.asarray(log_revenue, dtype=float)
        if values.size < 3:
            return np.zeros(1, dtype=float)
        smoothed = pd.Series(values).ewm(alpha=0.35, adjust=False).mean().to_numpy()
        residuals = np.clip(values - smoothed, -0.35, 0.35)
        residuals = residuals[np.isfinite(residuals)]
        return residuals if residuals.size else np.zeros(1, dtype=float)

    def _bootstrap_horizon_samples(self, revenue_path: np.ndarray, spend_path: np.ndarray, residuals: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed + horizon)
        residual_pool = residuals if residuals.size else np.zeros(1, dtype=float)
        revenue_samples = np.zeros(self.bootstrap_simulations, dtype=float)
        spend_samples = np.zeros(self.bootstrap_simulations, dtype=float)

        log_revenue_path = _safe_log1p(revenue_path)
        log_spend_path = _safe_log1p(spend_path)

        for index in range(self.bootstrap_simulations):
            sampled_residuals = rng.choice(residual_pool, size=horizon, replace=True)
            revenue_noise = np.clip(sampled_residuals * rng.uniform(0.80, 1.05, size=horizon), -0.25, 0.25)
            spend_noise = np.clip(rng.normal(0.0, 0.03, size=horizon), -0.10, 0.10)

            simulated_revenue = np.expm1(log_revenue_path + revenue_noise)
            simulated_spend = np.expm1(log_spend_path + spend_noise)
            simulated_revenue = np.clip(simulated_revenue, 1e-6, None)
            simulated_spend = np.clip(simulated_spend, 1e-6, None)

            revenue_samples[index] = float(np.sum(simulated_revenue))
            spend_samples[index] = float(np.sum(simulated_spend))

        return {
            "revenue": revenue_samples,
            "spend": spend_samples,
            "roas": np.clip(revenue_samples / np.maximum(spend_samples, 1e-6), self.min_roas * 0.5, self.warning_roas_threshold),
        }

    def _percentile_triplet(self, values: np.ndarray) -> Tuple[float, float, float]:
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        return float(p10), float(p50), float(p90)

    def _confidence_score(self, revenue_samples: np.ndarray, spend_samples: np.ndarray, diagnostics: Dict[str, float]) -> float:
        revenue_cv = np.std(revenue_samples) / max(np.mean(revenue_samples), 1e-6)
        spend_cv = np.std(spend_samples) / max(np.mean(spend_samples), 1e-6)
        baseline_roas = diagnostics["baseline_roas"]
        roas_gap = abs(baseline_roas - 5.0) / 10.0
        score = 1.0 - min(0.55 * revenue_cv + 0.30 * spend_cv + 0.15 * roas_gap, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _safe_roas(self, revenue: float, spend: float) -> float:
        spend = max(float(spend), 1e-6)
        roas = float(revenue) / spend
        if not np.isfinite(roas):
            return self.min_roas
        return float(np.clip(roas, self.min_roas, self.warning_roas_threshold))

    def _bounded_roas(self, candidate: float, historical: float) -> float:
        value = 0.6 * float(candidate) + 0.4 * float(historical)
        return float(np.clip(value, self.min_roas, self.max_roas))

    def _apply_revenue_bounds(self, p10: float, p50: float, p90: float, point: float, historical_daily: float, horizon: int) -> Tuple[float, float, float]:
        lower = historical_daily * horizon * 0.70
        upper = historical_daily * horizon * 1.20
        point = float(np.clip(point, lower, upper))
        p10 = float(np.clip(p10, lower * 0.85, point))
        p50 = float(np.clip(p50, p10, point))
        p90 = float(np.clip(p90, p50, upper))
        return self._enforce_monotonic_triplet(p10, p50, p90)

    def _apply_spend_bounds(self, point: float, historical_daily: float, horizon: int) -> float:
        lower = historical_daily * horizon * 0.80
        upper = historical_daily * horizon * 1.20
        return float(np.clip(point, lower, upper))

    def _validate_forecast_rows(self, rows: List[ForecastResult], diagnostics: Dict[str, float]) -> None:
        for row in rows:
            if not (row.predicted_revenue > 0 and row.predicted_roas > 0):
                raise ValueError("Forecast validation failed: revenue and ROAS must be positive.")
            if not (row.p10 < row.p50 < row.p90):
                raise ValueError("Forecast validation failed: P10 < P50 < P90 must hold.")
            if not (0.0 <= row.confidence_score <= 1.0):
                raise ValueError("Forecast validation failed: confidence score must be between 0 and 1.")
            daily_average = row.predicted_revenue / max(int(row.forecast_period.rstrip("d")), 1)
            if daily_average > 3.0 * diagnostics["historical_daily_revenue"]:
                raise ValueError("Forecast validation failed: daily forecast exceeds 3x historical average.")

    def _enforce_monotonic_triplet(self, p10: float, p50: float, p90: float) -> Tuple[float, float, float]:
        p10 = max(float(p10), 1e-6)
        p50 = max(float(p50), p10 + 1e-6)
        p90 = max(float(p90), p50 + 1e-6)
        return p10, p50, p90

    def _blend_values(self, values: Iterable[float], weights: Iterable[float]) -> float:
        values_array = np.asarray(list(values), dtype=float)
        weights_array = np.asarray(list(weights), dtype=float)
        weights_array = weights_array / max(weights_array.sum(), 1e-6)
        return float(np.dot(values_array, weights_array))

    def _bounded_growth_rate(self, values: np.ndarray) -> float:
        values = _clip_positive(values)
        if len(values) < 2:
            return 0.0
        recent = values[-min(14, len(values)) :]
        prior = values[: len(values) - len(recent)] if len(values) > len(recent) else values[: max(len(values) // 2, 1)]
        recent_mean = float(np.mean(recent))
        prior_mean = float(np.mean(prior)) if len(prior) else recent_mean
        if prior_mean <= 0:
            return 0.0
        growth = recent_mean / prior_mean - 1.0
        return float(np.clip(growth, -0.10, 0.12))

    def _weekday_profile(self, dates: pd.Series, values: np.ndarray) -> Dict[int, float]:
        frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
        frame["weekday"] = frame["date"].dt.weekday
        profile = frame.groupby("weekday")["value"].mean()
        overall = float(frame["value"].mean()) if len(frame) else 1.0
        overall = overall if np.isfinite(overall) and overall > 0 else 1.0
        result = {int(cast(Any, weekday)): float(val / overall) for weekday, val in profile.items() if np.isfinite(val)}
        for weekday in range(7):
            result.setdefault(weekday, 1.0)
        return result
