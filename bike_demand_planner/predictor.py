"""
predictor.py
Core logic for the Bike Demand Planner: loading models, validating inputs,
applying business rules, predicting, and turning predictions into decisions.

Three models are served:
  MLR         - Multiple Linear Regression trained on daily data
  ANN         - Artificial Neural Network trained on daily data
  Hourly ANN  - ANN trained on hourly data; its 24 hourly predictions are summed
                to give the daily total, and also give the hourly demand profile

The Gradio interface (app.py) only calls functions from this file,
so every rule here can be tested on its own (see test_predictor.py).
"""
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ---------------------------------------------------------------------------
# Constants taken from the UCI Bike Sharing dataset documentation
# ---------------------------------------------------------------------------
TEMP_MAX, HUM_MAX, WIND_MAX = 41.0, 100.0, 67.0      # normalisation divisors

WEATHER = {
    "Clear / few clouds": 1,
    "Mist / cloudy": 2,
    "Light rain / light snow": 3,
    "Heavy rain / snow / fog": 4,
}
WEATHER_BY_CODE = {v: k for k, v in WEATHER.items()}
SEASON_NAMES = {1: "Winter (Dec 21 - Mar 20)", 2: "Spring (Mar 21 - Jun 20)",
                3: "Summer (Jun 21 - Sep 22)", 4: "Autumn (Sep 23 - Dec 20)"}
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Business assumptions for turning a prediction into operational decisions.
# These are planning assumptions, not facts from the dataset. Change them to match the operator.
RENTALS_PER_STAFF = 1500   # one field staff member per 1,500 daily rentals
MIN_STAFF = 2              # never schedule fewer than 2 field staff


class ValidationError(ValueError):
    """Raised when an input breaks a hard rule. The message is shown to the user."""


@dataclass
class DayInput:
    day: date
    holiday: bool
    weather: int          # 1-4
    temp_c: float         # degrees Celsius
    humidity: float       # percent
    windspeed: float      # km/h (dataset scale, max 67)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class ModelBundle:
    """Holds the trained MLR, ANN, scaler and metadata produced by the notebook."""

    def __init__(self, model_dir=MODEL_DIR):
        prep_path = os.path.join(model_dir, "preprocess_mlr.joblib")
        ann_path = os.path.join(model_dir, "ann_model.keras")
        prep_h_path = os.path.join(model_dir, "preprocess_hourly.joblib")
        ann_h_path = os.path.join(model_dir, "ann_hourly.keras")
        meta_path = os.path.join(model_dir, "metrics.json")
        for p in (prep_path, ann_path, prep_h_path, ann_h_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing model file: {p}. Run every cell in bike_mlr_vs_ann.ipynb first.")
        try:
            import joblib
            from tensorflow import keras
            prep = joblib.load(prep_path)
            self.ann = keras.models.load_model(ann_path)
            prep_h = joblib.load(prep_h_path)
            self.ann_h = keras.models.load_model(ann_h_path)
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)
        except Exception as e:  # corrupted or incompatible files
            raise RuntimeError(f"Could not load the trained models: {e}") from e

        self.scaler = prep["scaler"]
        self.mlr = prep["mlr"]
        self.columns = prep["columns"]
        self.y_mean, self.y_std = prep["y_mean"], prep["y_std"]
        self.scaler_h = prep_h["scaler"]
        self.columns_h = prep_h["columns"]
        self.yh_mu, self.yh_sd = prep_h["y_mu"], prep_h["y_sd"]
        self.ranges = self.meta["training_ranges"]
        self.best_model = self.meta["best_model"]

    def predict_raw(self, rows: pd.DataFrame):
        """rows: DataFrame of raw features (yr, holiday, workingday, temp, hum, windspeed,
        season, mnth, weekday, weathersit). Returns (mlr_preds, ann_preds)."""
        enc = pd.get_dummies(rows, columns=["season", "mnth", "weekday", "weathersit"], dtype=int)
        enc = enc.reindex(columns=self.columns, fill_value=0)
        Xs = self.scaler.transform(enc)
        p_mlr = self.mlr.predict(Xs)
        p_ann = self.ann.predict(Xs, verbose=0).ravel() * self.y_std + self.y_mean
        return p_mlr, p_ann

    def predict_hourly(self, day_row: dict):
        """Expand one day's features to 24 hourly rows and return 24 hourly predictions."""
        rows = pd.DataFrame([{**day_row, "hr": hr} for hr in range(24)])
        enc = pd.get_dummies(rows, columns=["season", "mnth", "weekday", "weathersit", "hr"], dtype=int)
        enc = enc.reindex(columns=self.columns_h, fill_value=0)
        p = self.ann_h.predict(self.scaler_h.transform(enc), verbose=0).ravel() * self.yh_sd + self.yh_mu
        return np.clip(p, 0, None)


# ---------------------------------------------------------------------------
# Feature derivation (business rules that the user should not have to enter)
# ---------------------------------------------------------------------------
def season_of(d: date) -> int:
    """Season code exactly as the dataset defines it (astronomical seasons)."""
    md = (d.month, d.day)
    if md >= (12, 21) or md <= (3, 20):
        return 1
    if md <= (6, 20):
        return 2
    if md <= (9, 22):
        return 3
    return 4


def weekday_of(d: date) -> int:
    """Dataset convention: 0 = Sunday ... 6 = Saturday."""
    return (d.weekday() + 1) % 7


def parse_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None or str(value).strip() == "":
        raise ValidationError("Date is required.")
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    try:  # gradio DateTime may send a timestamp
        return datetime.fromtimestamp(float(s)).date()
    except (ValueError, OverflowError, OSError):
        raise ValidationError(f"'{value}' is not a valid date. Use YYYY-MM-DD.")


def _to_number(value, name):
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValidationError(f"{name} is required.")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number (got '{value}').")
    if not np.isfinite(x):
        raise ValidationError(f"{name} must be a finite number.")
    return x


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(day, holiday, weather, temp_c, humidity, windspeed, ranges) -> DayInput:
    """Hard rules raise ValidationError. Soft rules add a warning but still allow the prediction."""
    d = parse_date(day)
    temp_c = _to_number(temp_c, "Temperature")
    humidity = _to_number(humidity, "Humidity")
    windspeed = _to_number(windspeed, "Wind speed")

    # Weather: accept label or code
    if isinstance(weather, str) and weather in WEATHER:
        w = WEATHER[weather]
    else:
        try:
            w = int(float(weather))
        except (TypeError, ValueError):
            raise ValidationError(f"Unknown weather '{weather}'. Use one of: {', '.join(WEATHER)}.")
    if w not in WEATHER_BY_CODE:
        raise ValidationError("Weather code must be 1, 2, 3 or 4.")

    if isinstance(holiday, str):
        h = holiday.strip().lower()
        if h not in ("0", "1", "true", "false", "yes", "no", ""):
            raise ValidationError(f"Holiday must be yes/no (got '{holiday}').")
        holiday = h in ("1", "true", "yes")
    holiday = bool(holiday)

    # --- Hard rules: physically impossible or outside the model's input scale ---
    if not 0 <= temp_c <= TEMP_MAX:
        raise ValidationError(f"Temperature must be between 0 and {TEMP_MAX:.0f} °C (dataset scale).")
    if not 0 <= humidity <= HUM_MAX:
        raise ValidationError("Humidity must be between 0 and 100 %.")
    if not 0 <= windspeed <= WIND_MAX:
        raise ValidationError(f"Wind speed must be between 0 and {WIND_MAX:.0f} (dataset scale).")
    if holiday and weekday_of(d) in (0, 6):
        raise ValidationError(
            f"{d:%Y-%m-%d} is a {DAY_NAMES[weekday_of(d)]}. In this dataset, public holidays "
            "only fall on weekdays (weekend holidays are already non-working days). Untick 'holiday'.")

    inp = DayInput(d, holiday, w, temp_c, humidity, windspeed)

    # --- Soft rules: allowed, but the prediction is less reliable ---
    if d.year < 2011:
        inp.warnings.append("Date is before the training data (2011-2012). Treated as 2011 demand level.")
    elif d.year > 2012:
        inp.warnings.append(
            f"The model only learned 2011-2012. {d.year} is treated as the 2012 demand level, "
            "so real demand may be higher if the service has grown.")
    if w == 4:
        inp.warnings.append("Heavy rain/snow never appears in the training data. Prediction is an extrapolation.")
    for name, val, scale in (("temp", temp_c, TEMP_MAX), ("hum", humidity, HUM_MAX), ("windspeed", windspeed, WIND_MAX)):
        lo, hi = ranges[name][0] * scale, ranges[name][1] * scale
        if not lo <= val <= hi:
            label = {"temp": "Temperature", "hum": "Humidity", "windspeed": "Wind speed"}[name]
            inp.warnings.append(f"{label} {val:g} is outside the training range ({lo:.1f}-{hi:.1f}). Treat with caution.")
    if humidity < 10:
        inp.warnings.append("Humidity below 10 % is very unusual. Check the input or sensor.")
    if temp_c > 25 and w == 3 and "snow" in WEATHER_BY_CODE[3]:
        inp.warnings.append("Light rain/snow above 25 °C - assumed to be rain, not snow.")
    return inp


def to_feature_row(inp: DayInput) -> dict:
    wd = weekday_of(inp.day)
    return {
        "yr": 0 if inp.day.year <= 2011 else 1,
        "holiday": int(inp.holiday),
        "workingday": int(wd not in (0, 6) and not inp.holiday),
        "temp": inp.temp_c / TEMP_MAX,
        "hum": inp.humidity / HUM_MAX,
        "windspeed": inp.windspeed / WIND_MAX,
        "season": season_of(inp.day),
        "mnth": inp.day.month,
        "weekday": wd,
        "weathersit": inp.weather,
    }


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
def demand_level(cnt, q33, q66):
    if cnt < q33:
        return "Low"
    if cnt < q66:
        return "Medium"
    return "High"


def peak_windows(hourly, min_share=0.5):
    """Main morning (before 12:00) and afternoon/evening peaks that reach at least
    min_share of the day's highest hour. Returns a list of peak hours."""
    hourly = np.asarray(hourly, dtype=float)
    if hourly.max() <= 0:
        return []
    peaks = []
    for lo, hi in ((5, 12), (12, 22)):
        h = lo + int(np.argmax(hourly[lo:hi]))
        if hourly[h] >= min_share * hourly.max():
            peaks.append(h)
    # Two "peaks" within 3 hours are one broad peak (e.g. weekends): keep the higher one
    if len(peaks) == 2 and abs(peaks[1] - peaks[0]) <= 3:
        peaks = [max(peaks, key=lambda h: hourly[h])]
    return peaks


def plan(pred, rmse, q33, q66):
    """Turn a predicted rental count into operational recommendations."""
    pred = max(float(pred), 0.0)
    low, high = max(pred - rmse, 0.0), pred + rmse
    level = demand_level(pred, q33, q66)
    staff = max(MIN_STAFF, int(np.ceil(high / RENTALS_PER_STAFF)))
    if level == "Low":
        action = "Good day for maintenance: take extra bikes off the street for servicing."
    elif level == "Medium":
        action = "Normal operations: standard redistribution schedule."
    else:
        action = "Peak day: postpone maintenance, add redistribution runs, keep all bikes in service."
    return {"pred": pred, "low": low, "high": high, "level": level,
            "staff": staff, "action": action}


def predict_day(bundle: ModelBundle, day, holiday, weather, temp_c, humidity, windspeed):
    """Validate one day and return predictions from both models plus a plan."""
    inp = validate(day, holiday, weather, temp_c, humidity, windspeed, bundle.ranges)
    row = pd.DataFrame([to_feature_row(inp)])
    p_mlr, p_ann = bundle.predict_raw(row)
    p_mlr, p_ann = float(p_mlr[0]), float(p_ann[0])
    if min(p_mlr, p_ann) < 0:
        inp.warnings.append("A model predicted a negative count. It was set to 0.")
    hourly = bundle.predict_hourly(to_feature_row(inp))
    preds = {"MLR": max(p_mlr, 0.0), "ANN": max(p_ann, 0.0), "Hourly ANN": float(hourly.sum())}
    m = bundle.meta
    best = preds[bundle.best_model]
    rmse = m["test_metrics"][bundle.best_model]["RMSE"]
    return {"input": inp, "features": row.iloc[0].to_dict(),
            "mlr": preds["MLR"], "ann": preds["ANN"], "hann": preds["Hourly ANN"],
            "hourly": hourly, "peaks": peak_windows(hourly),
            "plan": plan(best, rmse, m["cnt_q33"], m["cnt_q66"])}


REQUIRED_COLS = ["date", "holiday", "weather", "temp_c", "humidity", "windspeed"]
MAX_ROWS = 31


def predict_week(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Batch prediction for a weather forecast file. Bad rows are reported, not fatal."""
    if df is None or len(df) == 0:
        raise ValidationError("The file is empty.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED_COLS)}.")
    if len(df) > MAX_ROWS:
        raise ValidationError(f"Too many rows ({len(df)}). Maximum is {MAX_ROWS} days.")

    out = []
    seen = set()
    for i, r in df.iterrows():
        rec = {"row": i + 1, "date": r["date"]}
        try:
            res = predict_day(bundle, r["date"], r["holiday"], r["weather"],
                              r["temp_c"], r["humidity"], r["windspeed"])
            d = res["input"].day
            if d in seen:
                raise ValidationError("Duplicate date.")
            seen.add(d)
            p = res["plan"]
            rec.update({"date": d.isoformat(), "day": DAY_NAMES[weekday_of(d)],
                        "weather": WEATHER_BY_CODE[res["input"].weather],
                        "MLR": round(res["mlr"]), "ANN": round(res["ann"]), "Hourly ANN": round(res["hann"]),
                        "prediction": round(p["pred"]), "range": f"{p['low']:.0f}-{p['high']:.0f}",
                        "level": p["level"], "staff": p["staff"],
                        "peaks": ", ".join(f"{h:02d}:00" for h in res["peaks"]),
                        "status": "OK" if not res["input"].warnings else "Warning: " + " | ".join(res["input"].warnings)})
        except ValidationError as e:
            rec.update({"status": f"Error: {e}"})
        out.append(rec)
    return pd.DataFrame(out)
