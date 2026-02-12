"""Flask app for Rainfall Forecasting + Fuzzy Flood Risk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# App + Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

TRAIN_CSV_PATH = DATA_DIR / "weather_data.csv"
INPUT_CSV_PATH = DATA_DIR / "input.csv"
BULK_CSV_PATH = DATA_DIR / "bulk_save.csv"
BULK_CURSOR_PATH = DATA_DIR / "bulk_cursor.json"
OBS_RAIN_PATH = DATA_DIR / "observed_rain.csv"
LAST_FORECAST_PATH = DATA_DIR / "last_forecast.json"
MODEL_PATH = MODELS_DIR / "rainfall_model.keras"

# -----------------------------
# Load model
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}.\n"
        f"Please place rainfall_model.keras inside the models/ folder."
    )

print(f"Loading model from {MODEL_PATH}...")
import tf_keras

# -----------------------------
# PATCH 1: Fix legacy 'batch_shape' in InputLayer config
# Older Keras saved models use 'batch_shape'; newer tf_keras expects 'batch_input_shape'
# -----------------------------
try:
    from tf_keras.src.engine.input_layer import InputLayer as _InputLayer
    _orig_input_from_config = _InputLayer.from_config.__func__

    @classmethod
    def _patched_input_from_config(cls, config):
        if 'batch_shape' in config:
            config = dict(config)
            config['batch_input_shape'] = config.pop('batch_shape')
        return _orig_input_from_config(cls, config)

    _InputLayer.from_config = _patched_input_from_config
    print("PATCH 1 applied: InputLayer batch_shape fix.")
except Exception as e:
    print(f"Warning: PATCH 1 failed: {e}")

# -----------------------------
# PATCH 2: Fix 'str has no attribute name' in mixed_precision policy
# base_layer.py imports get_policy directly so we must patch _set_dtype_policy instead
# -----------------------------
try:
    from tf_keras.src.engine import base_layer as _base_layer
    from tf_keras.src.mixed_precision import policy as _mp_policy

    _orig_set_dtype_policy = _base_layer.BaseLayer._set_dtype_policy

    def _patched_set_dtype_policy(self, dtype):
        if isinstance(dtype, str):
            try:
                dtype = _mp_policy.Policy(dtype)
            except Exception:
                pass
        _orig_set_dtype_policy(self, dtype)

    _base_layer.BaseLayer._set_dtype_policy = _patched_set_dtype_policy
    print("PATCH 2 applied: _set_dtype_policy string->Policy fix.")
except Exception as e:
    print(f"Warning: PATCH 2 failed: {e}")

model = tf_keras.models.load_model(str(MODEL_PATH), compile=False)
print("Model loaded successfully!")

# Simple scaler - 19 features (matching original model)
N_FEATURES = 19
scaler = MinMaxScaler()
scaler.fit([[0] * N_FEATURES, [100] * N_FEATURES])

SEQUENCE_LENGTH     = 120
HORIZON_STEPS       = 18
DATA_FREQ_MIN       = 10
RECENT_WINDOW_STEPS = HORIZON_STEPS

FEATURE_COLS: List[str] = [
    "p", "T", "Tpot", "Tdew", "rh", "VPmax", "VPact", "VPdef",
    "sh", "H2OC", "rho", "wv", "max. wv", "wd", "rain", "raining",
    "SWDR", "PAR", "Tlog"
]

FEATURE_RANGES = {
    "T": (-10.0, 60.0),
    "rh": (0.0, 100.0),
    "p": (850.0, 1100.0),
    "wv": (0.0, 60.0),
    "max. wv": (0.0, 100.0),
    "raining": (0.0, 1.0),
}


def coerce_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def validate_feature_ranges(row: dict) -> List[str]:
    errs: List[str] = []
    for k, (lo, hi) in FEATURE_RANGES.items():
        if k not in row:
            continue
        v = coerce_float(row.get(k), None)
        if v is None:
            errs.append(f"{k} must be numeric")
            continue
        if v < lo or v > hi:
            errs.append(f"{k} out of range [{lo}, {hi}] (got {v})")
    if "raining" in row:
        rv = coerce_float(row.get("raining"), 0.0)
        if rv not in (0.0, 1.0):
            errs.append("raining must be 0 or 1")
    return errs


def _read_cursor() -> int:
    if not BULK_CURSOR_PATH.exists():
        return 0
    try:
        obj = json.loads(BULK_CURSOR_PATH.read_text(encoding="utf-8"))
        return int(obj.get("idx", 0))
    except Exception:
        return 0


def _write_cursor(idx: int) -> None:
    BULK_CURSOR_PATH.write_text(json.dumps({"idx": int(idx)}, indent=2), encoding="utf-8")


def _read_last_forecast() -> List[float] | None:
    if not LAST_FORECAST_PATH.exists():
        return None
    try:
        obj = json.loads(LAST_FORECAST_PATH.read_text(encoding="utf-8"))
        arr = obj.get("forecast_series_mm")
        if not isinstance(arr, list):
            return None
        return [float(x) for x in arr]
    except Exception:
        return None


def _write_last_forecast(series: np.ndarray) -> None:
    LAST_FORECAST_PATH.write_text(
        json.dumps(
            {
                "forecast_series_mm": [float(x) for x in series.reshape(-1).tolist()],
                "horizon_steps": int(HORIZON_STEPS),
                "data_frequency_minutes": int(DATA_FREQ_MIN),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _append_observed_rain(value_mm: float) -> None:
    df = pd.DataFrame([{"rain_mm": float(value_mm)}])
    df.to_csv(OBS_RAIN_PATH, mode="a", index=False, header=not OBS_RAIN_PATH.exists())


def load_recent_rain_series() -> np.ndarray | None:
    parts = []
    if TRAIN_CSV_PATH.exists():
        try:
            train_df = pd.read_csv(TRAIN_CSV_PATH)
            if "rain" in train_df.columns:
                train_rain = pd.to_numeric(train_df["rain"], errors="coerce").dropna().to_numpy(dtype=np.float32)
                if len(train_rain) > 0:
                    parts.append(train_rain)
        except Exception:
            pass
    if OBS_RAIN_PATH.exists():
        try:
            obs_df = pd.read_csv(OBS_RAIN_PATH)
            if "rain_mm" in obs_df.columns:
                obs_rain = pd.to_numeric(obs_df["rain_mm"], errors="coerce").dropna().to_numpy(dtype=np.float32)
                if len(obs_rain) > 0:
                    parts.append(obs_rain)
        except Exception:
            pass
    if not parts:
        return None
    full = np.concatenate(parts, axis=0)
    if len(full) < RECENT_WINDOW_STEPS:
        return None
    return full[-RECENT_WINDOW_STEPS:].astype(np.float32)


def build_feature_row_from_json(data: dict) -> dict:
    row = {}
    for col in FEATURE_COLS:
        val = data.get(col, data.get(col.replace(". ", "_"), 0.0))
        row[col] = coerce_float(val, 0.0)
    return row


def run_model_forecast(window_df: pd.DataFrame) -> np.ndarray:
    row = window_df.iloc[-1]
    feature_values = [coerce_float(row[c] if c in row.index else 0.0) for c in FEATURE_COLS]
    scaled_input = scaler.transform([feature_values])[0]
    input_sequence = np.array([scaled_input] * SEQUENCE_LENGTH)
    input_sequence = input_sequence.reshape((1, SEQUENCE_LENGTH, N_FEATURES))
    prediction = model.predict(input_sequence, verbose=0)
    base_val = float(prediction[0][0])
    series = np.maximum(0, np.array([base_val] * HORIZON_STEPS, dtype=float))
    return series


def get_simulated_forecast_series() -> Tuple[np.ndarray | None, str | None]:
    if not BULK_CSV_PATH.exists():
        return None, "bulk_save.csv not found. Upload a scenario first."
    df = pd.read_csv(BULK_CSV_PATH)
    if "rain" in df.columns:
        col = "rain"
    elif "rain_mm" in df.columns:
        col = "rain_mm"
    else:
        return None, "bulk_save.csv must contain a 'rain' or 'rain_mm' column."
    series = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if len(series) < HORIZON_STEPS:
        return None, f"bulk_save.csv needs at least {HORIZON_STEPS} rows."
    idx = _read_cursor()
    if idx + HORIZON_STEPS > len(series):
        return None, "Simulation scenario exhausted. Reset cursor or upload a longer scenario."
    forecast = series[idx : idx + HORIZON_STEPS].astype(float)
    _write_cursor(idx + 1)
    return forecast, None


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def _trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def fuzzy_risk_mamdani(forecast_total: float, forecast_peak: float, step_delta: float) -> Tuple[float, str, List[str]]:
    total_low = _trapmf(forecast_total, 0, 0, 3, 8)
    total_med = _trimf(forecast_total, 5, 12, 20)
    total_high = _trapmf(forecast_total, 15, 22, 35, 60)
    peak_low = _trapmf(forecast_peak, 0, 0, 0.8, 2.0)
    peak_med = _trimf(forecast_peak, 1.5, 3.0, 5.0)
    peak_high = _trapmf(forecast_peak, 4.0, 6.0, 10.0, 20.0)
    delta_down = _trapmf(step_delta, -5.0, -5.0, -0.25, 0.0)
    delta_flat = _trimf(step_delta, -0.15, 0.0, 0.15)
    delta_up = _trapmf(step_delta, 0.0, 0.25, 5.0, 5.0)

    y = np.linspace(0.0, 1.0, 401)

    def out_low(yy):
        return np.array([_trapmf(v, 0.0, 0.0, 0.15, 0.35) for v in yy])

    def out_med(yy):
        return np.array([_trimf(v, 0.25, 0.50, 0.75) for v in yy])

    def out_high(yy):
        return np.array([_trapmf(v, 0.65, 0.80, 1.0, 1.0) for v in yy])

    r1 = min(total_high, peak_high)
    r2 = min(total_high, max(peak_med, delta_up))
    r3 = min(total_med, peak_high)
    r4 = min(total_med, peak_med)
    r5 = min(total_high, peak_low)
    r6 = min(total_low, peak_low)
    r7 = min(total_low, max(peak_med, peak_high))
    r8 = min(total_med, peak_low)
    r9 = min(delta_down, max(total_low, peak_low))

    agg = np.zeros_like(y)
    agg = np.maximum(agg, np.minimum(r1, out_high(y)))
    agg = np.maximum(agg, np.minimum(r2, out_high(y)))
    agg = np.maximum(agg, np.minimum(r3, out_high(y)))
    agg = np.maximum(agg, np.minimum(r4, out_med(y)))
    agg = np.maximum(agg, np.minimum(r5, out_med(y)))
    agg = np.maximum(agg, np.minimum(r7, out_med(y)))
    agg = np.maximum(agg, np.minimum(r6, out_low(y)))
    agg = np.maximum(agg, np.minimum(r8, out_low(y)))
    agg = np.maximum(agg, np.minimum(r9, out_low(y)))

    if float(np.sum(agg)) <= 1e-9:
        risk = 0.0
    else:
        risk = float(np.sum(y * agg) / np.sum(agg))

    if risk < 0.30:
        level = "Green"
    elif risk < 0.55:
        level = "Yellow"
    elif risk < 0.80:
        level = "Orange"
    else:
        level = "Red"

    why = [
        f"Fuzzy inputs: total={forecast_total:.2f}mm, peak={forecast_peak:.2f}mm/10min, step_delta={step_delta:+.2f}",
        f"Memberships: total(L/M/H)=({total_low:.2f},{total_med:.2f},{total_high:.2f}), "
        f"peak(L/M/H)=({peak_low:.2f},{peak_med:.2f},{peak_high:.2f}), "
        f"delta(down/flat/up)=({delta_down:.2f},{delta_flat:.2f},{delta_up:.2f})",
    ]
    return risk, level, why


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/docs")
def documentation():
    return render_template("documentation.html")


@app.route("/data")
def view_data():
    return render_template("data_view.html")


@app.route("/save", methods=["POST"])
def save_data():
    try:
        data = request.json or {}
        row = build_feature_row_from_json(data)
        errs = validate_feature_ranges(row)
        if errs:
            return jsonify({"error": "Invalid input: " + ", ".join(errs)}), 400
        df = pd.DataFrame([row])
        df.to_csv(INPUT_CSV_PATH, mode="a", index=False, header=not INPUT_CSV_PATH.exists())
        return jsonify({"message": "Observed input saved successfully!", "path": str(INPUT_CSV_PATH)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/bulk_upload", methods=["POST"])
def bulk_upload():
    try:
        data = request.json or {}
        overwrite = bool(data.get("overwrite", False))
        rows = []
        if "rain_series" in data and isinstance(data["rain_series"], list):
            for v in data["rain_series"]:
                rows.append({"rain_mm": coerce_float(v, 0.0)})
        else:
            rain_mm = coerce_float(data.get("rain_mm", 0.0), 0.0)
            count = int(data.get("count", 0))
            if count <= 0:
                return jsonify({"error": "Provide rain_series (list) OR rain_mm + count"}), 400
            rows = [{"rain_mm": rain_mm} for _ in range(count)]
        df = pd.DataFrame(rows)
        if overwrite and BULK_CSV_PATH.exists():
            BULK_CSV_PATH.unlink()
        df.to_csv(BULK_CSV_PATH, mode="a", index=False, header=not BULK_CSV_PATH.exists())
        if overwrite:
            _write_cursor(0)
        return jsonify({
            "message": f"Scenario uploaded ({len(df)} rows).",
            "path": str(BULK_CSV_PATH),
            "cursor_idx": _read_cursor(),
            "note": "Simulation consumes 1 step per prediction call.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/bulk_reset", methods=["POST"])
def bulk_reset():
    try:
        _write_cursor(0)
        return jsonify({"message": "Simulation cursor reset to 0.", "cursor_idx": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset_auto_history", methods=["POST"])
def reset_auto_history():
    try:
        if OBS_RAIN_PATH.exists():
            OBS_RAIN_PATH.unlink()
        if LAST_FORECAST_PATH.exists():
            LAST_FORECAST_PATH.unlink()
        return jsonify({"message": "Auto history cleared (observed_rain.csv + last_forecast.json removed)."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        simulation_mode = bool(data.get("simulation_mode", False))
        auto_roll = bool(data.get("auto_roll", True))

        current_row = build_feature_row_from_json(data)
        errs = validate_feature_ranges(current_row)
        if errs:
            return jsonify({"error": "Invalid input: " + ", ".join(errs)}), 400

        current_df = pd.DataFrame([current_row])
        rolled_in_mm = None
        mode_used = "model"

        if simulation_mode:
            pred_series, err = get_simulated_forecast_series()
            if err:
                return jsonify({"error": err}), 400
            pred_series = np.array(pred_series, dtype=float)
            mode_used = "simulation"
        else:
            prev = _read_last_forecast() if auto_roll else None
            prev_first = float(prev[0]) if (prev and len(prev) > 0) else None
            if auto_roll and prev_first is not None:
                rolled_in_mm = prev_first
                _append_observed_rain(rolled_in_mm)
            pred_series = run_model_forecast(current_df)
            if auto_roll:
                _write_last_forecast(pred_series)

        prev_peak = float(max(_read_last_forecast() or [pred_series[0]])) if auto_roll else float(pred_series[0])
        curr_peak = float(np.max(pred_series))
        step_delta = float(curr_peak - prev_peak)

        forecast_total = float(np.sum(np.maximum(pred_series, 0.0)))
        forecast_peak = float(np.max(np.maximum(pred_series, 0.0)))

        risk_conf, alert_level, fuzzy_why = fuzzy_risk_mamdani(forecast_total, forecast_peak, step_delta)

        explanation = [
            f"Forecast accumulation (next {HORIZON_STEPS * DATA_FREQ_MIN} min): {forecast_total:.2f} mm",
            f"Peak intensity: {forecast_peak:.2f} mm per {DATA_FREQ_MIN} min",
            f"Step delta (peak vs last call): {step_delta:+.2f} mm",
        ]
        explanation.extend(fuzzy_why)

        return jsonify({
            "mode": mode_used,
            "risk_engine": "fuzzy_mamdani",
            "auto_roll": auto_roll,
            "rolled_in_rain_mm": rolled_in_mm,
            "forecast_series_mm": pred_series.tolist(),
            "flood_risk_confidence": float(risk_conf),
            "alert_level": alert_level,
            "explanation": explanation,
            "forecast_horizon_hours": round((HORIZON_STEPS * DATA_FREQ_MIN) / 60, 2),
            "forecast_total_mm": forecast_total,
            "forecast_peak_mm_per_step": forecast_peak,
            "data_frequency_minutes": DATA_FREQ_MIN,
            "simulation_cursor_idx": _read_cursor() if mode_used == "simulation" else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data")
def get_data():
    try:
        import math
        csv_path = INPUT_CSV_PATH if INPUT_CSV_PATH.exists() else TRAIN_CSV_PATH
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        if not csv_path.exists():
            return jsonify({"data": [], "total": 0, "page": page, "per_page": per_page, "total_pages": 0})
        df = pd.read_csv(csv_path)
        df = df.fillna(0)
        total = len(df)
        total_pages = math.ceil(total / per_page)
        start = (page - 1) * per_page
        return jsonify({
            "data": df.iloc[start: start + per_page].to_dict("records"),
            "total": total, "page": page,
            "per_page": per_page, "total_pages": total_pages,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)