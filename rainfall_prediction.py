"""Flask app for Rainfall Forecasting + Fuzzy Flood Risk.

GitHub-friendly version:
- Uses relative paths (./data, ./models, ./templates)
- Supports PORT env var (Render/Heroku style)
- Optionally auto-downloads model artifacts from Google Drive if missing

Environment variables (optional, for auto-download):
  GDRIVE_MODEL_ID
  GDRIVE_FEATURE_SCALER_ID
  GDRIVE_RAIN_SCALER_ID
  GDRIVE_FEATURE_COLS_ID
  GDRIVE_META_ID

If any artifact is missing and the corresponding env var is not set, the app will raise a
clear error telling you what to provide.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import threading
import pandas as pd
import tensorflow as tf
import joblib
from flask import Flask, jsonify, render_template, request
from subsystem_b import FloodDetector
from werkzeug.utils import secure_filename
import io

model_lock = threading.Lock()
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

# Read-only training/backfill dataset
TRAIN_CSV_PATH = DATA_DIR / "weather_data.csv"

# Observed runtime feature inputs (1 row per input, 10-min sampling)
INPUT_CSV_PATH = DATA_DIR / "input.csv"

# Optional user-provided future rainfall scenario (10-min sampling)
BULK_CSV_PATH = DATA_DIR / "bulk_save.csv"
BULK_CURSOR_PATH = DATA_DIR / "bulk_cursor.json"

# Auto-roll rainfall history (1 value per 10-min tick)
OBS_RAIN_PATH = DATA_DIR / "observed_rain.csv"

# Store last forecast so we can roll ONLY the first step next time
LAST_FORECAST_PATH = DATA_DIR / "last_forecast.json"

MODEL_PATH = MODELS_DIR / "rainfall_model.keras"

# -----------------------------
# Constants (must be before model load)
# -----------------------------
N_FEATURES          = 19
SEQUENCE_LENGTH     = 120
HORIZON_STEPS       = 18
DATA_FREQ_MIN       = 10
RECENT_WINDOW_STEPS = HORIZON_STEPS

FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
RAIN_SCALER_PATH = MODELS_DIR / "rain_scaler.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"
META_PATH = MODELS_DIR / "model_meta.json"

FEATURE_COLS: List[str] = [
    "p", "T", "Tpot", "Tdew", "rh", "VPmax", "VPact", "VPdef",
    "sh", "H2OC", "rho", "wv", "max. wv", "wd", "rain", "raining",
    "SWDR", "PAR", "Tlog"
]

# -----------------------------
# Load model (local only - no Google Drive)
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}.\n"
        f"Please place rainfall_model.keras inside the models/ folder."
    )

for p in [FEATURE_SCALER_PATH, RAIN_SCALER_PATH, FEATURE_COLS_PATH, META_PATH]:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required file: {p}. Please place the trained scaler/meta files inside the models/ folder."
        )

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(str(MODEL_PATH))
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
rain_scaler = joblib.load(RAIN_SCALER_PATH)

with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

SEQUENCE_LENGTH = int(META["sequence_length"])
HORIZON_STEPS = int(META["horizon_steps"])
DATA_FREQ_MIN = int(META.get("data_frequency_minutes", 10))
RECENT_WINDOW_STEPS = HORIZON_STEPS
N_FEATURES = len(FEATURE_COLS)

print("Model loaded successfully!")

# Warm up model
dummy_input = np.zeros((1, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
_ = model(dummy_input, training=False)
print("Model warm-up complete!")

# Load Subsystem B (CNN Flood Detection)
CNN_MODEL_PATH = MODELS_DIR / "best_flood_early_warning_unet.pt"
flood_detector = FloodDetector(str(CNN_MODEL_PATH))


# -----------------------------
# Input validation ranges (UI + API)
# -----------------------------
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
    # raining should be 0 or 1 ideally
    if "raining" in row:
        rv = coerce_float(row.get("raining"), 0.0)
        if rv not in (0.0, 1.0):
            errs.append("raining must be 0 or 1")
    return errs


# -----------------------------
# Cursor helpers for scenario simulation
# -----------------------------

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


# -----------------------------
# Auto-roll helpers
# -----------------------------

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
    """Recent rain window used by fuzzy risk (train rain + rolled-in rain)."""
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


# -----------------------------
# Feature window builder
# -----------------------------

def prep_features(df: pd.DataFrame | None, df_cols: List[str]) -> pd.DataFrame | None:
    if df is None:
        return None
    available_cols = [c for c in df_cols if c in df.columns]
    if len(available_cols) == 0:
        return None
    feat_df = df[available_cols].apply(pd.to_numeric, errors="coerce").dropna()

    # Ensure all model cols exist
    for col in df_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    return feat_df[df_cols]


def load_recent_history(df_cols: List[str]) -> Tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    LSTM feature history window (PAST ONLY):
      - Prefer INPUT_CSV_PATH
      - If not enough, backfill from TRAIN_CSV_PATH
    Returns:
      feat_hist: DataFrame length (SEQUENCE_LENGTH - 1)
      recent_rain: np array length RECENT_WINDOW_STEPS
    """
    history_needed = SEQUENCE_LENGTH - 1

    train_df = pd.read_csv(TRAIN_CSV_PATH) if TRAIN_CSV_PATH.exists() else None
    input_df = pd.read_csv(INPUT_CSV_PATH) if INPUT_CSV_PATH.exists() else None

    train_feat = prep_features(train_df, df_cols)
    input_feat = prep_features(input_df, df_cols)

    input_rows = 0 if input_feat is None else len(input_feat)
    use_input = min(history_needed, input_rows)

    parts = []
    if use_input > 0:
        parts.append(input_feat.tail(use_input))

    remaining = history_needed - use_input
    if remaining > 0:
        if train_feat is None or len(train_feat) < remaining:
            return None, None
        parts.insert(0, train_feat.tail(remaining))

    feat_hist = pd.concat(parts, ignore_index=True)
    recent_rain = load_recent_rain_series()
    return feat_hist, recent_rain


def build_feature_row_from_json(data: dict) -> dict:
    """Build a single feature row matching training feature columns (19 features)."""
    row = {}
    for col in FEATURE_COLS:
        val = data.get(col, data.get(col.replace(". ", "_"), 0.0))
        row[col] = coerce_float(val, 0.0)
    return row


def run_model_forecast(window_df: pd.DataFrame) -> np.ndarray:
    """
    Run the trained LSTM forecast on a full feature window and convert output back to rainfall mm.
    """
    for col in FEATURE_COLS:
        if col not in window_df.columns:
            window_df[col] = 0.0

    feature_values = window_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if len(feature_values) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected {SEQUENCE_LENGTH} rows for prediction, got {len(feature_values)}")

    scaled_input = feature_scaler.transform(feature_values)
    input_sequence = scaled_input.reshape((1, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)

    with model_lock:
        prediction = model(input_sequence, training=False).numpy().reshape(-1, 1)

    pred_series = rain_scaler.inverse_transform(prediction).reshape(-1)
    pred_series = np.maximum(0.0, pred_series)

    if len(pred_series) != HORIZON_STEPS:
        raise ValueError(f"Expected {HORIZON_STEPS} forecast steps, got {len(pred_series)}")

    return pred_series



# -----------------------------
# Forecast sources
# -----------------------------

def get_simulated_forecast_series() -> Tuple[np.ndarray | None, str | None]:
    """Use bulk_save.csv as the future 18-step forecast; advances cursor by 1 per call."""
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


# -----------------------------
# FUZZY LOGIC risk assessment (Mamdani)
# Inputs: forecast_total, forecast_peak, step_delta (this-call peak - previous-call peak)
# Output: flood risk in [0,1] + label
# -----------------------------

def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    # c < x < d
    return (d - x) / (d - c)


def _trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a + 1e-12)
    idx = (b <= x) & (x < c)
    y[idx] = (c - x[idx]) / (c - b + 1e-12)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def _trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a + 1e-12)
    idx = (b <= x) & (x <= c)
    y[idx] = 1.0
    idx = (c < x) & (x < d)
    y[idx] = (d - x[idx]) / (d - c + 1e-12)
    return np.clip(y, 0.0, 1.0)


def _norm01(x, cap):
    if cap <= 0:
        return 0.0
    return float(max(0.0, min(float(x) / float(cap), 1.0)))


# =========================================================
# Fuzzy flood risk engine
# Uses:
#   1) forecast total rainfall over next 3 hours
#   2) peak rainfall intensity in forecast
#   3) recent observed rainfall accumulation
#   4) step delta = current next-step forecast - previous next-step forecast
# Returns:
#   risk score (0~1), alert level, explanation lines, and summary details
# =========================================================
def fuzzy_flood_risk(pred_series: np.ndarray, recent_series: np.ndarray | None, step_delta: float):
    # Safety clamp: prediction values should not go below 0
    pred_series = np.maximum(np.array(pred_series, dtype=float), 0.0)

    # Core values used by fuzzy rules
    forecast_total = float(np.sum(pred_series))                           # next 180 min accumulation
    peak = float(np.max(pred_series)) if len(pred_series) else 0.0       # strongest 10-min step
    recent_total = float(np.sum(recent_series)) if recent_series is not None else 0.0  # past 180 min accumulation

    # -----------------------------
    # Helper membership functions
    # -----------------------------
    def tri(x, a, b, c):
        # Triangular membership function
        x = float(x)
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    def trap(x, a, b, c, d):
        # Trapezoidal membership function
        x = float(x)
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (d - x) / (d - c)

    def centroid(xs, mus):
        # Defuzzification using centroid method
        num = 0.0
        den = 0.0
        for x, mu in zip(xs, mus):
            mu = float(mu)
            num += x * mu
            den += mu
        return 0.0 if den <= 1e-9 else num / den

    # -----------------------------
    # Input memberships
    # -----------------------------
    # Forecast total rainfall memberships
    total_low  = trap(forecast_total, 0.0, 0.0, 2.0, 6.0)
    total_med  = tri (forecast_total, 4.0, 10.0, 16.0)
    total_high = trap(forecast_total, 12.0, 18.0, 40.0, 60.0)

    # Peak intensity memberships
    peak_low  = trap(peak, 0.0, 0.0, 0.5, 1.2)
    peak_med  = tri (peak, 0.8, 1.8, 3.5)
    peak_high = trap(peak, 2.5, 4.0, 8.0, 15.0)

    # Recent rainfall history memberships
    recent_low  = trap(recent_total, 0.0, 0.0, 1.0, 4.0)
    recent_med  = tri (recent_total, 2.0, 6.0, 12.0)
    recent_high = trap(recent_total, 8.0, 12.0, 30.0, 60.0)

    # Forecast trend memberships (step-to-step change)
    delta_down = trap(step_delta, -5.0, -5.0, -0.20, -0.02)
    delta_flat = tri (step_delta, -0.08, 0.0, 0.08)
    delta_up   = trap(step_delta, 0.02, 0.15, 5.0, 5.0)

    # -----------------------------
    # Output risk universe
    # -----------------------------
    xs = np.linspace(0.0, 1.0, 401)

    # Output fuzzy sets for final risk
    risk_low  = np.array([trap(x, 0.0, 0.0, 0.18, 0.38) for x in xs])
    risk_med  = np.array([tri (x, 0.25, 0.50, 0.75) for x in xs])
    risk_high = np.array([trap(x, 0.62, 0.80, 1.0, 1.0) for x in xs])

    # -----------------------------
    # Fuzzy rule base
    # -----------------------------
    rules = []

    # R1: low forecast + low peak + low recent rain => low risk
    rules.append((
        "R1(total_low AND peak_low AND recent_low => low)",
        min(total_low, peak_low, recent_low),
        risk_low
    ))

    # R2: medium forecast total OR medium recent rain => medium risk
    rules.append((
        "R2(total_med OR recent_med => med)",
        max(total_med, recent_med),
        risk_med
    ))

    # R3: rising trend + (medium peak OR medium total) => medium risk
    rules.append((
        "R3(delta_rising AND (peak_med OR total_med) => med)",
        min(delta_up, max(peak_med, total_med)),
        risk_med
    ))

    # R4: high total OR high peak => high risk
    rules.append((
        "R4(total_high OR peak_high => high)",
        max(total_high, peak_high),
        risk_high
    ))

    # R5: high recent rain + (medium total OR medium peak) => high risk
    rules.append((
        "R5(recent_high AND (total_med OR peak_med) => high)",
        min(recent_high, max(total_med, peak_med)),
        risk_high
    ))

    # R6: rising trend + high total => high risk
    rules.append((
        "R6(delta_rising AND total_high => high)",
        min(delta_up, total_high),
        risk_high
    ))

    # -----------------------------
    # Rule aggregation
    # -----------------------------
    agg = np.zeros_like(xs)
    fired = []

    for name, strength, shape in rules:
        if strength > 0:
            # Mamdani inference: clip rule output by rule strength, then aggregate with max
            agg = np.maximum(agg, np.minimum(float(strength), shape))
            fired.append((name, float(strength)))

    # Final crisp risk value
    risk = centroid(xs, agg)

    # -----------------------------
    # Risk label mapping
    # -----------------------------
    if risk < 0.30:
        level = "Green"
    elif risk < 0.55:
        level = "Yellow"
    elif risk < 0.80:
        level = "Orange"
    else:
        level = "Red"

    # Keep only top 3 strongest fired rules for explanation
    fired = sorted(fired, key=lambda x: x[1], reverse=True)[:3]

    # -----------------------------
    # Explanation text
    # -----------------------------
    explanation = [
        f"Forecast accumulation (next 180 min): {forecast_total:.2f} mm",
        f"Peak intensity: {peak:.2f} mm per 10 min",
        f"Recent accumulation (past 180 min): {recent_total:.2f} mm",
        f"Step delta (current next-step - previous next-step): {step_delta:.2f} mm",
    ]

    if fired:
        explanation.append(
            "Top fuzzy rules fired: " +
            "; ".join([f"{name}: {strength:.2f}" for name, strength in fired])
        )

    # -----------------------------
    # Extra summary data for UI/debug
    # -----------------------------
    summary = {
        "forecast_total_mm": forecast_total,
        "forecast_peak_mm_per_step": peak,
        "recent_total_mm": recent_total,
        "step_delta_mm": step_delta,
        "risk_engine": "fuzzy_flood_risk_mamdani",
        "memberships": {
            "total": {
                "low": total_low,
                "med": total_med,
                "high": total_high,
            },
            "peak": {
                "low": peak_low,
                "med": peak_med,
                "high": peak_high,
            },
            "recent": {
                "low": recent_low,
                "med": recent_med,
                "high": recent_high,
            },
            "delta": {
                "down": delta_down,
                "flat": delta_flat,
                "up": delta_up,
            }
        }
    }

    return float(risk), level, explanation, summary


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/subsystem_a")
def subsystem_a_page():
    """Subsystem A page - LSTM rainfall forecasting."""
    return render_template("index_old.html")

@app.route("/docs")
def documentation():
    return render_template("documentation.html")


@app.route("/data")
def view_data():
    return render_template("data_view.html")


@app.route("/subsystem_b")
def subsystem_b_page():
    """Subsystem B page - CNN satellite flood detection."""
    return render_template("subsystem_b.html")


@app.route("/save", methods=["POST"])
def save_data():
    """Save ONE observed feature input row into input.csv."""
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
    """Upload/append future rainfall scenario into bulk_save.csv."""
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

        return jsonify(
            {
                "message": f"Scenario uploaded ({len(df)} rows).",
                "path": str(BULK_CSV_PATH),
                "cursor_idx": _read_cursor(),
                "note": "Simulation consumes 1 step per prediction call.",
            }
        )
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
    """Reset auto-rolled rain history and last forecast."""
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
    """Predict 3-hour rainfall series + fuzzy flood risk."""
    try:
        data = request.json or {}
        simulation_mode = bool(data.get("simulation_mode", False))
        auto_roll = bool(data.get("auto_roll", True))

        current_row = build_feature_row_from_json(data)
        errs = validate_feature_ranges(current_row)
        if errs:
            return jsonify({"error": "Invalid input: " + ", ".join(errs)}), 400

        current_df = pd.DataFrame([current_row])

        # Build past 119 feature rows so the LSTM receives the same type of input
        feat_hist, recent_rain = load_recent_history(FEATURE_COLS)
        if feat_hist is None:
            return jsonify({
                "error": (
                    "Not enough past history to form the LSTM window. "
                    f"Need at least {SEQUENCE_LENGTH - 1} observed rows (input.csv) OR enough rows in weather_data.csv for backfill."
                )
            }), 400

        window_df = pd.concat([feat_hist, current_df], ignore_index=True)
        for col in FEATURE_COLS:
            if col not in window_df.columns:
                window_df[col] = 0.0
        window_df = window_df[FEATURE_COLS]

        rolled_in_mm = None
        prev_first_step = None

        if simulation_mode:
            pred_series, err = get_simulated_forecast_series()
            if err is not None:
                return jsonify({"error": err}), 400
            pred_series = np.array(pred_series, dtype=float)
            mode_used = "simulation"
        else:
            # Auto-roll: move ONLY previous forecast first-step into observed rain history
            if auto_roll:
                prev = _read_last_forecast()
                if prev is not None and len(prev) > 0:
                    prev_first_step = float(prev[0])
                    rolled_in_mm = float(prev_first_step)
                    _append_observed_rain(rolled_in_mm)

            # Run model on full history window (119 past rows + current row)
            pred_series = run_model_forecast(window_df)
            pred_series = np.array(pred_series, dtype=float)

            if auto_roll:
                _write_last_forecast(pred_series)

            mode_used = "model"

        # compare current next-step vs previous next-step
        step_delta = 0.0
        if prev_first_step is not None and len(pred_series) > 0:
            step_delta = float(pred_series[0]) - float(prev_first_step)

        risk_conf, alert_level, explanation, forecast_summary = fuzzy_flood_risk(
            pred_series,
            recent_rain,
            step_delta,
        )

        return jsonify(
            {
                "mode": mode_used,
                "risk_engine": forecast_summary.get("risk_engine", "fuzzy_flood_risk_mamdani"),
                "auto_roll": auto_roll,
                "rolled_in_rain_mm": rolled_in_mm,
                "forecast_series_mm": pred_series.tolist(),
                "flood_risk_confidence": float(risk_conf),
                "alert_level": alert_level,
                "explanation": explanation,
                "forecast_horizon_hours": round((HORIZON_STEPS * DATA_FREQ_MIN) / 60, 2),
                "forecast_total_mm": float(forecast_summary.get("forecast_total_mm", np.sum(pred_series))),
                "forecast_peak_mm_per_step": float(forecast_summary.get("forecast_peak_mm_per_step", np.max(pred_series))),
                "recent_total_mm": float(forecast_summary.get("recent_total_mm", 0.0)),
                "step_delta_mm": float(forecast_summary.get("step_delta_mm", step_delta)),
                "memberships": forecast_summary.get("memberships", {}),
                "data_frequency_minutes": DATA_FREQ_MIN,
                "simulation_cursor_idx": _read_cursor() if mode_used == "simulation" else None,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_image", methods=["POST"])
def predict_image():
    """Predict flood from uploaded satellite image (Subsystem B: CNN)."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Predict using CNN
        result, error = flood_detector.predict(image_bytes)
        
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify({
            "mode": "cnn_satellite",
            "subsystem": "B",
            "flood_percentage": result['flood_percentage'],
            "avg_confidence": result['avg_confidence'],
            "max_confidence": result['max_confidence'],
            "alert_level": result['risk_level'],
            "risk_confidence": result['risk_confidence'],
            "explanation": result['explanation']
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
        # Replace NaN with 0 to avoid JSON serialization issues
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


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", "5000"))
#     app.run(host="0.0.0.0", port=port, debug=True)
