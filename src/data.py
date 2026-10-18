"""Data loading and feature engineering for the UCI hydraulic-system
condition-monitoring dataset.

Each of the 2205 operating cycles is 60 s long. Sensors are sampled at
different rates, so every sensor file has shape (2205, n_samples).
profile.txt holds the four component-health targets plus a stability flag.

Reference: Helwig et al., "Condition monitoring of hydraulic systems",
UCI Machine Learning Repository (donated 2018).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SENSORS = ["CE", "CP", "EPS1", "FS1", "FS2",
           "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
           "SE", "TS1", "TS2", "TS3", "TS4", "VS1"]

# Human-readable sensor descriptions (from UCI documentation)
SENSOR_INFO = {
    "PS1": "Pressure, 100 Hz (bar)", "PS2": "Pressure, 100 Hz (bar)",
    "PS3": "Pressure, 100 Hz (bar)", "PS4": "Pressure, 100 Hz (bar)",
    "PS5": "Pressure, 100 Hz (bar)", "PS6": "Pressure, 100 Hz (bar)",
    "TS1": "Temperature, 1 Hz (C)", "TS2": "Temperature, 1 Hz (C)",
    "TS3": "Temperature, 1 Hz (C)", "TS4": "Temperature, 1 Hz (C)",
    "FS1": "Volume flow, 10 Hz (l/min)", "FS2": "Volume flow, 10 Hz (l/min)",
    "EPS1": "Motor power, 100 Hz (W)", "VS1": "Vibration, 1 Hz (mm/s)",
    "SE": "Efficiency factor, 1 Hz (%)", "CE": "Cooling efficiency, 1 Hz (%)",
    "CP": "Cooling power, 1 Hz (kW)",
}

TARGETS = ["cooler", "valve", "pump", "accumulator"]

TARGET_LABELS = {
    "cooler": {3: "close to failure", 20: "reduced efficiency", 100: "full efficiency"},
    "valve": {100: "optimal switching", 90: "small lag", 80: "severe lag", 73: "close to failure"},
    "pump": {0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
    "accumulator": {130: "optimal pressure", 115: "slightly reduced",
                    100: "severely reduced", 90: "close to failure"},
}


def load_profile():
    """Return DataFrame with the four targets + stability flag, one row per cycle."""
    cols = TARGETS + ["stable"]
    df = pd.DataFrame(
        np.loadtxt(os.path.join(DATA_DIR, "profile.txt"), delimiter="\t"),
        columns=cols,
    )
    for c in cols:
        df[c] = df[c].astype(int)
    return df


def load_sensor(name):
    """Return (2205, n_samples) array for one sensor."""
    return np.loadtxt(os.path.join(DATA_DIR, f"{name}.txt"), delimiter="\t")


def signal_features(x):
    """Statistical feature vector for one 1-D cycle signal."""
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x ** 2))
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "median": np.median(x),
        "ptp": np.ptp(x),
        "rms": rms,
        "crest": np.max(np.abs(x)) / (rms + 1e-12),
        "skew": stats.skew(x),
        "kurt": stats.kurtosis(x),
        "energy": np.mean(x ** 2),
    }


def build_feature_frame(sensors=SENSORS, verbose=True):
    """Build one feature row per cycle. Processes sensors one at a time
    to keep memory bounded (raw files total ~500 MB)."""
    feats = {}
    for s in sensors:
        if verbose:
            print(f"  {s} ...", flush=True)
        mat = load_sensor(s)
        cols = {}
        for i in range(mat.shape[0]):
            f = signal_features(mat[i])
            for k, v in f.items():
                cols.setdefault(f"{s}_{k}", []).append(v)
        for k, v in cols.items():
            feats[k] = np.array(v, dtype=float)
        del mat
    return pd.DataFrame(feats)


def sensor_of(feature_name):
    return feature_name.split("_")[0]
