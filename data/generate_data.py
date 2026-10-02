"""Generate deterministic synthetic potato-farm records for SWE 3513 A2."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "season", "cooperative_support"]
COLUMNS = FEATURES + ["yield_tons", "high_risk"]

def generate_data(seed: int, output_path: str | Path) -> tuple[pd.DataFrame, str]:
    if not 0 <= int(seed) <= 9999:
        raise ValueError("Seed must be a four-digit integer from 0000 to 9999.")
    rng = np.random.default_rng(int(seed))
    n = 240 + (int(seed) % 41)
    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    moisture = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    field_size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), n)
    support = rng.integers(0, 2, n)
    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, n)
    yield_tons = (0.004 * rainfall + 0.06 * moisture + 0.02 * fertilizer + 1.30 * field_size
                  - 0.25 * np.abs(temp - 18) + 0.70 * support + season_effect + noise)
    yield_tons = np.clip(yield_tons, 0.50, None)
    yield_per_ha = yield_tons / field_size
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)
    df = pd.DataFrame({"rainfall_mm": rainfall, "avg_temp_c": temp, "soil_moisture_pct": moisture,
        "fertilizer_kg": fertilizer, "field_size_ha": field_size, "season": season,
        "cooperative_support": support, "yield_tons": yield_tons, "high_risk": high_risk})
    m = round(0.05 * n)
    moisture_idx = rng.choice(n, size=m, replace=False)
    fertilizer_idx = rng.choice(n, size=m, replace=False)
    df.loc[moisture_idx, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_idx, "fertilizer_kg"] = np.nan
    df = pd.concat([df, df.iloc[:3].copy()], ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")
    fingerprint = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return df, fingerprint

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, required=True); p.add_argument('--output', default='data/AI_A2_2527201.csv')
    a = p.parse_args(); frame, digest = generate_data(a.seed, a.output)
    print(f"Rows: {len(frame)} | SHA-256: {digest}")
