from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from fuzzy_risk_engine import evaluate_risk


MAX_ROWS = 1_000


def add_tx_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute 24h transaction count per user as tx_velocity."""

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp", "tx_id"]).reset_index(drop=True)

    df["timestamp_sec"] = df["timestamp"].astype("int64") // 10**9

    window_sec = 24 * 3600
    tx_velocity = np.zeros(len(df), dtype="int32")

    for _, group in df.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        times = group["timestamp_sec"].to_numpy()
        dq: deque[int] = deque()
        for row_idx, t in zip(idx, times):
            while dq and t - dq[0] > window_sec:
                dq.popleft()
            dq.append(t)
            tx_velocity[row_idx] = len(dq)

    df["tx_velocity"] = tx_velocity
    df = df.drop(columns=["timestamp_sec"])
    return df


def map_user_profile_strength(avg_freq_30d: float, avg_amount_30d: float) -> Tuple[str, float]:
    """Map per-user baseline into categorical + numeric profile strength.

    Returns (label, numeric_score_in_0_10).
    """

    if pd.isna(avg_freq_30d) or pd.isna(avg_amount_30d):
        return "weak", 2.0

    # Very light / low-value usage
    if avg_freq_30d < 0.3 and avg_amount_30d < 2_000:
        return "weak", 2.0

    # Strong profile: frequent or higher-value usage
    if avg_freq_30d >= 2.0 or avg_amount_30d >= 8_000:
        return "strong", 8.0

    # Everything else: medium
    return "medium", 5.0


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    tx_path = base_dir / "tx_with_scores.csv"
    baseline_path = base_dir / "user_baseline_summary.csv"

    df = pd.read_csv(tx_path)
    # Prototype: only process first MAX_ROWS transactions to keep runtime low
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
    df = add_tx_velocity(df)

    # Attach baseline per user for profile strength
    baseline = pd.read_csv(baseline_path)
    df = df.merge(
        baseline[["user_id", "avg_amount_30d", "avg_freq_30d"]],
        on="user_id",
        how="left",
    )

    labels = []
    scores = []
    for _, row in df.iterrows():
        lbl, sc = map_user_profile_strength(
            avg_freq_30d=row.get("avg_freq_30d", np.nan),
            avg_amount_30d=row.get("avg_amount_30d", np.nan),
        )
        labels.append(lbl)
        scores.append(sc)

    df["user_profile_strength_label"] = labels
    df["user_profile_strength_score"] = scores

    # Evaluate fuzzy risk for each transaction
    fuzzy_categories = []
    fuzzy_scores = []
    for _, row in df.iterrows():
        cat, score = evaluate_risk(
            amount_value=float(row["amount"]),
            anomaly_score_value=float(row["anomaly_score"]),
            tx_velocity_value=float(row["tx_velocity"]),
            user_profile_strength_value=float(row["user_profile_strength_score"]),
        )
        fuzzy_categories.append(cat)
        fuzzy_scores.append(score)

    df["fuzzy_risk_category"] = fuzzy_categories
    df["fuzzy_risk_score"] = fuzzy_scores

    out_path = base_dir / "tx_with_fuzzy.csv"
    df.to_csv(out_path, index=False)

    print("Fuzzy pipeline complete. Wrote:", out_path)


if __name__ == "__main__":
    main()
