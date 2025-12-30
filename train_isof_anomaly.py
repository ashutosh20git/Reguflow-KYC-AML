import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
CONTAMINATION = 0.02

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "time_since_last_tx",
    "tx_count_24h",
    "avg_amount_30d",
    "z_amount_vs_user_baseline",
    "device_change_flag",
    "geo_change_flag",
]


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx_path = base_dir / "synthetic_txns.csv"
    baseline_path = base_dir / "user_baseline_summary.csv"

    df_tx = pd.read_csv(tx_path)
    df_baseline = pd.read_csv(baseline_path)
    return df_tx, df_baseline


def engineer_features(df_tx: pd.DataFrame, df_baseline: pd.DataFrame) -> pd.DataFrame:
    # Basic typing
    df_tx = df_tx.copy()
    df_tx["timestamp"] = pd.to_datetime(df_tx["timestamp"])
    df_tx["amount"] = df_tx["amount"].astype(float)
    df_tx["is_labelled_fraud"] = df_tx["is_labelled_fraud"].astype(int)

    # Merge per-user 30d baseline
    df_baseline = df_baseline.copy()
    df_baseline["avg_amount_30d"] = df_baseline["avg_amount_30d"].astype(float)
    df = df_tx.merge(df_baseline[["user_id", "avg_amount_30d"]], on="user_id", how="left")

    # Per-user mean/std for z-score
    user_stats = (
        df.groupby("user_id")["amount"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "user_amount_mean", "std": "user_amount_std"})
    )
    df = df.join(user_stats, on="user_id")
    df["user_amount_std"] = df["user_amount_std"].replace(0.0, 1e-6)
    df["z_amount_vs_user_baseline"] = (
        df["amount"] - df["user_amount_mean"]
    ) / df["user_amount_std"]

    # Sort by user + time for sequential features
    df = df.sort_values(["user_id", "timestamp", "tx_id"]).reset_index(drop=True)

    # time_since_last_tx (seconds)
    df["time_since_last_tx"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0.0)
    )

    # device_change_flag
    prev_device = df.groupby("user_id")["device_id"].shift(1)
    df["device_change_flag"] = (prev_device != df["device_id"]).fillna(False).astype(int)

    # geo_change_flag based on large jumps in lat/lon
    prev_lat = df.groupby("user_id")["geo_lat"].shift(1)
    prev_lon = df.groupby("user_id")["geo_lon"].shift(1)
    geo_jump = (
        prev_lat.notna()
        & (
            ((df["geo_lat"] - prev_lat).abs() > 1.0)
            | ((df["geo_lon"] - prev_lon).abs() > 1.0)
        )
    )
    df["geo_change_flag"] = geo_jump.astype(int)

    # tx_count_24h using a sliding window per user
    df["timestamp_sec"] = df["timestamp"].astype("int64") // 10**9
    window_sec = 24 * 3600
    tx_count_24h = np.zeros(len(df), dtype="int32")

    for _, group in df.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        times = group["timestamp_sec"].to_numpy()
        dq: deque[int] = deque()
        for row_idx, t in zip(idx, times):
            while dq and t - dq[0] > window_sec:
                dq.popleft()
            dq.append(t)
            tx_count_24h[row_idx] = len(dq)

    df["tx_count_24h"] = tx_count_24h
    df = df.drop(columns=["timestamp_sec", "user_amount_mean", "user_amount_std"])

    # log_amount and clean baseline amount
    df["log_amount"] = np.log1p(df["amount"])
    df["avg_amount_30d"] = df["avg_amount_30d"].fillna(0.0)

    return df


def train_isolation_forest(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].astype(float).values
    y_true = df["is_labelled_fraud"].astype(int).values

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "isof",
                IsolationForest(
                    n_estimators=200,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X)

    # IsolationForest: lower scores = more anomalous; invert and normalise to [0, 1]
    raw_scores = -pipeline.score_samples(X)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        anomaly_score = (raw_scores - min_s) / (max_s - min_s)
    else:
        anomaly_score = np.zeros_like(raw_scores)

    anomaly_label = (pipeline.predict(X) == -1).astype(int)

    precision = precision_score(y_true, anomaly_label, zero_division=0)
    recall = recall_score(y_true, anomaly_label, zero_division=0)
    f1 = f1_score(y_true, anomaly_label, zero_division=0)

    if anomaly_label.sum() > 0:
        threshold = float(anomaly_score[anomaly_label == 1].min())
    else:
        threshold = 1.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "contamination": float(CONTAMINATION),
        "num_true_anomalies": int(y_true.sum()),
        "num_predicted_anomalies": int(anomaly_label.sum()),
        "anomaly_score_threshold": threshold,
    }

    return pipeline, anomaly_score, anomaly_label, metrics


def compute_permutation_importance(
    pipeline: Pipeline,
    df: pd.DataFrame,
    n_repeats: int = 10,
) -> dict:
    """Custom permutation importance using change in raw anomaly scores."""

    X = df[FEATURE_COLUMNS].astype(float).copy()
    rng = np.random.RandomState(RANDOM_SEED)

    base_scores = -pipeline.score_samples(X.values)
    importances: dict[str, dict[str, float]] = {}

    for col in FEATURE_COLUMNS:
        diffs = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_scores = -pipeline.score_samples(X_perm.values)
            diffs.append(float(np.mean(np.abs(base_scores - perm_scores))))
        importances[col] = {
            "importance_mean": float(np.mean(diffs)),
            "importance_std": float(np.std(diffs)),
        }

    return importances


def save_outputs(
    base_dir: Path,
    df: pd.DataFrame,
    anomaly_score: np.ndarray,
    anomaly_label: np.ndarray,
    pipeline: Pipeline,
    importances: dict,
    metrics: dict,
) -> None:
    df = df.copy()
    df["anomaly_score"] = anomaly_score
    df["anomaly_label"] = anomaly_label.astype(int)

    # Preserve ISO-like timestamp format with 'T'
    df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    out_cols = [
        "user_id",
        "tx_id",
        "timestamp_str",
        "amount",
        "currency",
        "tx_type",
        "counterparty_id",
        "device_id",
        "geo_lat",
        "geo_lon",
        "balance_after",
        "is_labelled_fraud",
        "anomaly_score",
        "anomaly_label",
    ]

    df_out = df[out_cols].rename(columns={"timestamp_str": "timestamp"})
    df_out.to_csv(base_dir / "tx_with_scores.csv", index=False)

    # Save model (pipeline includes StandardScaler + IsolationForest)
    with open(base_dir / "isof_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # Save feature importance and metrics
    feature_list = [
        {
            "name": name,
            "importance_mean": importances[name]["importance_mean"],
            "importance_std": importances[name]["importance_std"],
        }
        for name in FEATURE_COLUMNS
    ]

    meta = {
        "random_seed": RANDOM_SEED,
        "features": feature_list,
        "metrics": metrics,
    }

    with open(base_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    df_tx, df_baseline = load_data(base_dir)
    df = engineer_features(df_tx, df_baseline)

    pipeline, anomaly_score, anomaly_label, metrics = train_isolation_forest(df)
    importances = compute_permutation_importance(pipeline, df)

    save_outputs(base_dir, df, anomaly_score, anomaly_label, pipeline, importances, metrics)

    print("Training complete.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
