from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
from plotly.subplots import make_subplots

# Use non-interactive backend for server-side PNG generation
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "tx_with_fuzzy.csv"
META_PATH = BASE_DIR / "feature_importance.json"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_threshold(default: float = 0.5) -> float:
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("metrics", {}).get("anomaly_score_threshold", default))
    except FileNotFoundError:
        return default


def compute_user_window(
    df: pd.DataFrame, user_id: str, days: int = 30, threshold: float = 0.5
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return (user_30, daily_agg, anomalies) for user over last N days.

    If the user has no transactions, returns (None, None, None).
    If the user has transactions but none in the last N days, returns (user_df, None, None).
    """

    user_df = df[df["user_id"] == user_id].copy().sort_values("timestamp")
    if user_df.empty:
        return None, None, None

    end_time = user_df["timestamp"].max()
    start_time = end_time - pd.Timedelta(days=days)
    mask = user_df["timestamp"].between(start_time, end_time)
    user_30 = user_df.loc[mask].copy()
    if user_30.empty:
        return user_df, None, None

    daily = (
        user_30.set_index("timestamp")["amount"]
        .resample("D")
        .sum()
        .rename("amount_sum")
        .to_frame()
    )
    daily["rolling_mean"] = daily["amount_sum"].rolling(window=7, min_periods=1).mean()
    daily["rolling_std"] = (
        daily["amount_sum"].rolling(window=7, min_periods=1).std().fillna(0.0)
    )
    daily["upper_band"] = daily["rolling_mean"] + 2 * daily["rolling_std"]
    daily["lower_band"] = (daily["rolling_mean"] - 2 * daily["rolling_std"]).clip(
        lower=0.0
    )

    is_if_anom = user_30["anomaly_score"] >= threshold
    if "fuzzy_risk_category" in user_30.columns:
        is_fuzzy_high = (
            user_30["fuzzy_risk_category"].astype(str).str.lower().eq("high")
        )
    else:
        is_fuzzy_high = pd.Series(False, index=user_30.index)

    user_30["is_anomaly_flag"] = is_if_anom | is_fuzzy_high
    anom = user_30[user_30["is_anomaly_flag"]].copy()
    return user_30, daily, anom


def create_matplotlib_dashboard(
    user_id: str,
    user_30: pd.DataFrame,
    daily: pd.DataFrame,
    anom: pd.DataFrame,
    threshold: float,
    out_path: Path,
) -> None:
    fig, (ax_main, ax_sub) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_main.plot(daily.index, daily["amount_sum"], label="Daily sum", color="C0")
    ax_main.plot(daily.index, daily["rolling_mean"], label="7d mean", color="C1")
    ax_main.fill_between(
        daily.index,
        daily["lower_band"],
        daily["upper_band"],
        color="C1",
        alpha=0.2,
        label="7d mean ± 2 std",
    )

    ax_main.scatter(
        user_30["timestamp"],
        user_30["amount"],
        s=20,
        color="gray",
        alpha=0.7,
        label="Tx",
    )

    if anom is not None and not anom.empty:
        ax_main.scatter(
            anom["timestamp"], anom["amount"], s=40, color="red", label="Anomaly"
        )

    ax_main.set_ylabel("Amount (INR)")
    ax_main.set_title(f"User {user_id} – last 30 days")
    ax_main.legend(loc="upper left")

    ax_sub.plot(
        user_30["timestamp"],
        user_30["anomaly_score"],
        color="C2",
        label="anomaly_score",
    )
    ax_sub.axhline(threshold, color="red", linestyle="--", alpha=0.6, label="threshold")
    ax_sub.set_ylabel("Anomaly score")
    ax_sub.set_xlabel("Date")
    ax_sub.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_plotly_dashboard(
    user_id: str,
    user_30: pd.DataFrame,
    daily: pd.DataFrame,
    anom: pd.DataFrame,
    threshold: float,
    out_path: Path,
) -> str:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
    )

    # Row 1: daily sums and bands
    fig.add_trace(
        go.Scatter(x=daily.index, y=daily["amount_sum"], mode="lines", name="Daily sum"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=daily.index, y=daily["rolling_mean"], mode="lines", name="7d mean"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["upper_band"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["lower_band"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Band ±2 std",
            hoverinfo="skip",
            opacity=0.2,
        ),
        row=1,
        col=1,
    )

    # Transaction points
    fig.add_trace(
        go.Scatter(
            x=user_30["timestamp"],
            y=user_30["amount"],
            mode="markers",
            marker=dict(size=6, color="gray"),
            name="Tx",
        ),
        row=1,
        col=1,
    )

    # Anomalies
    if anom is not None and not anom.empty:
        fig.add_trace(
            go.Scatter(
                x=anom["timestamp"],
                y=anom["amount"],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Anomaly",
            ),
            row=1,
            col=1,
        )

    # Row 2: anomaly score
    fig.add_trace(
        go.Scatter(
            x=user_30["timestamp"],
            y=user_30["anomaly_score"],
            mode="lines+markers",
            marker=dict(size=5),
            name="anomaly_score",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=threshold, line=dict(color="red", dash="dash"), row=2, col=1
    )

    fig.update_layout(
        title=f"User {user_id} – last 30 days anomaly dashboard",
        xaxis_title="Date",
        yaxis_title="Amount (INR)",
        height=800,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")

    # Return HTML snippet (without full page) for embedding
    from plotly.io import to_html

    return to_html(fig, include_plotlyjs="cdn", full_html=False)


@app.route("/")
def index() -> "flask.Response":
    # Serve existing index.html so current UI continues to work
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/user_dashboard")
def user_dashboard() -> "flask.Response":
    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return redirect(url_for("index"))

    if not DATA_PATH.exists():
        return "tx_with_fuzzy.csv not found. Run the training and fuzzy pipeline first.", 500

    df = load_data()
    threshold = load_threshold()

    user_30, daily, anom = compute_user_window(df, user_id=user_id, days=30, threshold=threshold)
    if user_30 is None:
        return f"No transactions found for user {user_id}.", 404
    if daily is None:
        return (
            f"User {user_id} has transactions but none in the last 30 days. "
            "Try a different user.",
            200,
        )

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    html_path = STATIC_DIR / "user_last30_anomaly.html"

    plot_div = create_plotly_dashboard(
        user_id=user_id,
        user_30=user_30,
        daily=daily,
        anom=anom,
        threshold=threshold,
        out_path=html_path,
    )

    template = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>User {{ user_id }} anomaly dashboard</title>
    <style>
      body { font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; background: #020617; color: #e5e7eb; margin: 0; padding: 1.5rem; }
      a { color: #38bdf8; }
      .container { max-width: 1120px; margin: 0 auto; }
      img { max-width: 100%; height: auto; border-radius: 0.75rem; box-shadow: 0 20px 40px rgba(0,0,0,0.45); margin-bottom: 1.5rem; }
      h1, h2 { margin-bottom: 0.5rem; }
      .back { margin-bottom: 1rem; }
      .panel { background: #020617; border-radius: 1rem; padding: 1.25rem 1.5rem; border: 1px solid #1f2937; box-shadow: 0 18px 40px rgba(15,23,42,0.9); margin-bottom: 1.5rem; }
    </style>
  </head>
  <body>
    <div class=\"container\">
      <div class=\"back\"><a href=\"{{ index_url }}\">&#8592; Back to home</a></div>
      <h1>User {{ user_id }} – last 30 days anomaly dashboard</h1>
      <p>Thresholded IsolationForest anomaly score and fuzzy risk are used to flag anomalies.</p>
      <div class=\"panel\">
        <h2>Interactive anomaly chart</h2>
        {{ plot_div | safe }}
      </div>
    </div>
  </body>
</html>"""

    return render_template_string(
        template,
        user_id=user_id,
        index_url=url_for("index"),
        plot_div=plot_div,
    )


if __name__ == "__main__":
    # Run `python app.py` and open http://127.0.0.1:5000/
    app.run(debug=True)
