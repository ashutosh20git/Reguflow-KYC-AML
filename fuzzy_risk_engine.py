from __future__ import annotations

from typing import Tuple

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Universe definitions ---

# Amount in INR
amount = ctrl.Antecedent(np.linspace(0, 250_000, 1001), "amount")

# IsolationForest anomaly score in [0, 1]
anomaly_score = ctrl.Antecedent(np.linspace(0, 1, 101), "anomaly_score")

# Transaction velocity: tx_count_24h
# In practice this will usually be small (< 50), but we allow up to 100
tx_velocity = ctrl.Antecedent(np.linspace(0, 100, 101), "tx_velocity")

# User profile strength mapped to numeric [0, 10]
# (weak ≈ 0–3, medium ≈ 3–7, strong ≈ 7–10)
user_profile_strength = ctrl.Antecedent(
    np.linspace(0, 10, 101), "user_profile_strength"
)

# Output risk score in [0, 100]
risk = ctrl.Consequent(np.linspace(0, 100, 101), "risk")


# --- Membership functions (parameters explicitly defined) ---

# amount: low / medium / high (INR)
amount["low"] = fuzz.trapmf(amount.universe, [0, 0, 2_000, 6_000])
amount["medium"] = fuzz.trimf(amount.universe, [2_000, 20_000, 80_000])
amount["high"] = fuzz.trapmf(amount.universe, [30_000, 80_000, 250_000, 250_000])

# anomaly_score: low / medium / high
anomaly_score["low"] = fuzz.trapmf(anomaly_score.universe, [0.0, 0.0, 0.2, 0.4])
anomaly_score["medium"] = fuzz.trimf(anomaly_score.universe, [0.2, 0.5, 0.8])
anomaly_score["high"] = fuzz.trapmf(anomaly_score.universe, [0.6, 0.8, 1.0, 1.0])

# tx_velocity (tx_count_24h): low / medium / high
tx_velocity["low"] = fuzz.trapmf(tx_velocity.universe, [0, 0, 1, 3])
tx_velocity["medium"] = fuzz.trimf(tx_velocity.universe, [2, 6, 15])
tx_velocity["high"] = fuzz.trapmf(tx_velocity.universe, [10, 20, 100, 100])

# user_profile_strength: weak / medium / strong
user_profile_strength["weak"] = fuzz.trapmf(
    user_profile_strength.universe, [0, 0, 2, 4]
)
user_profile_strength["medium"] = fuzz.trimf(
    user_profile_strength.universe, [3, 5, 7]
)
user_profile_strength["strong"] = fuzz.trapmf(
    user_profile_strength.universe, [6, 8, 10, 10]
)

# risk: low / medium / high
risk["low"] = fuzz.trapmf(risk.universe, [0, 0, 20, 40])
risk["medium"] = fuzz.trimf(risk.universe, [30, 50, 70])
risk["high"] = fuzz.trapmf(risk.universe, [60, 80, 100, 100])


# --- Rule base (Mamdani max-min inference) ---

rules = []

# 1) High anomaly & high amount -> high risk
rules.append(ctrl.Rule(anomaly_score["high"] & amount["high"], risk["high"]))

# 2) High anomaly & high tx_velocity -> high risk
rules.append(ctrl.Rule(anomaly_score["high"] & tx_velocity["high"], risk["high"]))

# 3) High anomaly & weak profile -> high risk
rules.append(
    ctrl.Rule(anomaly_score["high"] & user_profile_strength["weak"], risk["high"])
)

# 4) High anomaly & medium amount & medium velocity -> high risk
rules.append(
    ctrl.Rule(
        anomaly_score["high"]
        & amount["medium"]
        & tx_velocity["medium"],
        risk["high"],
    )
)

# 5) Medium anomaly & high amount & high velocity -> high risk
rules.append(
    ctrl.Rule(
        anomaly_score["medium"]
        & amount["high"]
        & tx_velocity["high"],
        risk["high"],
    )
)

# 6) Medium anomaly & high amount & weak profile -> high risk
rules.append(
    ctrl.Rule(
        anomaly_score["medium"]
        & amount["high"]
        & user_profile_strength["weak"],
        risk["high"],
    )
)

# 7) Medium anomaly & medium amount & medium velocity -> medium risk
rules.append(
    ctrl.Rule(
        anomaly_score["medium"]
        & amount["medium"]
        & tx_velocity["medium"],
        risk["medium"],
    )
)

# 8) Medium anomaly & medium velocity & strong profile -> medium risk
rules.append(
    ctrl.Rule(
        anomaly_score["medium"]
        & tx_velocity["medium"]
        & user_profile_strength["strong"],
        risk["medium"],
    )
)

# 9) Low anomaly & high amount & high velocity -> medium risk
rules.append(
    ctrl.Rule(
        anomaly_score["low"] & amount["high"] & tx_velocity["high"], risk["medium"]
    )
)

# 10) Low anomaly & medium amount & medium velocity -> medium risk
rules.append(
    ctrl.Rule(
        anomaly_score["low"] & amount["medium"] & tx_velocity["medium"], risk["medium"]
    )
)

# 11) Low anomaly & low amount & low velocity & strong profile -> low risk
rules.append(
    ctrl.Rule(
        anomaly_score["low"]
        & amount["low"]
        & tx_velocity["low"]
        & user_profile_strength["strong"],
        risk["low"],
    )
)

# 12) Low anomaly & low amount & low velocity & medium profile -> low risk
rules.append(
    ctrl.Rule(
        anomaly_score["low"]
        & amount["low"]
        & tx_velocity["low"]
        & user_profile_strength["medium"],
        risk["low"],
    )
)


risk_ctrl_system = ctrl.ControlSystem(rules)
_risk_sim = ctrl.ControlSystemSimulation(risk_ctrl_system)


def evaluate_risk(
    amount_value: float,
    anomaly_score_value: float,
    tx_velocity_value: float,
    user_profile_strength_value: float,
) -> Tuple[str, float]:
    """Evaluate fuzzy risk.

    Returns
    -------
    fuzzy_risk_category : {"low", "medium", "high"}
    crisp_risk_score : float in [0, 100]
    """

    # Clean and clamp inputs to universes; treat NaNs as low/default values
    vals = np.array(
        [amount_value, anomaly_score_value, tx_velocity_value, user_profile_strength_value],
        dtype=float,
    )

    if np.any(np.isnan(vals)):
        # Fallback: no valid inputs -> lowest risk
        return "low", 0.0

    amount_clamped = float(np.clip(vals[0], amount.universe[0], amount.universe[-1]))
    anomaly_clamped = float(
        np.clip(vals[1], anomaly_score.universe[0], anomaly_score.universe[-1])
    )
    velocity_clamped = float(
        np.clip(vals[2], tx_velocity.universe[0], tx_velocity.universe[-1])
    )
    profile_clamped = float(
        np.clip(
            vals[3],
            user_profile_strength.universe[0],
            user_profile_strength.universe[-1],
        )
    )

    try:
        # Reuse global simulation for speed; reset state each call
        sim = _risk_sim
        sim.reset()

        sim.input["amount"] = amount_clamped
        sim.input["anomaly_score"] = anomaly_clamped
        sim.input["tx_velocity"] = velocity_clamped
        sim.input["user_profile_strength"] = profile_clamped

        sim.compute()
        if "risk" not in sim.output:
            raise KeyError("risk")
        crisp_score = float(sim.output["risk"])
    except Exception:
        # Any numerical or inference issue -> safe low-risk default
        return "low", 0.0

    # Derive categorical label from membership degrees at the crisp score
    mu_low = float(fuzz.interp_membership(risk.universe, risk["low"].mf, crisp_score))
    mu_med = float(
        fuzz.interp_membership(risk.universe, risk["medium"].mf, crisp_score)
    )
    mu_high = float(
        fuzz.interp_membership(risk.universe, risk["high"].mf, crisp_score)
    )

    memberships = {"low": mu_low, "medium": mu_med, "high": mu_high}
    fuzzy_category = max(memberships, key=memberships.get)

    return fuzzy_category, crisp_score


if __name__ == "__main__":
    # Simple smoke test so you can run: python fuzzy_risk_engine.py
    cat, score = evaluate_risk(
        amount_value=50_000,
        anomaly_score_value=0.9,
        tx_velocity_value=20,
        user_profile_strength_value=3.0,
    )
    print("[fuzzy] category=", cat, "score=", round(score, 2))
