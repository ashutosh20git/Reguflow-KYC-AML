import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, date

N_USERS = 500
TARGET_ROWS = 50000
DAYS_BACK = 180
CURRENCY = "INR"
NEW_DEVICE_PROB = 0.03
GEO_JUMP_PROB = 0.05

NUM_SMURF_EPISODES = 50
NUM_BIG_EPISODES = 100
NUM_DEVICE_GEO_EVENTS = 200

RANDOM_SEED = 42


def create_user_profiles():
    profiles = []
    for i in range(1, N_USERS + 1):
        user_id = f"U{i:04d}"
        r = random.random()
        if r < 0.6:
            tier = "low"
            activity_factor = random.uniform(0.3, 0.9)
        elif r < 0.9:
            tier = "medium"
            activity_factor = random.uniform(0.9, 2.0)
        else:
            tier = "high"
            activity_factor = random.uniform(2.0, 4.0)

        amount_scale = random.uniform(0.7, 1.3)
        start_hour = random.randint(6, 12)
        window = random.randint(8, 14)
        end_hour = min(start_hour + window, 23)

        base_lat = random.uniform(8.0, 28.0)
        base_lon = random.uniform(72.0, 88.0)

        device_count = random.randint(1, 3)
        base_devices = [f"D_{user_id}_{j + 1}" for j in range(device_count)]

        starting_balance = random.uniform(20000.0, 200000.0)

        profile = {
            "user_id": user_id,
            "tier": tier,
            "activity_factor": activity_factor,
            "amount_scale": amount_scale,
            "active_start": start_hour,
            "active_end": end_hour,
            "base_lat": base_lat,
            "base_lon": base_lon,
            "base_devices": base_devices,
            "all_devices": set(base_devices),
            "new_device_counter": 0,
            "starting_balance": starting_balance,
        }
        profiles.append(profile)
    return profiles


def base_tx_count_for_user(profile):
    if profile["tier"] == "low":
        base_min, base_max = 5, 60
    elif profile["tier"] == "medium":
        base_min, base_max = 60, 180
    else:
        base_min, base_max = 180, 350
    base = random.randint(base_min, base_max)
    return max(1, int(base * profile["activity_factor"]))


def random_datetime_for_user(profile, start_date, days):
    day_offset = random.randint(0, days - 1)
    base_day = start_date + timedelta(days=day_offset)
    start_hour = profile["active_start"]
    end_hour = profile["active_end"]
    if end_hour <= start_hour:
        end_hour = start_hour + 1
    hour = random.randint(start_hour, min(end_hour, 23))
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime.combine(base_day, datetime.min.time()) + timedelta(
        hours=hour, minutes=minute, seconds=second
    )


def sample_amount_small(profile):
    mu = 5.0
    sigma = 0.9
    value = random.lognormvariate(mu, sigma) * profile["amount_scale"]
    value = max(50.0, min(value, 2000.0))
    return round(value, 2)


def sample_amount_large(profile):
    mu = 9.0
    sigma = 0.7
    value = random.lognormvariate(mu, sigma) * profile["amount_scale"]
    value = max(10000.0, min(value, 200000.0))
    return round(value, 2)


def sample_amount_normal(profile):
    if random.random() < 0.97:
        return sample_amount_small(profile)
    return sample_amount_large(profile)


def choose_device(profile, force_new=False, new_prob=NEW_DEVICE_PROB):
    if force_new or random.random() < new_prob:
        profile["new_device_counter"] += 1
        device_id = f"D_{profile['user_id']}_X{profile['new_device_counter']}"
        profile["all_devices"].add(device_id)
        return device_id, True
    return random.choice(profile["base_devices"]), False


def sample_geo(profile, force_jump=False):
    if force_jump or random.random() < GEO_JUMP_PROB:
        lat_shift = random.choice([-1.0, 1.0]) * random.uniform(5.0, 15.0)
        lon_shift = random.choice([-1.0, 1.0]) * random.uniform(5.0, 15.0)
        lat = max(-90.0, min(90.0, profile["base_lat"] + lat_shift))
        lon = max(-180.0, min(180.0, profile["base_lon"] + lon_shift))
    else:
        lat = profile["base_lat"] + random.gauss(0.0, 0.1)
        lon = profile["base_lon"] + random.gauss(0.0, 0.1)
    return round(lat, 6), round(lon, 6)


def random_counterparty_id():
    prefix = random.choice(["MCH", "WLT", "BNK", "EXT"])
    num = random.randint(10000, 999999)
    return f"{prefix}{num}"


def generate_base_transactions(profiles, start_date, days, temp_id_start=0):
    txs = []
    temp_id = temp_id_start
    for profile in profiles:
        count = base_tx_count_for_user(profile)
        for _ in range(count):
            dt = random_datetime_for_user(profile, start_date, days)
            amount = sample_amount_normal(profile)
            tx_type = "debit" if random.random() < 0.55 else "credit"
            device_id, _ = choose_device(profile)
            lat, lon = sample_geo(profile)
            tx = {
                "user_id": profile["user_id"],
                "dt": dt,
                "amount": amount,
                "currency": CURRENCY,
                "tx_type": tx_type,
                "counterparty_id": random_counterparty_id(),
                "device_id": device_id,
                "geo_lat": lat,
                "geo_lon": lon,
                "is_labelled_fraud": 0,
                "fraud_type": None,
                "fraud_episode_id": None,
                "temp_id": temp_id,
            }
            temp_id += 1
            txs.append(tx)
    return txs, temp_id


def generate_smurfing_anomalies(profiles, start_date, days, temp_id_start):
    txs = []
    temp_id = temp_id_start
    for episode_id in range(1, NUM_SMURF_EPISODES + 1):
        profile = random.choice(profiles)
        day_offset = random.randint(0, days - 1)
        base_day = start_date + timedelta(days=day_offset)
        num_tx = random.randint(20, 40)
        for _ in range(num_tx):
            hour = random.randint(profile["active_start"], min(profile["active_end"], 23))
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            dt = datetime.combine(base_day, datetime.min.time()) + timedelta(
                hours=hour, minutes=minute, seconds=second
            )
            amount = sample_amount_small(profile)
            device_id, _ = choose_device(profile)
            lat, lon = sample_geo(profile)
            tx = {
                "user_id": profile["user_id"],
                "dt": dt,
                "amount": amount,
                "currency": CURRENCY,
                "tx_type": "credit",
                "counterparty_id": random_counterparty_id(),
                "device_id": device_id,
                "geo_lat": lat,
                "geo_lon": lon,
                "is_labelled_fraud": 1,
                "fraud_type": "smurfing",
                "fraud_episode_id": episode_id,
                "temp_id": temp_id,
            }
            temp_id += 1
            txs.append(tx)
    return txs, temp_id


def generate_big_in_out_anomalies(profiles, start_date, days, temp_id_start):
    txs = []
    temp_id = temp_id_start
    for episode_id in range(1, NUM_BIG_EPISODES + 1):
        profile = random.choice(profiles)
        dt_deposit = random_datetime_for_user(profile, start_date, days)
        delta_minutes = random.randint(5, 6 * 60)
        dt_withdraw = dt_deposit + timedelta(minutes=delta_minutes)
        amount_in = sample_amount_large(profile)
        amount_out = round(amount_in * random.uniform(0.8, 1.0), 2)
        device_id, _ = choose_device(profile)
        lat, lon = sample_geo(profile)

        tx_deposit = {
            "user_id": profile["user_id"],
            "dt": dt_deposit,
            "amount": amount_in,
            "currency": CURRENCY,
            "tx_type": "credit",
            "counterparty_id": random_counterparty_id(),
            "device_id": device_id,
            "geo_lat": lat,
            "geo_lon": lon,
            "is_labelled_fraud": 1,
            "fraud_type": "big_inflow_outflow",
            "fraud_episode_id": episode_id,
            "temp_id": temp_id,
        }
        temp_id += 1
        txs.append(tx_deposit)

        tx_withdraw = {
            "user_id": profile["user_id"],
            "dt": dt_withdraw,
            "amount": amount_out,
            "currency": CURRENCY,
            "tx_type": "debit",
            "counterparty_id": random_counterparty_id(),
            "device_id": device_id,
            "geo_lat": lat,
            "geo_lon": lon,
            "is_labelled_fraud": 1,
            "fraud_type": "big_inflow_outflow",
            "fraud_episode_id": episode_id,
            "temp_id": temp_id,
        }
        temp_id += 1
        txs.append(tx_withdraw)

    return txs, temp_id


def generate_device_geo_anomalies(profiles, start_date, days, temp_id_start):
    txs = []
    temp_id = temp_id_start
    for _ in range(NUM_DEVICE_GEO_EVENTS):
        profile = random.choice(profiles)
        dt = random_datetime_for_user(profile, start_date, days)
        amount = sample_amount_large(profile)
        device_id, _ = choose_device(profile, force_new=True)
        lat, lon = sample_geo(profile, force_jump=True)
        tx = {
            "user_id": profile["user_id"],
            "dt": dt,
            "amount": amount,
            "currency": CURRENCY,
            "tx_type": "debit" if random.random() < 0.5 else "credit",
            "counterparty_id": random_counterparty_id(),
            "device_id": device_id,
            "geo_lat": lat,
            "geo_lon": lon,
            "is_labelled_fraud": 1,
            "fraud_type": "device_geo_anomaly",
            "fraud_episode_id": None,
            "temp_id": temp_id,
        }
        temp_id += 1
        txs.append(tx)
    return txs, temp_id


def adjust_transaction_count_to_target(txs, target_rows, profiles, start_date, days, temp_id_start):
    temp_id = temp_id_start
    n = len(txs)
    if n == target_rows:
        return txs, temp_id
    if n > target_rows:
        diff = n - target_rows
        normal_indices = [i for i, tx in enumerate(txs) if tx["is_labelled_fraud"] == 0]
        if diff <= len(normal_indices):
            drop_indices = set(random.sample(normal_indices, diff))
        else:
            drop_indices = set(random.sample(range(n), diff))
        txs = [tx for idx, tx in enumerate(txs) if idx not in drop_indices]
        return txs, temp_id

    needed = target_rows - n
    for _ in range(needed):
        profile = random.choice(profiles)
        dt = random_datetime_for_user(profile, start_date, days)
        amount = sample_amount_normal(profile)
        tx_type = "debit" if random.random() < 0.55 else "credit"
        device_id, _ = choose_device(profile)
        lat, lon = sample_geo(profile)
        tx = {
            "user_id": profile["user_id"],
            "dt": dt,
            "amount": amount,
            "currency": CURRENCY,
            "tx_type": tx_type,
            "counterparty_id": random_counterparty_id(),
            "device_id": device_id,
            "geo_lat": lat,
            "geo_lon": lon,
            "is_labelled_fraud": 0,
            "fraud_type": None,
            "fraud_episode_id": None,
            "temp_id": temp_id,
        }
        temp_id += 1
        txs.append(tx)
    return txs, temp_id


def assign_tx_ids_and_balances(txs, profiles):
    profile_map = {p["user_id"]: p for p in profiles}
    txs_sorted = sorted(txs, key=lambda tx: (tx["user_id"], tx["dt"], tx["temp_id"]))
    user_balance = {}
    rows = []
    for idx, tx in enumerate(txs_sorted):
        user_id = tx["user_id"]
        profile = profile_map[user_id]
        balance = user_balance.get(user_id, profile["starting_balance"])
        amount = tx["amount"]
        if tx["tx_type"] == "credit":
            balance += amount
        else:
            balance -= amount
        user_balance[user_id] = balance
        row = {
            "row_index": idx,
            "user_id": user_id,
            "tx_id": f"T{idx + 1:08d}",
            "timestamp": tx["dt"].isoformat(timespec="seconds"),
            "amount": round(amount, 2),
            "currency": tx["currency"],
            "tx_type": tx["tx_type"],
            "counterparty_id": tx["counterparty_id"],
            "device_id": tx["device_id"],
            "geo_lat": tx["geo_lat"],
            "geo_lon": tx["geo_lon"],
            "balance_after": round(balance, 2),
            "is_labelled_fraud": tx["is_labelled_fraud"],
            "fraud_type": tx["fraud_type"],
            "fraud_episode_id": tx["fraud_episode_id"],
        }
        rows.append(row)
    return rows


def write_transactions_csv(rows, path):
    fieldnames = [
        "user_id",
        "tx_id",
        "timestamp",
        "amount",
        "currency",
        "tx_type",
        "counterparty_id",
        "device_id",
        "geo_lat",
        "geo_lon",
        "balance_after",
        "is_labelled_fraud",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow(
                [
                    row["user_id"],
                    row["tx_id"],
                    row["timestamp"],
                    f"{row['amount']:.2f}",
                    row["currency"],
                    row["tx_type"],
                    row["counterparty_id"],
                    row["device_id"],
                    f"{row['geo_lat']:.6f}",
                    f"{row['geo_lon']:.6f}",
                    f"{row['balance_after']:.2f}",
                    row["is_labelled_fraud"],
                ]
            )


def write_user_baseline_summary(rows, profiles, path):
    datetimes = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    max_dt = max(datetimes)
    cutoff = max_dt - timedelta(days=30)

    amount_sum_30 = defaultdict(float)
    count_30 = defaultdict(int)
    device_counts = defaultdict(Counter)
    geo_sum_lat = defaultdict(float)
    geo_sum_lon = defaultdict(float)
    geo_count = defaultdict(int)

    for r, dt in zip(rows, datetimes):
        user_id = r["user_id"]
        if dt >= cutoff:
            amount_sum_30[user_id] += r["amount"]
            count_30[user_id] += 1
        device_counts[user_id][r["device_id"]] += 1
        geo_sum_lat[user_id] += r["geo_lat"]
        geo_sum_lon[user_id] += r["geo_lon"]
        geo_count[user_id] += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_id",
            "avg_amount_30d",
            "avg_freq_30d",
            "primary_device",
            "primary_geo",
        ])
        for p in profiles:
            user_id = p["user_id"]
            total_amt = amount_sum_30.get(user_id, 0.0)
            count = count_30.get(user_id, 0)
            avg_amt = total_amt / count if count > 0 else 0.0
            avg_freq = count / 30.0
            device_counter = device_counts.get(user_id, Counter())
            primary_device = ""
            if device_counter:
                primary_device = device_counter.most_common(1)[0][0]
            if geo_count.get(user_id, 0) > 0:
                lat = geo_sum_lat[user_id] / geo_count[user_id]
                lon = geo_sum_lon[user_id] / geo_count[user_id]
            else:
                lat = p["base_lat"]
                lon = p["base_lon"]
            primary_geo = f"{lat:.6f},{lon:.6f}"
            writer.writerow(
                [
                    user_id,
                    f"{avg_amt:.2f}",
                    f"{avg_freq:.4f}",
                    primary_device,
                    primary_geo,
                ]
            )


def write_metadata_json(rows, seed, path):
    smurf_episodes = defaultdict(list)
    big_episodes = defaultdict(list)
    device_geo_indices = []

    for r in rows:
        idx = r["row_index"]
        ft = r["fraud_type"]
        if ft == "smurfing":
            smurf_episodes[r["fraud_episode_id"]].append(idx)
        elif ft == "big_inflow_outflow":
            big_episodes[r["fraud_episode_id"]].append(idx)
        elif ft == "device_geo_anomaly":
            device_geo_indices.append(idx)

    head_sample = [
        {
            "row_index": r["row_index"],
            "user_id": r["user_id"],
            "tx_id": r["tx_id"],
            "timestamp": r["timestamp"],
            "amount": r["amount"],
            "tx_type": r["tx_type"],
            "is_labelled_fraud": r["is_labelled_fraud"],
            "fraud_type": r["fraud_type"],
        }
        for r in rows[:5]
    ]

    fraud_sample = [
        {
            "row_index": r["row_index"],
            "user_id": r["user_id"],
            "tx_id": r["tx_id"],
            "timestamp": r["timestamp"],
            "amount": r["amount"],
            "tx_type": r["tx_type"],
            "is_labelled_fraud": r["is_labelled_fraud"],
            "fraud_type": r["fraud_type"],
        }
        for r in rows
        if r["is_labelled_fraud"] == 1
    ][:10]

    meta = {
        "random_seed": seed,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "num_users": len({r["user_id"] for r in rows}),
        "num_transactions": len(rows),
        "index_base": 0,
        "columns": [
            "user_id",
            "tx_id",
            "timestamp",
            "amount",
            "currency",
            "tx_type",
            "counterparty_id",
            "device_id",
            "geo_lat",
            "geo_lon",
            "balance_after",
            "is_labelled_fraud",
        ],
        "anomalies": {
            "smurfing": {
                "episodes": len(smurf_episodes),
                "episode_indices": {str(k): v for k, v in smurf_episodes.items()},
            },
            "big_inflow_outflow": {
                "episodes": len(big_episodes),
                "episode_indices": {str(k): v for k, v in big_episodes.items()},
            },
            "device_geo_anomaly": {
                "events": len(device_geo_indices),
                "transaction_indices": device_geo_indices,
            },
        },
        "samples": {
            "head": head_sample,
            "fraud": fraud_sample,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    random.seed(RANDOM_SEED)
    today = date.today()
    start_date = today - timedelta(days=DAYS_BACK)
    profiles = create_user_profiles()

    temp_id = 0
    base_txs, temp_id = generate_base_transactions(profiles, start_date, DAYS_BACK, temp_id)
    smurf_txs, temp_id = generate_smurfing_anomalies(profiles, start_date, DAYS_BACK, temp_id)
    big_txs, temp_id = generate_big_in_out_anomalies(profiles, start_date, DAYS_BACK, temp_id)
    device_geo_txs, temp_id = generate_device_geo_anomalies(profiles, start_date, DAYS_BACK, temp_id)

    all_txs = base_txs + smurf_txs + big_txs + device_geo_txs
    all_txs, temp_id = adjust_transaction_count_to_target(
        all_txs, TARGET_ROWS, profiles, start_date, DAYS_BACK, temp_id
    )

    rows = assign_tx_ids_and_balances(all_txs, profiles)

    write_transactions_csv(rows, "synthetic_txns.csv")
    write_user_baseline_summary(rows, profiles, "user_baseline_summary.csv")
    write_metadata_json(rows, RANDOM_SEED, "synthetic_txns_metadata.json")
    print(f"Generated {len(rows)} transactions.")


if __name__ == "__main__":
    main()
