"""Send sample requests to the API so Grafana / logs have something to show.

Usage: python scripts/send_traffic.py --url http://localhost:8000 --n 100
"""
import argparse
import random
import time

import requests


def random_patient(rng: random.Random) -> dict:
    return {
        "age": rng.randint(29, 77), "sex": rng.randint(0, 1), "cp": rng.randint(1, 4),
        "trestbps": rng.randint(94, 200), "chol": rng.randint(126, 450),
        "fbs": rng.randint(0, 1), "restecg": rng.randint(0, 2),
        "thalach": rng.randint(71, 202), "exang": rng.randint(0, 1),
        "oldpeak": round(rng.uniform(0, 5), 1), "slope": rng.randint(1, 3),
        "ca": rng.randint(0, 3), "thal": rng.choice([3, 6, 7]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(7)
    counts = {0: 0, 1: 0}
    for _ in range(args.n):
        resp = requests.post(f"{args.url}/predict", json=random_patient(rng), timeout=10)
        resp.raise_for_status()
        counts[resp.json()["prediction"]] += 1
        time.sleep(args.sleep)
    requests.post(f"{args.url}/predict", json={"age": -1}, timeout=10)  # one 4xx for the dashboard
    print(f"sent {args.n} requests; predictions: {counts}")


if __name__ == "__main__":
    main()
