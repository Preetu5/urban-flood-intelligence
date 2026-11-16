"""Regenerate data/demo_flood_data.csv with the project's fixed demo seed."""

from flood_intelligence.data import DATA_PATH, generate_demo_data


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data().to_csv(DATA_PATH, index=False)
    print(f"Wrote 1,200 synthetic demonstration rows to {DATA_PATH}")
