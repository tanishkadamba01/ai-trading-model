import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Inspect a parquet data file.")
    parser.add_argument(
        "--path",
        default="data/raw/btcusdt_1m.parquet",
        help="Path to parquet file.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.path)

    print(df.head())
    print("\nData info:")
    print(df.info())


if __name__ == "__main__":
    main()
