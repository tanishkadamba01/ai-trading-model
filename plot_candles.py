import pandas as pd
import mplfinance as mpf
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot candlestick chart from parquet data.")
    parser.add_argument(
        "--path",
        default="data/raw/btcusdt_1m.parquet",
        help="Path to OHLCV parquet file.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="Number of latest rows to plot.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.path)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]]

    df_last = df.tail(args.rows)

    mpf.plot(
        df_last,
        type="candle",
        volume=True,
        style="charles",
        title="BTCUSDT 1-minute Candles",
        ylabel="Price (USDT)",
        ylabel_lower="Volume"
    )


if __name__ == "__main__":
    main()
