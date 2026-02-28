import pandas as pd


def main():
    data = pd.read_parquet("data/labeled/btcusdt_labeled.parquet")

    # Ensure time order
    data = data.sort_index()

    print("Start:", data.index.min())
    print("End:  ", data.index.max())
    print("Rows: ", len(data))

    n = len(data)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    X_train = train_data.drop(columns=["label"])
    y_train = train_data["label"]

    X_val = val_data.drop(columns=["label"])
    y_val = val_data["label"]

    X_test = test_data.drop(columns=["label"])
    y_test = test_data["label"]

    print("\nSplit sizes:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    print("\nDate ranges:")
    print("Train:", X_train.index.min(), "->", X_train.index.max())
    print("Val:  ", X_val.index.min(), "->", X_val.index.max())
    print("Test: ", X_test.index.min(), "->", X_test.index.max())

    X_train.to_parquet("data/splits/X_train.parquet")
    y_train.to_frame("label").to_parquet("data/splits/y_train.parquet")

    X_val.to_parquet("data/splits/X_val.parquet")
    y_val.to_frame("label").to_parquet("data/splits/y_val.parquet")

    X_test.to_parquet("data/splits/X_test.parquet")
    y_test.to_frame("label").to_parquet("data/splits/y_test.parquet")

    print("Splits saved")


if __name__ == "__main__":
    main()
