import pandas as pd


class DataLoader:
    @staticmethod
    def load_data(filepath):
        print("\nLOADING DATA...")
        print("-" * 80)

        df = pd.read_csv(filepath)

        print(f"  Dataset loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        print(f"\n  Label distribution:")
        # OBSERVATION: The WELFake Dataset documentation is INCORRECT!
        # Actual content inspection shows: Label 0 = Legit, Label 1 = Fake
        print(f"    Legit news (0): {(df['label'] == 0).sum()}")
        print(f"    Fake news (1): {(df['label'] == 1).sum()}")

        missing = df.isnull().sum()
        if missing.any():
            print(f"\n  Missing values detected:")
            print(missing[missing > 0])
        else:
            print(f"\n  No missing values")

        return df
