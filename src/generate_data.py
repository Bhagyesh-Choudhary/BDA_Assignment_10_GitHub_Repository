from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def main():
    # Synthetic, intentionally imbalanced dataset; not real customer data.
    X, y = make_classification(
        n_samples=12000, n_features=12, n_informative=7, n_redundant=3,
        n_repeated=0, n_classes=2, weights=[0.985, 0.015],
        class_sep=1.15, flip_y=0.005, random_state=42
    )
    columns = [f"feature_{i}" for i in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=columns)
    rng = np.random.default_rng(42)
    df.insert(0, "transaction_id", [f"TX{i:06d}" for i in range(len(df))])
    df["amount"] = np.round(np.exp(rng.normal(3.0, 1.0, len(df))), 2)
    df["label"] = y.astype(int)
    df.to_csv(DATA / "transactions.csv", index=False)
    print(f"Saved {len(df)} synthetic transactions to {DATA / 'transactions.csv'}")
    print(f"Fraud rate: {df['label'].mean():.3%}")

if __name__ == "__main__":
    main()
