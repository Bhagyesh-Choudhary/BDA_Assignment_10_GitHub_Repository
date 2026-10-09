from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "fraud_model.joblib"
DATA_PATH = ROOT / "data" / "transactions.csv"
OUT_PATH = ROOT / "outputs" / "scored_transactions.csv"

def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: python src/train_model.py")
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    df = pd.read_csv(DATA_PATH)
    df["fraud_probability"] = model.predict_proba(df[features])[:, 1]
    df["predicted_label"] = (df["fraud_probability"] >= 0.5).astype(int)
    df["decision"] = df["predicted_label"].map({0: "LOW_RISK", 1: "REVIEW"})
    df[["transaction_id", "amount", "fraud_probability", "predicted_label", "decision"]].to_csv(
        OUT_PATH, index=False
    )
    print(df[["transaction_id", "fraud_probability", "decision"]].head(10).to_string(index=False))
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
