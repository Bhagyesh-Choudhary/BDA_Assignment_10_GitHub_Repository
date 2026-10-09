from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    if not DATA.exists():
        raise SystemExit("Dataset not found. Run: python src/generate_data.py")
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c.startswith("feature_")] + ["amount"]
    X, y = df[feature_cols], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=180, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, pred, digits=4, zero_division=0)
    metrics = (
        f"Test transactions: {len(y_test)}\n"
        f"Fraud cases in test: {int(y_test.sum())}\n"
        f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}\n"
        f"Recall: {recall_score(y_test, pred, zero_division=0):.4f}\n"
        f"F1-score: {f1_score(y_test, pred, zero_division=0):.4f}\n"
        f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}\n\n"
        f"Classification report:\n{report}\n"
    )
    (OUT / "evaluation.txt").write_text(metrics, encoding="utf-8")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred),
                                  display_labels=["Legitimate", "Fraud"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix — Synthetic Test Data")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close()
    joblib.dump({"model": model, "features": feature_cols}, OUT / "fraud_model.joblib")
    test_df = df.loc[X_test.index, ["transaction_id", "amount", "label"]].copy()
    test_df["fraud_probability"] = prob
    test_df["predicted_label"] = pred
    test_df["decision"] = test_df["predicted_label"].map({0: "LOW_RISK", 1: "REVIEW"})
    test_df.to_csv(OUT / "predictions.csv", index=False)
    print(metrics)
    print(f"Saved outputs in {OUT}")

if __name__ == "__main__":
    main()
