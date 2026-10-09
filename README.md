# Real-Time Fraud Detection in Digital Payments Using Kafka and Spark

Educational prototype for BDA Assignment No. 10.

## What this repository demonstrates
- Reproducible synthetic transaction data generation.
- A Random Forest baseline classifier for fraud-risk prediction.
- Evaluation using precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- A proposed scalable architecture using Apache Kafka for transaction events and Apache Spark Structured Streaming for continuous processing.

**Important:** The runnable code is a local batch prototype using synthetic data. It does not claim to implement a production Kafka/Spark cluster or use real bank data. The paper describes Kafka/Spark as the scalable target architecture.

## Requirements
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
python src/score_transactions.py
```

Outputs are saved under `outputs/`:
- `evaluation.txt`
- `predictions.csv`
- `confusion_matrix.png`
- `fraud_model.joblib`

## Dataset
The default dataset is generated locally and contains synthetic transactions, so no personal or financial information is used. For a real public benchmark, see the Kaggle Credit Card Fraud Detection dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The Kaggle dataset contains 284,807 transactions and 492 fraud cases from European cardholders over two days. It is strongly imbalanced, so accuracy alone is not a suitable metric.

## Proposed streaming architecture
Payment application -> Kafka topic `transactions` -> Spark Structured Streaming validation/features -> fraud scoring service -> Kafka topic `fraud-alerts` -> review dashboard / case-management system.

For official references:
- Kafka documentation: https://kafka.apache.org/documentation/
- Spark Structured Streaming: https://spark.apache.org/docs/latest/streaming/getting-started.html

## Limitations
Synthetic data is only for demonstrating the workflow. Performance on synthetic data does not establish real-world fraud-detection effectiveness. A production system needs validated data, leakage-safe temporal testing, privacy controls, monitoring, threshold tuning, and human review.

## Authors
- Bhagyesh Choudhary — implementation and documentation
- [Add second author's full name] — [add actual contribution]

Update the author and contribution details before submission.
