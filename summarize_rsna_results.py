import argparse
import csv
import math
import os
import re
import statistics


def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def last_float(text: str, pattern: str) -> float:
    matches = re.findall(pattern, text)
    if not matches:
        return float("nan")
    value = matches[-1]
    try:
        return float(value)
    except Exception:
        return float("nan")


def nanmean(values):
    valid = [v for v in values if not math.isnan(v)]
    return float("nan") if not valid else statistics.mean(valid)


def nanstd(values):
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) < 2:
        return 0.0 if valid else float("nan")
    return statistics.stdev(valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    cols = [
        "fold",
        "train_loss",
        "train_acc1",
        "val_bacc",
        "val_f1",
        "val_auc_roc",
        "test_bacc",
        "test_f1",
        "test_auc_roc",
    ]
    rows = []

    train_losses, train_acc1s = [], []
    val_baccs, val_f1s, val_aucs = [], [], []
    test_baccs, test_f1s, test_aucs = [], [], []

    for fold in range(args.num_folds):
        train_log = read_text(os.path.join(args.results_dir, f"fold{fold}_train.log"))
        test_log = read_text(os.path.join(args.results_dir, f"fold{fold}_test.log"))

        train_loss = last_float(train_log, r"train_loss:\s*([0-9eE+\-.]+|nan)")
        train_acc1 = last_float(train_log, r"train_acc1:\s*([0-9eE+\-.]+|nan)")
        val_auc = last_float(train_log, r"val_AUROC:\s*([0-9eE+\-.]+|nan)")
        val_bacc = last_float(train_log, r"val_BACC:\s*([0-9eE+\-.]+|nan)")
        val_f1 = last_float(train_log, r"val_F1:\s*([0-9eE+\-.]+|nan)")

        test_auc = last_float(test_log, r"### AUC:\s*([0-9eE+\-.]+|nan)")
        test_bacc = last_float(test_log, r"### Balanced Accuracy:\s*([0-9eE+\-.]+|nan)")
        test_f1 = last_float(test_log, r"### F1:\s*([0-9eE+\-.]+|nan)")

        train_losses.append(train_loss)
        train_acc1s.append(train_acc1)
        val_baccs.append(val_bacc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_baccs.append(test_bacc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)

        rows.append(
            {
                "fold": fold,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "val_bacc": val_bacc,
                "val_f1": val_f1,
                "val_auc_roc": val_auc,
                "test_bacc": test_bacc,
                "test_f1": test_f1,
                "test_auc_roc": test_auc,
            }
        )

    rows.append(
        {
            "fold": "mean",
            "train_loss": nanmean(train_losses),
            "train_acc1": nanmean(train_acc1s),
            "val_bacc": nanmean(val_baccs),
            "val_f1": nanmean(val_f1s),
            "val_auc_roc": nanmean(val_aucs),
            "test_bacc": nanmean(test_baccs),
            "test_f1": nanmean(test_f1s),
            "test_auc_roc": nanmean(test_aucs),
        }
    )
    rows.append(
        {
            "fold": "std",
            "train_loss": nanstd(train_losses),
            "train_acc1": nanstd(train_acc1s),
            "val_bacc": nanstd(val_baccs),
            "val_f1": nanstd(val_f1s),
            "val_auc_roc": nanstd(val_aucs),
            "test_bacc": nanstd(test_baccs),
            "test_f1": nanstd(test_f1s),
            "test_auc_roc": nanstd(test_aucs),
        }
    )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Summary CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
