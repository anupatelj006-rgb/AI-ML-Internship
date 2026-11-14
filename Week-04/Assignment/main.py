import os
import pandas as pd
from config import set_seed, OUT, CONFIGS, MEAN, STD, SEED
from data_loader import get_data, class_distribution
from models import build_model
from train import fit
from evaluate import (count_params, predict, plot_class_distribution, plot_curves,
                      per_class_metrics, plot_per_class, confusion_plot, top_confusions,
                      failure_gallery, confidence_stats, accuracy_ci)
from report import build_report


def main():
    set_seed()

    print("Step 1: Loading the data and counting classes...")
    train_loader, val_loader, test_loader, test_ds, targets = get_data()
    dist = class_distribution(targets)
    dist.to_csv(os.path.join(OUT, "01_class_distribution.csv"), index=False)
    plot_class_distribution(dist)
    splits = {k: len(v) for k, v in targets.items()}
    print(dist.to_string(index=False))
    print("Split sizes:", splits)

    print("\nStep 2: Training each configuration...")
    histories, rows, preds = {}, [], {}
    for cfg in CONFIGS:
        set_seed()
        model = build_model(cfg)
        hist = fit(model, train_loader, val_loader, cfg)
        pred, true, conf = predict(model, test_loader)
        histories[cfg["name"]] = hist
        preds[cfg["name"]] = (pred, true, conf)
        b = hist["best_epoch"] - 1
        rows.append({**cfg,
                     "params": count_params(model),
                     "best_epoch": hist["best_epoch"],
                     "train_acc": hist["train_acc"][b],
                     "val_acc": hist["best_val_acc"],
                     "test_acc": float((pred == true).mean()),
                     "seconds": hist["seconds"]})

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "02_config_comparison.csv"), index=False)
    print("\n", table[["name", "params", "best_epoch", "train_acc", "val_acc", "test_acc"]].to_string(index=False))

    print("\nStep 3: Final evaluation of the best configuration...")
    best = table.loc[table["val_acc"].idxmax(), "name"]
    print("Final model (best validation accuracy):", best)
    pred, true, conf = preds[best]

    plot_curves(histories)
    per_class = per_class_metrics(true, pred)
    plot_per_class(per_class)
    cm = confusion_plot(true, pred)
    pairs = top_confusions(cm)
    failures = failure_gallery(test_ds, pred, true, conf)
    stats = confidence_stats(pred, true, conf)

    acc = float((pred == true).mean())
    lo, hi = accuracy_ci(acc, len(true))
    print(f"Test accuracy: {acc*100:.2f}%  (95% CI {lo*100:.2f}% to {hi*100:.2f}%)")
    print(per_class.to_string(index=False))

    print("\nStep 4: Writing the report...")
    ctx = {
        "dataset": {"name": "Fashion-MNIST",
                    "source": "Zalando Research (Xiao, Rasul and Vollgraf, 2017), loaded with torchvision.datasets.FashionMNIST",
                    "link": "https://github.com/zalandoresearch/fashion-mnist",
                    "license": "MIT license",
                    "image": "28 x 28 pixels, 1 grey channel, 10 classes",
                    "total": sum(splits.values()), "splits": splits, "dist": dist,
                    "mean": MEAN, "std": STD, "seed": SEED},
        "configs": rows,
        "best": best,
        "final": {"test_acc": acc, "ci_low": lo, "ci_high": hi,
                  "macro_f1": float(per_class["F1"].mean()),
                  "n_test": len(true), "n_wrong": int((pred != true).sum())},
        "per_class": per_class,
        "pairs": pairs,
        "failures": failures,
        "conf": stats,
    }
    build_report(ctx, OUT)
    print("\nDone. Open evaluation_report.html in your browser.")


if __name__ == "__main__":
    main()