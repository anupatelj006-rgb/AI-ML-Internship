import os
import pandas as pd
from config import set_seed, OUT, EPOCHS, MODEL_PATH, GROUP_NAMES, CLASS_TO_GROUP
from data_loader import show_samples, get_loaders
from models import build_model
from train import fit
from model_io import save_model, load_model
from evaluate import (count_params, predict, category_accuracy, plot_curves,
                      per_class_accuracy, confusion, make_galleries)

SPECS = [("Fully connected", False),
         ("Small CNN", False),
         ("Deep CNN (BatchNorm + Dropout)", False),
         ("Deep CNN + augmentation", True)]


def main():
    print("Step 1: Looking at the data...")
    set_seed()
    show_samples()
    loaders = {False: get_loaders(augment=False), True: get_loaders(augment=True)}
    test_loader, test_ds = loaders[False][2], loaders[False][3]

    print("\nStep 2: Training and comparing 4 models...")
    rows, histories, trained, results = [], {}, {}, {}
    for name, aug in SPECS:
        set_seed()
        model = build_model(name)
        train_loader, val_loader = loaders[aug][0], loaders[aug][1]
        hist = fit(model, train_loader, val_loader, epochs=EPOCHS, name=name)
        pred, true, conf = predict(model, test_loader)
        best = hist["best_epoch"] - 1
        rows.append({
            "Model": name,
            "Parameters": count_params(model),
            "Best epoch": hist["best_epoch"],
            "Train acc": round(hist["train_acc"][best], 4),
            "Val acc": round(hist["best_val_acc"], 4),
            "Test acc": round(float((pred == true).mean()), 4),
            "Category acc": round(category_accuracy(true, pred), 4),
            "Time (s)": round(hist["seconds"]),
        })
        histories[name], trained[name], results[name] = hist, model, (pred, true, conf)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "03_comparison_table.csv"), index=False)
    print("\nComparison table:\n", table.to_string(index=False))

    print("\nStep 3: Picking the best model (by validation accuracy) and saving it...")
    best_name = table.loc[table["Val acc"].idxmax(), "Model"]
    save_model(trained[best_name], best_name, MODEL_PATH)
    print("Best model:", best_name, "-> saved to", MODEL_PATH)

    reloaded, _ = load_model(MODEL_PATH)
    pred2, true2, _ = predict(reloaded, test_loader)
    print(f"Reload check: test accuracy of the saved model = {(pred2 == true2).mean():.4f}")

    print("\nStep 4: Charts, galleries and error analysis...")
    pred, true, conf = results[best_name]
    plot_curves(histories)
    make_galleries(test_ds, pred, true, conf)
    pc = per_class_accuracy(true, pred, best_name)
    confusion(true, pred, best_name)
    print("\nWeakest classes:\n", pc.head(4).to_string(index=False))
    print("\nStrongest classes:\n", pc.tail(3).to_string(index=False))

    print(f"\nDone. Everything is saved in: {OUT}")


if __name__ == "__main__":
    main()