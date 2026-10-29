import os
import pandas as pd
from config import set_seed, OUT, EPOCHS
from data_loader import show_samples, get_loaders
from models import FullyConnected, SmallCNN
from train import fit
from evaluate import (count_params, predict, plot_curves, confusion_and_report,
                      show_misclassified, plot_filters_and_maps)


def main():
    set_seed()

    print("Step 1: Loading and visualizing images...")
    show_samples()
    train_loader, val_loader, test_loader, test_ds = get_loaders()

    print("\nStep 2 and 3: Training the baseline and the CNN...")
    models = {"Fully connected": FullyConnected(), "CNN": SmallCNN()}
    histories, rows, results = {}, [], {}

    for name, model in models.items():
        hist = fit(model, train_loader, val_loader, epochs=EPOCHS, name=name)
        histories[name] = hist
        pred, true, conf = predict(model, test_loader)
        results[name] = (pred, true, conf)
        rows.append({
            "Model": name,
            "Parameters": count_params(model),
            "Train acc": round(hist["train_acc"][-1], 4),
            "Val acc": round(hist["val_acc"][-1], 4),
            "Test acc": round(float((pred == true).mean()), 4),
            "Training time (s)": round(hist["seconds"]),
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "03_model_comparison.csv"), index=False)
    print("\nComparison table:\n", table.to_string(index=False))

    print("\nStep 4: Plotting curves...")
    plot_curves(histories)

    print("\nStep 5: Error analysis...")
    for name in models:
        pred, true, _ = results[name]
        confusion_and_report(true, pred, f"{name}: confusion matrix",
                             f"03_confusion_{name.replace(' ', '_').lower()}.png")

    pred, true, conf = results["CNN"]
    show_misclassified(test_ds, pred, true, conf, n=5)
    plot_filters_and_maps(models["CNN"], test_ds)

    print(f"\nDone. Everything is saved in: {OUT}")


if __name__ == "__main__":
    main()