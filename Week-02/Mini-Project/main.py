from config import set_seed, OUT
from data_loader import load_data
from model import build_model, DESIGNS
from train import compare_designs, train_model, plot_curves
from evaluate import (get_predictions, report_test_accuracy, sample_predictions,
                      confusion_and_report, wrong_predictions)
from experiments import confidence_analysis, trust_threshold, neuron_weight_maps, stress_test


def main():
    set_seed()

    print("Step 1: Loading data...")
    train, val, test = load_data()
    x_test, y_test = test

    print("\nStep 2: Comparing network designs...")
    table = compare_designs(train, val)
    print(table)
    best = table.loc[0, "Design"]
    print("Best design:", best)

    print("\nStep 3: Training the final model...")
    model = build_model(**DESIGNS[best])
    history = train_model(model, train, val, epochs=10)
    plot_curves(history)

    print("\nStep 4: Testing...")
    report_test_accuracy(model, x_test, y_test)
    pred, conf = get_predictions(model, x_test)
    y_true = y_test.numpy()
    sample_predictions(x_test, y_true, pred, conf)
    cm = confusion_and_report(y_true, pred)
    wrong_predictions(x_test, y_true, pred, conf, cm)

    print("\nStep 5: Extra experiments...")
    confidence_analysis(pred, y_true, conf)
    trust_threshold(pred, y_true, conf)
    neuron_weight_maps(model)
    stress_test(model, x_test, y_test)

    print(f"\nDone. Charts and tables are saved in: {OUT}")


if __name__ == "__main__":
    main()