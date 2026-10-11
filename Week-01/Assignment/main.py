"""Task 1 runner.   Usage:  python main.py        or        python main.py --show"""
import argparse
import json
import os

import config
import tensor_ops
import visualize
from dataset import make_data, make_loaders
from trainer import train


def main():
    parser = argparse.ArgumentParser(description="Task 1: neural network basics")
    parser.add_argument("--show", action="store_true", help="also open plot windows")
    args = parser.parse_args()

    # Part 2: tensors
    tensor_ops.run_all()

    # Part 3: data
    x, y = make_data()
    train_loader, val_loader = make_loaders(x, y)
    print("\n" + "=" * 50)
    print("PART 3: DATA AND TRAINING")
    print("=" * 50)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Updates per epoch: {len(train_loader)}")
    visualize.plot_dataset(x, y, args.show)

    # Part 4: train and plot loss
    main_lr = config.LEARNING_RATES[0]
    model, history = train(train_loader, val_loader, lr=main_lr)
    visualize.plot_loss_curves(history, main_lr, args.show)
    visualize.plot_prediction(model, x, y, args.show)

    # Part 5: compare learning rates
    print("\n" + "=" * 50)
    print("PART 5: LEARNING RATE COMPARISON")
    print("=" * 50)
    all_history = {}
    summary = {}
    for lr in config.LEARNING_RATES:
        _, h = train(train_loader, val_loader, lr=lr)
        all_history[lr] = h
        summary[str(lr)] = {
            "final_train_loss": h["train_loss"][-1],
            "final_val_loss": h["val_loss"][-1],
        }
        print(f"lr = {lr:<7} train = {h['train_loss'][-1]:.5f}   "
              f"val = {h['val_loss'][-1]:.5f}")
    visualize.plot_lr_comparison(all_history, args.show)

    # Bonus: learning rate that is too large
    _, bonus = train(train_loader, val_loader, lr=config.BONUS_LR)
    summary[str(config.BONUS_LR)] = {
        "final_train_loss": bonus["train_loss"][-1],
        "final_val_loss": bonus["val_loss"][-1],
    }
    print(f"lr = {config.BONUS_LR:<7} train = {bonus['train_loss'][-1]:.5f}   "
          f"val = {bonus['val_loss'][-1]:.5f}  (bonus, too large)")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved plots and results.json in '{config.OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()