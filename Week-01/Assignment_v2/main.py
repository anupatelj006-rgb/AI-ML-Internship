import torch
from tensor_ops import tensor_demo
from train import make_data, plot_data, train_model, plot_loss, compare, save_results, TRUE_W, TRUE_B


def main():
    torch.manual_seed(42)

    print("Task 2: Tensor operations")
    tensor_demo()

    print("\nTask 3: Training on synthetic regression data")
    train, val = make_data()
    plot_data(train)
    print(f"True rule: y = {TRUE_W[0]:.1f}*x1 + ({TRUE_W[1]:.1f})*x2 + {TRUE_B:.1f}")

    results, histories = {}, {}
    for lr in [0.001, 0.05]:                       # Task 5: two learning rates
        model, hist = train_model(train, val, lr)
        plot_loss(hist, lr)                        # Task 4: training and validation loss
        histories[lr] = hist
        w = model.weight.detach().flatten().tolist()
        b = model.bias.item()
        results[f"lr_{lr}"] = {"final_train_loss": hist["train"][-1],
                               "final_val_loss": hist["val"][-1],
                               "learned_weights": w, "learned_bias": b}
        print(f"lr = {lr}: val loss {hist['val'][-1]:.4f} | weights {w[0]:.2f}, {w[1]:.2f} | bias {b:.2f}")

    compare(histories)
    save_results(results)
    print("\nDone. Charts and results.json are in the outputs folder.")


if __name__ == "__main__":
    main()