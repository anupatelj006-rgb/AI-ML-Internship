"""Train all models, calibrate, evaluate, analyse errors, and build the report.

    python main.py                 full run
    python main.py --epochs 1      quick smoke test
    python main.py --skip-train    reuse saved checkpoints
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import config
from data_loader import class_counts, get_loaders, set_seed
from evaluate import (evaluate_probs, fit_temperature, get_logits, mcnemar_p, plot_histories,
                      robustness, save_temperature, sorted_wrong)
from gradcam import gradcam_gallery
from predict import load_model
from report import build_report
from train import train_model


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    meta = {"temperatures": {}}
    infos, histories, trained = {}, {}, {}
    val_cal, test_cal, test_raw = {}, {}, {}
    y_val = y_test = test_samples = None

    for name in config.MODEL_NAMES:
        loaders, splits = get_loaders(config.AUGMENT[name])
        if "counts" not in meta:
            meta["counts"] = {k: class_counts(v) for k, v in splits.items()}
            print("Split sizes:", {k: len(v) for k, v in splits.items()})

        if args.skip_train:
            model, ckpt = load_model(name, device)
            history, info = ckpt["history"], ckpt["info"]
        else:
            set_seed()
            model, history, info = train_model(name, loaders, device, args.epochs)
        model.to(device).eval()

        y_val, lv = get_logits(model, loaders["val"], device)
        temperature = fit_temperature(lv, y_val)
        save_temperature(name, temperature)
        y_test, lt = get_logits(model, loaders["test"], device)
        test_samples = splits["test"]

        val_cal[name] = F.softmax(lv / temperature, dim=1).numpy()
        test_cal[name] = F.softmax(lt / temperature, dim=1).numpy()
        test_raw[name] = F.softmax(lt, dim=1).numpy()
        meta["temperatures"][name] = temperature
        info["best_val_acc"] = float((val_cal[name].argmax(1) == y_val).mean())
        infos[name], histories[name], trained[name] = info, history, model
        print(f"[{name}] temperature {temperature:.2f}")

    # ensemble = average of calibrated probabilities
    ens = config.ENSEMBLE
    val_cal[ens] = np.mean([val_cal[n] for n in config.MODEL_NAMES], axis=0)
    test_cal[ens] = np.mean([test_cal[n] for n in config.MODEL_NAMES], axis=0)
    test_raw[ens] = np.mean([test_raw[n] for n in config.MODEL_NAMES], axis=0)
    meta["temperatures"][ens] = None
    infos[ens] = {
        "best_val_acc": float((val_cal[ens].argmax(1) == y_val).mean()),
        "best_epoch": None, "epochs_run": None,
        "train_seconds": round(sum(infos[n]["train_seconds"] for n in config.MODEL_NAMES), 1),
        "params": sum(infos[n]["params"] for n in config.MODEL_NAMES),
    }

    all_names = config.MODEL_NAMES + [ens]
    results = {"meta": meta, "models": {}}
    for name in all_names:
        test = evaluate_probs(name, test_cal[name], test_raw[name], y_test, test_samples)
        if name != "baseline":
            test["mcnemar_vs_baseline"] = mcnemar_p(
                y_test, test_cal[name].argmax(1), test_cal["baseline"].argmax(1))
        print(f"[{name}] test accuracy {test['accuracy']:.4f}, macro F1 {test['macro_f1']:.4f}, "
              f"ECE {test['ece_raw']:.3f} -> {test['ece_cal']:.3f}")
        results["models"][name] = {"train": infos[name], "test": test}

    meta["final"] = max(all_names, key=lambda n: infos[n]["best_val_acc"])
    meta["cam_model"] = max(config.MODEL_NAMES, key=lambda n: infos[n]["best_val_acc"])
    meta["robustness_model"] = meta["cam_model"]
    print(f"Final model (best validation accuracy): {meta['final']}")

    cam = meta["cam_model"]
    meta["robustness"] = robustness(trained[cam], test_samples, device)
    wrong = sorted_wrong(test_cal[cam], y_test)
    gradcam_gallery(trained[cam], cam, test_samples, y_test, test_cal[cam].argmax(1), test_cal[cam],
                    wrong, config.OUTPUT_DIR / "gradcam_errors.png")

    plot_histories(histories, config.OUTPUT_DIR / "training_curves.png")
    (config.OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    build_report(results)
    print("\nDone. Open evaluation_report.html, then run: python app.py")


if __name__ == "__main__":
    main()