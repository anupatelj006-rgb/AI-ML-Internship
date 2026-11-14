"""Builds evaluation_report.md and a self-contained evaluation_report.html."""
import base64
import re
from datetime import date

import config

ALL = config.MODEL_NAMES + [config.ENSEMBLE]


def pct(x):
    return f"{x * 100:.2f}%"


def build_markdown(results):
    meta, models = results["meta"], results["models"]
    names = [n for n in ALL if n in models]
    final = meta["final"]
    cam_name = meta["cam_model"]
    ft = models[final]["test"]
    bt = models["baseline"]["test"]
    per = ft["per_class"]
    ranked = sorted(config.CLASSES, key=lambda c: per[c]["f1-score"], reverse=True)
    easiest, hardest = ranked[:2], ranked[-2:]
    rob = meta.get("robustness", {})

    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(n) for n in names) + " |"

    L = []
    L.append("# Waste Image Classification - Evaluation Report")
    L.append("Skill Nexis Deep Learning Internship | Major Project  ")
    L.append(f"Report date: {date.today().isoformat()}\n")

    L.append("## Summary\n")
    L.append(f"- Task: classify waste images into {len(config.CLASSES)} classes ({', '.join(config.CLASSES)}).")
    L.append(f"- Models compared: baseline CNN, improved CNN, ResNet18 transfer learning, and an ensemble of all three.")
    L.append(f"- Final model: **{config.MODEL_TITLES[final]}**, chosen on validation accuracy (not on the test set).")
    L.append(f"- Final test accuracy: **{pct(ft['accuracy'])}** (95% CI {pct(ft['ci_low'])} to {pct(ft['ci_high'])}), "
             f"macro F1 **{ft['macro_f1']:.3f}**. Baseline: {pct(bt['accuracy'])}.")
    p = ft.get("mcnemar_vs_baseline")
    if final != "baseline" and p is not None:
        verdict = "statistically significant" if p < 0.05 else "not statistically significant at the 5% level"
        L.append(f"- Final model vs baseline (exact McNemar test): p = {p:.4f}, so the improvement is {verdict}.")
    L.append(f"- Easiest classes: {', '.join(easiest)}. Hardest classes: {', '.join(hardest)}.")
    L.append(f"- {ft['n_wrong']} of {ft['n_test']:,} test images were misclassified.")
    L.append(f"- Calibration: expected calibration error (ECE) {ft['ece_raw']:.3f} before and {ft['ece_cal']:.3f} after "
             f"temperature scaling.\n")

    L.append("## 1. Dataset\n")
    L.append(f"- **Name:** {config.DATASET_NAME}")
    L.append(f"- **Source:** {config.DATASET_SOURCE}")
    L.append(f"- **Split:** stratified {config.SPLIT[0]:.0%} / {config.SPLIT[1]:.0%} / {config.SPLIT[2]:.0%} "
             f"(train / validation / test), seed {config.SEED}.")
    L.append(f"- **Preprocessing (training, evaluation and uploads):** EXIF rotation fix, RGB conversion, "
             f"resize to {config.IMG_SIZE}x{config.IMG_SIZE}, ImageNet mean/std normalisation.\n")
    L.append("| Class | Train | Validation | Test |")
    L.append("|---|---|---|---|")
    for c in config.CLASSES:
        L.append(f"| {c} | {meta['counts']['train'][c]} | {meta['counts']['val'][c]} | {meta['counts']['test'][c]} |")
    L.append("")

    L.append("## 2. Models\n")
    L.append("- **Baseline CNN:** 2 conv layers (16, 32 filters), max-pooling, 2 fully connected layers. "
             "No augmentation. Adam optimiser.")
    L.append("- **Improved CNN:** 4 double-conv blocks (32 to 256 filters) with BatchNorm, global average pooling, "
             "dropout 0.4, AdamW with weight decay, learning-rate reduction on plateau, and augmentation "
             "(random crop, flip, rotation, colour jitter).")
    L.append(f"- **Transfer learning:** ResNet18 pretrained on ImageNet with a new head. First {config.WARMUP_EPOCHS} "
             "epochs train only the head, then the whole network is fine-tuned at a lower learning rate. Same augmentation.")
    L.append("- **Ensemble:** average of the calibrated probabilities of the three models.")
    L.append(f"- All trained for up to {config.EPOCHS} epochs with early stopping (patience {config.PATIENCE}); "
             "the checkpoint with the best validation accuracy is kept.")
    L.append("- Note: the models differ in several ways at once (architecture, optimiser, augmentation, pretraining), "
             "so a gain cannot be credited to a single change.")
    tinfo = models.get("transfer", {}).get("train", {})
    if tinfo.get("pretrained_loaded") is False:
        L.append("- **Warning:** the pretrained ImageNet weights could not be downloaded, so the transfer model "
                 "was trained from scratch. Re-run with internet access for the intended result.")
    L.append("")

    L.append("## 3. Model comparison\n")
    L.append("| Metric | " + " | ".join(config.MODEL_TITLES[n].split(" (")[0] for n in names) + " |")
    L.append("|---|" + "---|" * len(names))
    L.append(row("Parameters", lambda n: f"{models[n]['train']['params']:,}"))
    L.append(row("Epochs run (best)", lambda n: "-" if models[n]["train"].get("epochs_run") is None
                 else f"{models[n]['train']['epochs_run']} ({models[n]['train']['best_epoch']})"))
    L.append(row("Training time", lambda n: f"{models[n]['train']['train_seconds']}s"))
    L.append(row("Validation accuracy", lambda n: pct(models[n]["train"]["best_val_acc"])))
    L.append(row("Test accuracy", lambda n: pct(models[n]["test"]["accuracy"])))
    L.append(row("Test 95% CI", lambda n: f"{pct(models[n]['test']['ci_low'])} to {pct(models[n]['test']['ci_high'])}"))
    L.append(row("Macro F1", lambda n: f"{models[n]['test']['macro_f1']:.3f}"))
    L.append(row("Misclassified", lambda n: str(models[n]["test"]["n_wrong"])))
    L.append(row("ECE before scaling", lambda n: f"{models[n]['test']['ece_raw']:.3f}"))
    L.append(row("ECE after scaling", lambda n: f"{models[n]['test']['ece_cal']:.3f}"))
    L.append(row("Temperature T", lambda n: "-" if meta["temperatures"].get(n) is None else f"{meta['temperatures'][n]:.2f}"))
    L.append(row("McNemar p vs baseline", lambda n: "-" if models[n]["test"].get("mcnemar_vs_baseline") is None
                 else f"{models[n]['test']['mcnemar_vs_baseline']:.4f}"))
    L.append("")
    L.append("Test accuracies come from one test set, so differences smaller than the confidence intervals "
             "should not be over-interpreted. The McNemar test checks whether a model's mistakes differ from the "
             "baseline's by more than chance.")
    L.append("\n![Training curves](training_curves.png)\n")

    L.append(f"## 4. Per-class results ({config.MODEL_TITLES[final].split(' (')[0]})\n")
    L.append("| Class | Precision | Recall | F1 | Support |")
    L.append("|---|---|---|---|---|")
    for c in config.CLASSES:
        r = per[c]
        L.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    L.append("")

    L.append("## 5. Confusion matrices (test set)\n")
    L.append(f"![Baseline confusion matrix]({bt['confusion_plot']})\n")
    if final != "baseline":
        L.append(f"![Final model confusion matrix]({ft['confusion_plot']})\n")

    L.append("## 6. Error analysis\n")
    if ft["top_confusions"]:
        L.append("Most frequent confusions of the final model (true class -> predicted class):\n")
        for t, pr, n in ft["top_confusions"]:
            L.append(f"- {t} -> {pr}: {n} images")
        L.append("")
    L.append("Things to check when reviewing the mistakes below: visually similar materials "
             "(clear plastic vs glass, cardboard vs paper), crumpled or stained items, reflections on metal and glass, "
             "cluttered or low-contrast backgrounds, and objects partly out of frame. These are hypotheses from "
             "inspecting the images, not proven causes.\n")
    if ft["error_plot"]:
        L.append(f"![Most confident mistakes]({ft['error_plot']})\n")
    L.append(f"### Grad-CAM on mistakes ({config.MODEL_TITLES[cam_name].split(' (')[0]})\n")
    L.append("Grad-CAM highlights the image regions that pushed the model towards its (wrong) answer. "
             "If the red areas sit on the background or on a label instead of the object, the model has learned "
             "a shortcut. Heatmaps show where the model looked, not proof of why it was wrong.\n")
    L.append("![Grad-CAM on mistakes](gradcam_errors.png)\n")

    L.append("## 7. Robustness\n")
    if rob:
        L.append(f"Test accuracy of the {config.MODEL_TITLES[meta['robustness_model']].split(' (')[0]} "
                 "when test images are degraded:\n")
        L.append("| Condition | Accuracy | Change vs original |")
        L.append("|---|---|---|")
        base = rob["original"]
        for k, v in rob.items():
            L.append(f"| {k} | {pct(v)} | {(v - base) * 100:+.2f} points |")
        L.append("")
        L.append("Large drops show conditions the model should not be trusted under.\n")

    L.append("## 8. Model scores do not guarantee correctness\n")
    L.append("The score shown by the app is the softmax output of the network (after temperature scaling). "
             "It is **not** a verified probability of being right.\n")
    L.append(f"- Mean top score on correct predictions: {ft['mean_conf_correct']:.2f}.")
    L.append(f"- Mean top score on wrong predictions: {ft['mean_conf_wrong']:.2f}.")
    L.append(f"- Wrong predictions with a score of at least 0.60: {ft['wrong_conf_ge_60']} of {ft['n_wrong']}; "
             f"at least 0.80: {ft['wrong_conf_ge_80']} of {ft['n_wrong']}.")
    L.append(f"- Calibration: ECE fell from {ft['ece_raw']:.3f} to {ft['ece_cal']:.3f} with temperature scaling "
             "(fitted on the validation set). Scores are better aligned with accuracy but individual predictions "
             "can still be confidently wrong.")
    L.append("- The model always picks one of the trained classes, so an image of something that is not waste "
             "(or a material outside these classes) still receives a label.")
    L.append("- Networks are often over-confident on images that differ from the training data "
             "(different lighting, camera, background).")
    L.append("- The app flags predictions as uncertain when the top score or the margin over the second choice is low. "
             "Treat every result as a hint for a human to check, not as a decision.\n")
    L.append(f"![Reliability diagram]({ft['reliability_plot']})\n")

    L.append("## 9. Limitations\n")
    L.append("- The dataset is small and photographed in controlled conditions; real-world photos will perform worse.")
    L.append("- Only five categories are covered; other waste types are forced into one of them.")
    L.append("- One random split and one training run per model; run-to-run variance is unknown.")
    L.append("- The app uses test-time augmentation (original plus mirrored image), which the reported test "
             "accuracies do not.")
    L.append("- Results come from a single dataset and should not be generalised to other collection setups.\n")

    L.append("## 10. Reproducing\n")
    L.append("```")
    L.append("python -m pip install -r requirements.txt")
    L.append("python main.py        # train, evaluate, build this report")
    L.append("python app.py         # launch the Gradio demo")
    L.append("```")
    return "\n".join(L) + "\n"


def build_report(results):
    import markdown

    md = build_markdown(results)
    (config.OUTPUT_DIR / "evaluation_report.md").write_text(md, encoding="utf-8")
    (config.ROOT / "evaluation_report.md").write_text(md, encoding="utf-8")

    body = markdown.markdown(md, extensions=["tables", "fenced_code"])

    def embed(match):
        path = config.OUTPUT_DIR / match.group(1)
        if not path.exists():
            return match.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}"'

    body = re.sub(r'<img src="([^"]+)"', embed, body)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Waste Image Classification - Evaluation Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#222}}
table{{border-collapse:collapse;margin:1rem 0;font-size:0.92rem}} th,td{{border:1px solid #ccc;padding:5px 10px;text-align:left}}
th{{background:#f2f2f2}} img{{max-width:100%}} code,pre{{background:#f6f6f6;padding:2px 6px;border-radius:4px}}
pre{{padding:10px;overflow-x:auto}}
</style></head><body>{body}</body></html>"""
    out = config.ROOT / "evaluation_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved {out}")