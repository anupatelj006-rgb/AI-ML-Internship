"""Gradio app: classify, batch classify, view model results, read the limitations."""
import csv
import json
import os
from datetime import datetime

import gradio as gr
from PIL import Image

import config
from predict import Predictor

LOG = config.OUTPUT_DIR / "prediction_log.csv"
_predictor = None

DISCLAIMER = (
    "**Please note:** the score is the model's own confidence, not a guarantee. A high score can still be wrong, "
    f"and the model always chooses one of the {len(config.CLASSES)} categories ({', '.join(config.CLASSES)}), "
    "even for objects that are not waste or are not in these categories. Treat the result as a suggestion "
    "and check it yourself."
)


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def log_prediction(source, model_name, res):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "source", "model", "label", "score", "margin", "uncertain"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), source, model_name, res["label"],
                    f"{res['score']:.4f}", f"{res['margin']:.4f}", res["uncertain"]])


def classify(image, model_name, use_tta, show_cam):
    if image is None:
        return None, "Please upload an image.", None
    p = get_predictor()
    res = p.predict(image, model_name, use_tta)
    log_prediction("single", model_name, res)
    lines = [f"### Predicted category: **{res['label']}**",
             f"Model score: **{res['score']:.1%}** (margin over second choice: {res['margin']:.1%})"]
    if res["uncertain"]:
        lines.append("\n**Uncertain.** The score or the margin is low. Try a clearer photo with the object "
                     "centred on a plain background, and check the result manually.")
    else:
        lines.append("\nThe score is fairly high, but it can still be wrong. Check it yourself.")
    cam = p.explain(image, res["label"]) if show_cam else None
    return res["scores"], "\n".join(lines), cam


def classify_batch(files, model_name, use_tta):
    if not files:
        return [], None
    p = get_predictor()
    rows = []
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        base = os.path.basename(path)
        try:
            res = p.predict(Image.open(path), model_name, use_tta)
            log_prediction("batch", model_name, res)
            rows.append([base, res["label"], round(res["score"], 3), round(res["margin"], 3),
                         "yes" if res["uncertain"] else "no"])
        except Exception as e:
            rows.append([base, "error: " + str(e)[:60], "", "", ""])
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "batch_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "predicted", "score", "margin", "uncertain"])
        w.writerows(rows)
    return rows, str(out)


def load_results():
    path = config.OUTPUT_DIR / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def results_text(r):
    if r is None:
        return "No results yet. Run `python main.py` first."
    names = [n for n in config.MODEL_NAMES + [config.ENSEMBLE] if n in r["models"]]
    lines = ["| Model | Test accuracy | Macro F1 | ECE after scaling |", "|---|---|---|---|"]
    for n in names:
        t = r["models"][n]["test"]
        lines.append(f"| {config.MODEL_TITLES[n]} | {t['accuracy']:.2%} | {t['macro_f1']:.3f} | {t['ece_cal']:.3f} |")
    lines.append(f"\nFinal model (best validation accuracy): **{config.MODEL_TITLES[r['meta']['final']]}**")
    return "\n".join(lines)


def build_demo():
    p = get_predictor()
    choices = p.available()
    r = load_results()
    default = r["meta"]["final"] if r and r["meta"]["final"] in choices else choices[0]

    with gr.Blocks(title="Waste Image Classifier") as demo:
        gr.Markdown("# Waste Image Classifier\nUpload a photo of a single waste item.")
        with gr.Tabs():
            with gr.TabItem("Classify"):
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(type="pil", label="Upload image")
                        model_dd = gr.Dropdown(choices, value=default, label="Model")
                        tta = gr.Checkbox(value=True, label="Test-time augmentation (also check the mirrored image)")
                        cam_box = gr.Checkbox(value=True, label="Show Grad-CAM (where the model looked)")
                        button = gr.Button("Classify", variant="primary")
                    with gr.Column():
                        result = gr.Markdown()
                        label = gr.Label(num_top_classes=len(config.CLASSES), label="Score per category")
                        cam_out = gr.Image(label="Grad-CAM heatmap", type="pil")
                gr.Markdown(DISCLAIMER)
                button.click(classify, [image, model_dd, tta, cam_box], [label, result, cam_out])
                image.upload(classify, [image, model_dd, tta, cam_box], [label, result, cam_out])

            with gr.TabItem("Batch"):
                files = gr.File(file_count="multiple", label="Upload several images")
                model_b = gr.Dropdown(choices, value=default, label="Model")
                tta_b = gr.Checkbox(value=True, label="Test-time augmentation")
                run_b = gr.Button("Classify all", variant="primary")
                table = gr.Dataframe(headers=["file", "predicted", "score", "margin", "uncertain"],
                                     interactive=False)
                csv_out = gr.File(label="Download results (CSV)")
                run_b.click(classify_batch, [files, model_b, tta_b], [table, csv_out])

            with gr.TabItem("Model results"):
                gr.Markdown(results_text(r))
                if r:
                    final = r["meta"]["final"]
                    for fname in (f"confusion_{final}.png", f"reliability_{final}.png"):
                        fp = config.OUTPUT_DIR / fname
                        if fp.exists():
                            gr.Image(value=str(fp), label=fname, interactive=False)

            with gr.TabItem("About"):
                gr.Markdown(
                    "## How it works\n"
                    "- Uploaded images go through the same preprocessing as the training data.\n"
                    "- Scores are calibrated with temperature scaling fitted on a validation set.\n"
                    "- Grad-CAM shows which image regions influenced the decision.\n"
                    "- Predictions are marked **uncertain** when the top score or the margin is low.\n"
                    "- Every prediction is logged to `outputs/prediction_log.csv`.\n\n"
                    "## Limitations\n" + DISCLAIMER)
    return demo


if __name__ == "__main__":
    build_demo().launch(inbrowser=True)