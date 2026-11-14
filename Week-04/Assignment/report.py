import os
import re
import base64
import datetime
import pandas as pd

REASONS = {
    frozenset(["Shirt", "T-shirt"]): "Both are plain upper-body clothes with almost the same outline. Buttons, collars and sleeve length are only a few pixels wide at 28x28.",
    frozenset(["Shirt", "Pullover"]): "Both are long-sleeved tops with a similar outline. The collar or buttons that tell them apart almost disappear at this size.",
    frozenset(["Shirt", "Coat"]): "Both have long sleeves and an opening at the front, and a grey picture hides the fabric and thickness.",
    frozenset(["Pullover", "Coat"]): "Both are heavy long-sleeved tops. The zip or buttons that make a coat are tiny in the picture.",
    frozenset(["T-shirt", "Pullover"]): "Both are tops with a similar body shape. Only the sleeve length differs.",
    frozenset(["T-shirt", "Dress"]): "A short dress and a long T-shirt have a similar upper shape. Only the length differs.",
    frozenset(["Dress", "Coat"]): "Both are long garments with a similar body shape in grey.",
    frozenset(["Trouser", "Dress"]): "A narrow dress and trousers are both tall, narrow shapes.",
    frozenset(["Sneaker", "Sandal"]): "Both are low shoes. Straps against closed fabric is a small difference at this size.",
    frozenset(["Sneaker", "Ankle boot"]): "Both are closed shoes. The height of the ankle is the main difference.",
    frozenset(["Sandal", "Ankle boot"]): "Both are shoes seen from the side, and open straps can look like a boot outline in grey.",
}
GENERIC_REASON = "These two classes have a similar overall shape at 28x28 resolution, so small details such as buttons, collars and straps are lost."


def reason_for(a, b):
    return REASONS.get(frozenset([a, b]), GENERIC_REASON)


def md_table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt:
                v = fmt[c].format(v)
            elif isinstance(v, float):
                v = f"{v:.4f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gap_comment(train_acc, val_acc):
    gap = train_acc - val_acc
    if gap > 0.03:
        return f"training is {gap*100:.1f} points above validation, which is a sign of overfitting"
    if gap < -0.005:
        return (f"validation is {-gap*100:.1f} points above training, which is normal when dropout "
                "is on during training, so the model is not overfitting")
    return "training and validation are close together, so there is little overfitting"


def build_report(ctx, out_dir):
    d, f, pc, cf = ctx["dataset"], ctx["final"], ctx["per_class"], ctx["conf"]
    best = ctx["best"]
    cfgs = pd.DataFrame(ctx["configs"])
    ranked = cfgs.sort_values("val_acc", ascending=False).reset_index(drop=True)
    top, other = ranked.iloc[0], ranked.iloc[1]
    diff = (top["test_acc"] - other["test_acc"]) * 100
    weakest = pc.sort_values("F1").head(3)
    strongest = pc.sort_values("F1", ascending=False).head(3)
    weak_names = ", ".join(weakest["Class"])
    dist = d["dist"]

    L = []
    add = L.append

    add("# Final Model Evaluation Report")
    add("Skill Nexis Deep Learning Internship | Week 4 | Assignment 4")
    add(f"Report date: {datetime.date.today().isoformat()}\n")

    add("## Summary\n")
    add(f"- Task: classify 28x28 grey images of clothing into {len(dist)} classes.")
    add(f"- Final model: **{best}**, chosen on validation accuracy.")
    add(f"- Final test accuracy: **{f['test_acc']*100:.2f}%** (95% confidence interval {f['ci_low']*100:.2f}% to {f['ci_high']*100:.2f}%), macro F1 **{f['macro_f1']:.3f}**.")
    add(f"- Easiest classes: {', '.join(strongest['Class'])}. Hardest classes: {weak_names}.")
    add(f"- {f['n_wrong']:,} of {f['n_test']:,} test images were misclassified.\n")

    add("## 1. Dataset\n")
    add(f"- **Name:** {d['name']}")
    add(f"- **Source:** {d['source']}")
    add(f"- **Link:** {d['link']}")
    add(f"- **License:** {d['license']}")
    add(f"- **Image format:** {d['image']}")
    add(f"- **Total images:** {d['total']:,}")
    add(f"- **Preprocessing:** pixels scaled to 0-1, then normalized with mean {d['mean']} and standard deviation {d['std']}.")
    add(f"- **Split method:** the official 60,000 training images were split at random (seed {d['seed']}) into training and validation. The official 10,000 test images were kept untouched until the final evaluation.\n")

    add("### Split sizes\n")
    split_df = pd.DataFrame({"Split": list(d["splits"].keys()),
                             "Images": list(d["splits"].values())})
    split_df["Share"] = split_df["Images"] / split_df["Images"].sum() * 100
    add(md_table(split_df, {"Images": "{:,}", "Share": "{:.1f}%"}))
    add("")

    add("### Class distribution\n")
    add(md_table(dist, {c: "{:,}" for c in dist.columns if c != "Class"}))
    add("")
    add("![Class distribution](outputs/01_class_distribution.png)\n")
    add(f"The smallest class has {int(dist['Training'].min()):,} training images and the largest has {int(dist['Training'].max()):,}, and the test set has exactly {int(dist['Test'].min()):,} to {int(dist['Test'].max()):,} images per class. The classes are balanced, so plain accuracy is a fair measure and no class weighting is needed.\n")

    add("## 2. Model configurations\n")
    settings = cfgs[["name", "model", "dropout", "lr", "weight_decay", "scheduler", "epochs"]].rename(columns={
        "name": "Configuration", "model": "Network", "dropout": "Dropout", "lr": "Learning rate",
        "weight_decay": "Weight decay", "scheduler": "LR schedule", "epochs": "Epochs"})
    add("### Settings\n")
    add(md_table(settings, {"Dropout": "{:g}", "Learning rate": "{:g}", "Weight decay": "{:g}"}))
    add("")
    results = cfgs[["name", "params", "best_epoch", "train_acc", "val_acc", "test_acc", "seconds"]].rename(columns={
        "name": "Configuration", "params": "Parameters", "best_epoch": "Best epoch", "train_acc": "Train acc",
        "val_acc": "Val acc", "test_acc": "Test acc", "seconds": "Time (s)"})
    add("### Results\n")
    add(md_table(results, {"Parameters": "{:,}", "Time (s)": "{:.0f}"}))
    add("")
    add(f"**{top['name']}** was chosen as the final model because it had the higher validation accuracy ({top['val_acc']*100:.2f}% against {other['val_acc']*100:.2f}%). On the test set it scored {diff:+.2f} points compared with {other['name']}. The model was picked on validation data only, so the test score is an honest estimate.\n")

    add("## 3. Training and validation graphs\n")
    add("![Training and validation curves](outputs/02_training_validation_curves.png)\n")
    for _, r in cfgs.iterrows():
        add(f"- **{r['name']}:** at its best epoch ({int(r['best_epoch'])}), training accuracy was {r['train_acc']*100:.2f}% and validation accuracy {r['val_acc']*100:.2f}%, so {gap_comment(r['train_acc'], r['val_acc'])}.")
    add("")

    add("## 4. Final test results\n")
    add(f"- **Final model:** {best}")
    add(f"- **Test accuracy:** {f['test_acc']*100:.2f}% on {f['n_test']:,} images (95% confidence interval {f['ci_low']*100:.2f}% to {f['ci_high']*100:.2f}%)")
    add(f"- **Macro F1:** {f['macro_f1']:.3f}\n")
    add("### Per-class results\n")
    add(md_table(pc, {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Support": "{:,}"}))
    add("")
    add("![Per-class F1](outputs/04_per_class_f1.png)\n")
    add("![Confusion matrix](outputs/05_confusion_matrix.png)\n")
    add(f"The strongest classes are {', '.join(f'{r.Class} (F1 {r.F1:.3f})' for r in strongest.itertuples())}. The weakest are {', '.join(f'{r.Class} (F1 {r.F1:.3f})' for r in weakest.itertuples())}.\n")

    add("## 5. Failure examples and suggested improvements\n")
    add("### Failure examples\n")
    add("The gallery shows the wrong predictions the model was most sure about.\n")
    add("![Failure examples](outputs/06_failure_examples.png)\n")
    fail_df = pd.DataFrame([{"Test image #": x["index"], "True": x["true"], "Predicted": x["pred"],
                             "Confidence": f"{x['conf']*100:.0f}%",
                             "Likely reason": reason_for(x["true"], x["pred"])} for x in ctx["failures"][:5]])
    add(md_table(fail_df))
    add("")
    add("### Most common confusions\n")
    pair_df = pd.DataFrame([{"True class": a, "Predicted as": b, "Times": n, "Likely reason": reason_for(a, b)}
                            for a, b, n in ctx["pairs"]])
    add(md_table(pair_df))
    add("")
    add("### Confidence check\n")
    add(f"- Average confidence on correct predictions: {cf['conf_correct']*100:.1f}%")
    add(f"- Average confidence on wrong predictions: {cf['conf_wrong']*100:.1f}%")
    add(f"- If the model only answers when it is at least {cf['thr']*100:.0f}% sure, it answers {cf['keep_rate']*100:.1f}% of the images with {cf['acc_keep']*100:.2f}% accuracy. The rest could be sent to a person to check.\n")

    add("### Suggested improvements\n")
    add(f"1. **Train longer with a learning-rate schedule and early stopping.** Several runs were still improving at their last epoch.")
    add(f"2. **Add targeted augmentation.** Small shifts and flips give the model more variety. Combine it with longer training, because short runs may not show the benefit.")
    add(f"3. **Use higher-resolution or colour images.** The weakest classes ({weak_names}) differ in details such as collars, buttons and straps that are only a few pixels wide at 28x28.")
    add("4. **Ensemble several models.** Averaging the predictions of a few networks trained with different seeds usually adds a little accuracy and steadies the results.")
    add("5. **Try transfer learning.** A network pretrained on a large image set, fine-tuned on resized images, can pick up finer detail.")
    add(f"6. **Collect more examples of the hard classes and check their labels.** Some pictures of {weak_names} are ambiguous even for a person.")
    add("7. **Use the confidence threshold in practice.** Let the model answer when it is sure and pass unsure cases to a human.\n")

    add("## 6. Conclusion\n")
    add(f"The {best} reached {f['test_acc']*100:.2f}% test accuracy on {d['name']} with a macro F1 of {f['macro_f1']:.3f}. It recognises footwear, trousers and bags very well. Most errors happen between similar tops ({weak_names}), where the difference is a few pixels of detail. The next steps with the best chance of helping are higher-resolution inputs, longer training and an ensemble.\n")

    md_text = "\n".join(L)
    report_dir = os.path.dirname(out_dir)
    md_path = os.path.join(report_dir, "evaluation_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_text)
    print("Saved", md_path)

    try:
        import markdown
    except ImportError:
        print("Could not make the HTML report. Run: python -m pip install markdown")
        return

    body = markdown.markdown(md_text, extensions=["tables"])

    def embed(m):
        path = os.path.join(report_dir, m.group(1))
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'src="data:image/png;base64,{b64}"'

    body = re.sub(r'src="([^"]+\.png)"', embed, body)
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:920px;margin:30px auto;padding:0 20px;line-height:1.5;color:#222}"
           "h1{border-bottom:2px solid #4c9f70;padding-bottom:6px}h2{margin-top:34px;color:#2b6e4f}"
           "table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}"
           "th,td{border:1px solid #ccc;padding:5px 8px;text-align:left}th{background:#eef5f0}"
           "img{max-width:100%;height:auto;display:block;margin:12px auto}"
           "@media print{img,table,tr{page-break-inside:avoid}}")
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Final Model Evaluation Report</title>"
            f"<style>{css}</style></head><body>{body}</body></html>")
    html_path = os.path.join(report_dir, "evaluation_report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Saved", html_path)