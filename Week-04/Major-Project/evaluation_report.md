# Waste Image Classification - Evaluation Report
Skill Nexis Deep Learning Internship | Major Project  
Report date: 2026-11-12

## Summary

- Task: classify waste images into 5 classes (cardboard, glass, metal, paper, plastic).
- Models compared: baseline CNN, improved CNN, ResNet18 transfer learning, and an ensemble of all three.
- Final model: **Ensemble (average of all models)**, chosen on validation accuracy (not on the test set).
- Final test accuracy: **89.97%** (95% CI 86.43% to 92.67%), macro F1 **0.895**. Baseline: 67.41%.
- Final model vs baseline (exact McNemar test): p = 0.0000, so the improvement is statistically significant.
- Easiest classes: paper, cardboard. Hardest classes: glass, metal.
- 36 of 359 test images were misclassified.
- Calibration: expected calibration error (ECE) 0.148 before and 0.192 after temperature scaling.

## 1. Dataset

- **Name:** TrashNet (5 of 6 classes; 'trash' excluded)
- **Source:** Yang and Thung, 2016. https://github.com/garythung/trashnet
- **Split:** stratified 70% / 15% / 15% (train / validation / test), seed 42.
- **Preprocessing (training, evaluation and uploads):** EXIF rotation fix, RGB conversion, resize to 128x128, ImageNet mean/std normalisation.

| Class | Train | Validation | Test |
|---|---|---|---|
| cardboard | 282 | 60 | 61 |
| glass | 350 | 76 | 75 |
| metal | 287 | 61 | 62 |
| paper | 416 | 89 | 89 |
| plastic | 337 | 73 | 72 |

## 2. Models

- **Baseline CNN:** 2 conv layers (16, 32 filters), max-pooling, 2 fully connected layers. No augmentation. Adam optimiser.
- **Improved CNN:** 4 double-conv blocks (32 to 256 filters) with BatchNorm, global average pooling, dropout 0.4, AdamW with weight decay, learning-rate reduction on plateau, and augmentation (random crop, flip, rotation, colour jitter).
- **Transfer learning:** ResNet18 pretrained on ImageNet with a new head. First 3 epochs train only the head, then the whole network is fine-tuned at a lower learning rate. Same augmentation.
- **Ensemble:** average of the calibrated probabilities of the three models.
- All trained for up to 20 epochs with early stopping (patience 5); the checkpoint with the best validation accuracy is kept.
- Note: the models differ in several ways at once (architecture, optimiser, augmentation, pretraining), so a gain cannot be credited to a single change.

## 3. Model comparison

| Metric | Baseline CNN | Improved CNN | Transfer Learning | Ensemble |
|---|---|---|---|---|
| Parameters | 2,102,629 | 1,174,501 | 11,179,077 | 14,456,207 |
| Epochs run (best) | 12 (7) | 20 (17) | 14 (9) | - |
| Training time | 215.5s | 2254.2s | 889.7s | 3359.4s |
| Validation accuracy | 63.79% | 74.37% | 89.69% | 90.53% |
| Test accuracy | 67.41% | 77.16% | 88.86% | 89.97% |
| Test 95% CI | 62.40% to 72.05% | 72.54% to 81.20% | 85.18% to 91.71% | 86.43% to 92.67% |
| Macro F1 | 0.668 | 0.756 | 0.887 | 0.895 |
| Misclassified | 117 | 82 | 40 | 36 |
| ECE before scaling | 0.174 | 0.069 | 0.053 | 0.148 |
| ECE after scaling | 0.046 | 0.054 | 0.044 | 0.192 |
| Temperature T | 2.55 | 0.95 | 1.18 | - |
| McNemar p vs baseline | - | 0.0024 | 0.0000 | 0.0000 |

Test accuracies come from one test set, so differences smaller than the confidence intervals should not be over-interpreted. The McNemar test checks whether a model's mistakes differ from the baseline's by more than chance.

![Training curves](training_curves.png)

## 4. Per-class results (Ensemble)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cardboard | 0.966 | 0.934 | 0.950 | 61 |
| glass | 0.900 | 0.840 | 0.869 | 75 |
| metal | 0.860 | 0.790 | 0.824 | 62 |
| paper | 0.927 | 1.000 | 0.962 | 89 |
| plastic | 0.844 | 0.903 | 0.872 | 72 |

## 5. Confusion matrices (test set)

![Baseline confusion matrix](confusion_baseline.png)

![Final model confusion matrix](confusion_ensemble.png)

## 6. Error analysis

Most frequent confusions of the final model (true class -> predicted class):

- metal -> plastic: 7 images
- glass -> plastic: 5 images
- glass -> metal: 4 images
- metal -> glass: 4 images
- cardboard -> paper: 3 images

Things to check when reviewing the mistakes below: visually similar materials (clear plastic vs glass, cardboard vs paper), crumpled or stained items, reflections on metal and glass, cluttered or low-contrast backgrounds, and objects partly out of frame. These are hypotheses from inspecting the images, not proven causes.

![Most confident mistakes](errors_ensemble.png)

### Grad-CAM on mistakes (Transfer Learning)

Grad-CAM highlights the image regions that pushed the model towards its (wrong) answer. If the red areas sit on the background or on a label instead of the object, the model has learned a shortcut. Heatmaps show where the model looked, not proof of why it was wrong.

![Grad-CAM on mistakes](gradcam_errors.png)

## 7. Robustness

Test accuracy of the Transfer Learning when test images are degraded:

| Condition | Accuracy | Change vs original |
|---|---|---|
| original | 88.86% | +0.00 points |
| darker (brightness x0.5) | 84.40% | -4.46 points |
| brighter (brightness x1.6) | 77.44% | -11.42 points |
| blurred (radius 2) | 86.91% | -1.95 points |
| rotated 30 degrees | 79.11% | -9.75 points |
| low resolution (32 px) | 70.19% | -18.66 points |
| noisy (sigma 25) | 87.74% | -1.11 points |

Large drops show conditions the model should not be trusted under.

## 8. Model scores do not guarantee correctness

The score shown by the app is the softmax output of the network (after temperature scaling). It is **not** a verified probability of being right.

- Mean top score on correct predictions: 0.73.
- Mean top score on wrong predictions: 0.49.
- Wrong predictions with a score of at least 0.60: 6 of 36; at least 0.80: 0 of 36.
- Calibration: ECE fell from 0.148 to 0.192 with temperature scaling (fitted on the validation set). Scores are better aligned with accuracy but individual predictions can still be confidently wrong.
- The model always picks one of the trained classes, so an image of something that is not waste (or a material outside these classes) still receives a label.
- Networks are often over-confident on images that differ from the training data (different lighting, camera, background).
- The app flags predictions as uncertain when the top score or the margin over the second choice is low. Treat every result as a hint for a human to check, not as a decision.

![Reliability diagram](reliability_ensemble.png)

## 9. Limitations

- The dataset is small and photographed in controlled conditions; real-world photos will perform worse.
- Only five categories are covered; other waste types are forced into one of them.
- One random split and one training run per model; run-to-run variance is unknown.
- The app uses test-time augmentation (original plus mirrored image), which the reported test accuracies do not.
- Results come from a single dataset and should not be generalised to other collection setups.

## 10. Reproducing

```
python -m pip install -r requirements.txt
python main.py        # train, evaluate, build this report
python app.py         # launch the Gradio demo
```
