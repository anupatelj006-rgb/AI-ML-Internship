# Assignment 3: Build and Evaluate a CNN (Fashion-MNIST)

Skill Nexis Deep Learning Internship | Week 3

## What this does
Loads Fashion-MNIST (70,000 grey 28x28 clothing images, 10 classes), builds a small CNN, compares it with a fully connected baseline, plots training and validation curves, and analyses misclassified examples.

## Results
| Model | Parameters | Test accuracy |
|---|---|---|
| Fully connected | ___ | ___ |
| CNN | ___ | ___ |

## Error analysis (five misclassified examples)
1. Test image #___: true ___, predicted ___. Reason: ___
2. ...

## How to run
1. Activate the virtual environment
2. `pip install -r requirements.txt`
3. `python main.py` (charts are saved to `outputs/`)

## Files
- `main.py`: runs everything in order
- `config.py`, `data_loader.py`, `models.py`, `train.py`, `evaluate.py`: the code, split by job
- `outputs/`: saved charts and tables