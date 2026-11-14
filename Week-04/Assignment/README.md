# Assignment 4: Final Model Evaluation

Skill Nexis Deep Learning Internship | Week 4

Trains two CNN configurations on Fashion-MNIST, evaluates the best one on the test set, and writes a full evaluation report.

## Report contents
1. Dataset source, class distribution and split sizes
2. Results from two model configurations
3. Training and validation graphs
4. Final test accuracy and per-class results
5. Failure examples and suggested improvements

## How to run
1. `python -m pip install -r requirements.txt`
2. `python main.py`
3. Open `evaluation_report.html` (print it to PDF to submit)

## Files
- `main.py`: runs everything in order
- `config.py`, `data_loader.py`, `models.py`, `train.py`, `evaluate.py`: the code, split by job
- `report.py`: writes `evaluation_report.md` and `evaluation_report.html`
- `outputs/`: charts and tables