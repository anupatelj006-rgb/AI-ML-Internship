"""Central settings for Task 1. Change values here, not in the other files."""

# Data
N_SAMPLES = 1000
TRAIN_SIZE = 800
NOISE_STD = 0.1
TRUE_SLOPE = 3.0
TRUE_INTERCEPT = 2.0
DATA_SEED = 42

# Model
HIDDEN_UNITS = 16
MODEL_SEED = 0

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATES = [0.1, 0.001]   # the two rates to compare
BONUS_LR = 1.0                  # deliberately too large

# Output
OUTPUT_DIR = "outputs"