"""Fit the location, radius and Mobius parameter for DCS with beta = 2."""
from pathlib import Path
from vi_functions import fit, save_fit

SEED = 26092901
STEPS = 4000
BATCH_SIZE = 2048
HOLDOUT_SIZE = 20000
AVERAGE_LAST = 500
LEARNING_RATES = (0.01, 0.002)
CHANGE_AFTER = 3000


if __name__ == "__main__":
    parameters, info = fit("dcs", seed=SEED, steps=STEPS, batch_size=BATCH_SIZE,
                           holdout_size=HOLDOUT_SIZE, average_last=AVERAGE_LAST,
                           learning_rates=LEARNING_RATES, change_after=CHANGE_AFTER)
    save_fit(Path(__file__).with_name("dcs_fit.json"), parameters, info)
