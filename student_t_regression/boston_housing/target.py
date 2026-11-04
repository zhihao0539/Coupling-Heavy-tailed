"""Student-t regression data and target log density."""
from pathlib import Path
import numpy as np
import pandas as pd

NU = 4                              # Degrees of freedom of the regression errors.
FOLDER = Path(__file__).resolve().parent


def load_data():
    data = pd.read_csv(FOLDER / 'BostonHousing.csv')
    X = data.drop(columns='medv').to_numpy(dtype=float)
    y = data['medv'].to_numpy(dtype=float)
    # Same standardization as StandardScaler; no intercept is added.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    return X, y


def log_density(theta, X, y):
    """Unnormalized density in (beta, log(sigma)) coordinates."""
    beta, u = theta[:-1], theta[-1]
    residuals = y - X @ beta
    # Stable log(1 + residuals**2 / (NU * exp(2*u))).
    with np.errstate(divide='ignore'):
        terms = 2 * np.log(np.abs(residuals)) - np.log(NU) - 2 * u
    return -.5 * (NU + 1) * np.logaddexp(0, terms).sum() - len(y) * u
