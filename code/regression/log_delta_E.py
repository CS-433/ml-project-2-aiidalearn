import os
import numpy as np
import pandas as pd
import sys
from time import sleep

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import xgboost as xgb

import pickle

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

# Set Up

sys.path.append(os.path.dirname(os.getcwd()))
from tools.utils import PERIODIC_TABLE_INFO, PTC_COLNAMES, encode_all_structures, Encoding, custom_mape

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "data/"
)

encoding = Encoding.COLUMN_MASS

# Class to apply log-transformation to dataset

class LogTransform:
    def __init__(self, y):
        self.miny = float(np.min(y))
        miny2 = np.sort(list(set(list(np.array(y.squeeze())))))[1]
        self.eps = (miny2 - self.miny) / 10

    def transform(self, y):
        return np.log(y - self.miny + self.eps)

    def inverse_transform(self, logy):
        return np.exp(logy) + self.miny - self.eps

# Data Loading

df = pd.read_csv(os.path.join(DATA_DIR, "data.csv"), index_col=0, na_filter= False)
df = encode_all_structures(df, encoding)


cols_raw = list(df.columns)
cols_trash = ["structure", 'converged', 'accuracy', "n_iterations", "time", "fermi", "total_energy"]
cols_independent = ['delta_E']
cols_drop = cols_trash + cols_independent

cols_dependent = cols_raw.copy()
for element in cols_drop:
    cols_dependent.remove(element)


X_raw = df[cols_dependent][df["converged"]]
y_raw = np.abs(df[cols_independent][df["converged"]]).squeeze()

# Log transform and train-test-tplit

log_transform = LogTransform(y_raw)

logy_raw = log_transform.transform(y_raw)
X_train, X_test, logy_train, logy_test = train_test_split(
    X_raw, logy_raw,
    test_size=0.2,
    random_state=42
)

# Model Definitions
# functions such that f(x) != 0 and f(+inf) = 0
linear_log_augmented_model = Pipeline([
    ('scaler_init', StandardScaler()),
    ('features', PolynomialFeatures(degree=2)),
    ('scaler_final', StandardScaler()),
    ('regressor', LinearRegression()),
])

rf_log_model = RandomForestRegressor(random_state=0)

gb_log_model = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05, random_state=0)

xgb_log_model = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.05, random_state=0)

models_log = {
    "Augmented Linear Regression - Log": linear_log_augmented_model,
    "Random Forest - Log": rf_log_model,
    # "Gradient Boosting - Log": gb_log_model,
    "XGBoost - Log": xgb_log_model,
}

# Model training
console = Console()
with console.status("[bold blue]Training models...") as status:
    for model_name, model in models_log.items():
        model.fit(X_train, logy_train)
        console.log(f"[blue]Finished training {model_name}[/blue]")

# Model evaluation

with console.status("[bold blue] Evaluating models...") as status:
    for model_name, model in models_log.items():
        sleep(0.5)
        console.log(f"[blue]Evaluating {model_name}[/blue]")

        logy_pred_train = model.predict(X_train)
        y_pred_train = log_transform.inverse_transform(logy_pred_train.squeeze())
        logy_pred_test = model.predict(X_test)
        y_pred_test = log_transform.inverse_transform(logy_pred_test.squeeze())

        y_train = log_transform.inverse_transform(logy_train.squeeze())
        y_test = log_transform.inverse_transform(logy_test.squeeze())

        mse_test = mean_squared_error(logy_test, logy_pred_test)
        mse_train = mean_squared_error(logy_train, logy_pred_train)

        titlestr = f'MSE log\n'
        contentstr = f"train:{mse_train:.4E}\ttest:{mse_test:.4E}"
        p1 = Panel(titlestr+contentstr)

        mse_test = mean_squared_error(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)

        titlestr = f'MSE\n'
        contentstr = f"train:{mse_train:.4E}\ttest:{mse_test:.4E}"
        p2 = Panel(titlestr+contentstr)

        mape_test = mean_absolute_percentage_error(logy_test, logy_pred_test)
        mape_train = mean_absolute_percentage_error(logy_train, logy_pred_train)

        titlestr = f'MAPE\n'
        contentstr = f"train:{mape_train:.4E}\ttest:{mape_test:.4E}"
        p3 = Panel(titlestr+contentstr)

        renderables = [p1, p2, p3]
        console.print(Columns(renderables))

        # mape_test = custom_mape(y_test, y_pred_test)
        # mape_train = custom_mape(y_train, y_pred_train)
        # print(f"Custom MAPE:\ttrain:{mape_train:.4E}\ttest:{mape_test:.4E}")


MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "models/log_delta_E/"
)

save_models = {
    "Random Forest": rf_log_model,
    "Gradient Boosting": gb_log_model,
    "XGBoost": xgb_log_model,
}

models_filenames = [
    "log_random_forest_model.pkl",
    "log_gb_model.pkl",
    "log_xgb_model.pkl",
]

with console.status("[bold green] Saving models...") as status:
    for filename, (model_name, model) in zip(models_filenames, save_models.items()):
        sleep(1)
        modelpath = MODELS_DIR + filename
        with open(modelpath, "wb") as file:
            pickle.dump(model, file)
        console.log(f"[green]Saved {model_name} to {modelpath}[/green]")

