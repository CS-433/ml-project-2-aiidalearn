import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.utils import Encoding, custom_mape, encode_all_structures

# Set Up
DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

MODELS_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "models/log_delta_E/"
)

encoding = Encoding.COLUMN_MASS

console = Console(record=True)

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
with console.status("") as status:
    status.update("[bold blue]Loading data...")
    df = pd.read_csv(
        os.path.join(DATA_DIR, "data.csv"), index_col=0, na_filter=False
    )

    status.update(f"[bold blue]Encoding structures ({encoding})...")
    df = encode_all_structures(df, encoding)

    status.update(f"[bold blue]Splitting data...")
    cols_raw = list(df.columns)
    cols_trash = [
        "structure",
        "converged",
        "accuracy",
        "n_iterations",
        "time",
        "fermi",
        "total_energy",
    ]
    cols_independent = ["delta_E"]
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
        X_raw, logy_raw, test_size=0.2, random_state=42
    )
console.log("Data loaded")

# Model Definitions
linear_log_augmented_model = Pipeline(
    [
        ("scaler_init", StandardScaler()),
        ("features", PolynomialFeatures(degree=2)),
        ("scaler_final", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)

rf_log_model = RandomForestRegressor(random_state=0)

gb_log_model = GradientBoostingRegressor(
    n_estimators=5000, learning_rate=0.05, random_state=0
)

xgb_log_model = xgb.XGBRegressor(
    n_estimators=5000, learning_rate=0.05, random_state=0
)

# detect if gpu is usable with xgboost by training on toy data
try:
    xgb_log_model.set_params(tree_method="gpu_hist")
    xgb_log_model.fit(np.array([[1, 2, 3]]), np.array([[1]]))
    console.print("[italic bright_black]Using GPU for XGBoost")
except:
    xgb_log_model.set_params(tree_method="hist")
    console.print("[italic bright_black]Using CPU for XGBoost")

models_log = {
    "Augmented Linear Regression - Log": linear_log_augmented_model,
    "Random Forest - Log": rf_log_model,
    # "Gradient Boosting - Log": gb_log_model,
    "XGBoost - Log": xgb_log_model,
}

# Model training
with console.status("") as status:
    for model_name, model in models_log.items():
        status.update(f"[bold blue]Training {model_name}...")
        model.fit(X_train, logy_train)
        console.log(f"[blue]Finished training {model_name}[/blue]")

# Model evaluation
with console.status("") as status:
    for model_name, model in models_log.items():
        status.update(f"[bold blue]Evaluating {model_name}...")

        table = Table(title=model_name)
        table.add_column("Loss name", justify="center", style="cyan")
        table.add_column("Train", justify="center", style="green")
        table.add_column("Test", justify="center", style="green")

        # first we evaluate the log-transformed prediction
        logy_pred_train = model.predict(X_train)
        logy_pred_test = model.predict(X_test)

        for loss_name, loss_fn in [
            ("MSE - log", mean_squared_error),
            ("MAE - log", mean_absolute_error),
            ("MAPE - log", mean_absolute_percentage_error),
            ("Custom MAPE - log", custom_mape),
        ]:
            train_loss = loss_fn(logy_train, logy_pred_train)
            test_loss = loss_fn(logy_test, logy_pred_test)
            table.add_row(loss_name, f"{train_loss:.4E}", f"{test_loss:.4E}")

        # then we transform the predictions back and evaluate
        y_pred_train = log_transform.inverse_transform(
            logy_pred_train.squeeze()
        )
        y_pred_test = log_transform.inverse_transform(logy_pred_test.squeeze())

        y_train = log_transform.inverse_transform(logy_train.squeeze())
        y_test = log_transform.inverse_transform(logy_test.squeeze())

        for loss_name, loss_fn in [
            ("MSE", mean_squared_error),
            ("MAE", mean_absolute_error),
            ("MAPE", mean_absolute_percentage_error),
            ("Custom MAPE", custom_mape),
        ]:
            train_loss = loss_fn(y_train, y_pred_train)
            test_loss = loss_fn(y_test, y_pred_test)
            table.add_row(loss_name, f"{train_loss:.4E}", f"{test_loss:.4E}")

        console.print(table)

if input("Save models? (y/[n]) ") == "y":
    save_models = {
        "Random Forest - log": (rf_log_model, "random_forest_model.pkl"),
        # "Gradient Boosting - log": (gb_log_model, "gb_model.pkl"),
        "XGBoost - log": (xgb_log_model, "xgb_model.pkl"),
    }

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    console.save_html(os.path.join(MODELS_DIR, "results.html"))
    with console.status("[bold green]Saving models...") as status:
        for model_name, (model, filename) in save_models.items():
            modelpath = MODELS_DIR + filename
            with open(modelpath, "wb") as file:
                pickle.dump(model, file)
            console.log(f"[green]Saved {model_name} to {modelpath}[/green]")
