import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from rich.console import Console
from rich.table import Table
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.data_loader import data_loader
from tools.utils import Encoding, Target, custom_mape

# Set Up
DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

MODELS_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "models/sim_time/"
)

encoding = Encoding.COLUMN
target = Target.SIM_TIME

console = Console(record=True)

# Loading Data
X_train, X_test, y_train, y_test = data_loader(
    target=target,
    encoding=encoding,
    data_path=DATA_PATH,
    test_size=0.2,
    generalization=True,
)

# Model definition
with console.status("") as status:
    status.update("[bold blue]Initializing models...")

    linear_augmented_model = Pipeline(
        [
            ("scaler_init", StandardScaler()),
            ("poly_features", PolynomialFeatures(degree=2)),
            ("scaler_final", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    # best parameters from randomized grid search in corr. notebook
    rf_params = {"n_estimators": 230, "max_features": "auto", "max_depth": 40}
    rf_model = RandomForestRegressor(random_state=0, **rf_params)

    # best parameters from randomized grid search in corr. notebook
    xgb_params = {
        "n_estimators": 500,
        "min_split_loss": 0.2,
        "max_depth": 12,
        "learning_rate": 0.1,
        "lambda": 1,
        "booster": "gbtree",
        "alpha": 0.5,
    }

    # xgb_model = xgb.XGBRegressor(
    #     max_depth=7, n_estimators=400, learning_rate=1.0, random_state=0
    # )

    xgb_model = xgb.XGBRegressor(random_state=0, **xgb_params)

    lgbm_model = lgb.LGBMRegressor(
        n_estimators=5000, learning_rate=0.05, random_state=0
    )

    status.update("[bold blue]Checking GPU usability for XGBoost...")
    try:
        xgb_model.set_params(tree_method="gpu_hist")
        xgb_model.fit(np.array([[1, 2, 3]]), np.array([[1]]))
        console.print("[italic bright_black]Using GPU for XGBoost")
    except:
        xgb_model.set_params(tree_method="hist")
        console.print("[italic bright_black]Using CPU for XGBoost")

models = {
    "Dummy": DummyRegressor(),
    "Linear": LinearRegression(),
    "Augmented Linear": linear_augmented_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    # "LightGBM": lgbm_model,
}

# Model training
with console.status("") as status:
    for model_name, model in models.items():
        status.update(f"[bold blue]Training {model_name}...")
        model.fit(X_train, y_train)
        console.log(f"[blue]Finished training {model_name}[/blue]")


# Model evaluation
with console.status("") as status:
    for model_name, model in models.items():
        status.update(f"[bold blue]Evaluating {model_name}...")

        table = Table(title=model_name)
        table.add_column("Loss name", justify="center", style="cyan")
        table.add_column("Train", justify="center", style="green")
        table.add_column("Test", justify="center", style="green")

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        for loss_name, loss_fn in [
            ("MSE", mean_squared_error),
            ("MAE", mean_absolute_error),
            ("MAPE", mean_absolute_percentage_error),
            ("Custom MAPE", lambda a, b: custom_mape(a, b, True)),
        ]:
            train_loss = loss_fn(y_train, y_pred_train)
            test_loss = loss_fn(y_test, y_pred_test)
            table.add_row(loss_name, f"{train_loss:.4E}", f"{test_loss:.4E}")

        console.print(table)

# print some samples
n_sample = 10

table = Table(title="Test samples")
table.add_column("Real", justify="center", style="green")
for model_name, _ in models.items():
    table.add_column(model_name, justify="center", style="yellow")

idx_sample = np.random.choice(X_test.index, size=n_sample, replace=False)
results = [np.array(y_test[y_test.index.intersection(idx_sample)].squeeze())]
for model_name, model in models.items():
    results.append(
        np.array(
            model.predict(
                X_test.loc[X_test.index.intersection(idx_sample)]
            ).squeeze()
        )
    )

for i in range(n_sample):
    table.add_row(*[f"{r[i]:.3E}" for r in results],)
console.print(table)


if input("Save models? (y/[n]) ") == "y":
    save_models = {
        "Random Forest": (rf_model, "random_forest_model.pkl"),
        "XGBoost": (xgb_model, "xgb_model.pkl"),
        "LightGBM": (lgbm_model, "lgbm_model.pkl"),
    }

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    results_file = os.path.join(MODELS_DIR, "results.html")
    console.save_html(results_file)
    console.print(f"Results stored in {results_file}")
    with console.status("[bold green]Saving models...") as status:
        for model_name, (model, filename) in save_models.items():
            modelpath = MODELS_DIR + filename
            with open(modelpath, "wb") as file:
                pickle.dump(model, file)
            console.log(f"[green]Saved {model_name} to {modelpath}[/green]")
