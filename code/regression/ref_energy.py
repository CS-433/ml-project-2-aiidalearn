import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.utils import (
    StructureEncoding,
    check_xgboost_gpu,
    encode_all_structures,
    custom_mape,
    percentile_absolute_percentage_error,
)

# Set Up
DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

DATA_PATH = os.path.join(DATA_DIR, "ref_energy.csv")

MODELS_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "models/ref_energy/"
)

BASELINES_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()),
    "baselines/ref_energy/",
)

encoding = StructureEncoding.ATOMIC

console = Console(record=True)

# Data Loading
with console.status("[bold blue]Loading data...") as status:
    df = pd.read_csv(DATA_PATH, na_filter=False)
    df = encode_all_structures(df, encoding)
    X = df.drop(columns=["structure", "total_energy"])
    y = df["total_energy"].squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    test_sets = [("Structure gen.", X_test, y_test)]

# Model Definitions
with console.status("") as status:
    status.update("[bold blue]Initializing models...")

    rf_model = RandomForestRegressor(random_state=0)

    xgb_model = xgb.XGBRegressor(
        max_depth=7, n_estimators=400, learning_rate=1.0, random_state=0
    )

    status.update("[bold blue]Checking GPU usability for XGBoost...")
    if check_xgboost_gpu():
        xgb_model.set_params(tree_method="gpu_hist")
        console.print("[italic bright_black]Using GPU for XGBoost")
    else:
        console.print("[italic bright_black]Using CPU for XGBoost")

    models = {
        "Dummy": DummyRegressor(),
        "Linear": LinearRegression(),
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
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
        for name, _, _ in test_sets:
            table.add_column(f"Test - {name}", justify="center", style="green")

        y_pred_train = model.predict(X_train)
        y_pred_tests = [model.predict(X_test) for _, X_test, _ in test_sets]

        for loss_name, loss_fn in [
            ("MSE", mean_squared_error),
            ("MAE", mean_absolute_error),
            ("MAPE", mean_absolute_percentage_error),
            ("Custom MAPE", lambda a, b: custom_mape(a, b, True)),
            (
                "50%-APE",
                lambda a, b: percentile_absolute_percentage_error(a, b, 50),
            ),
            (
                "90%-APE",
                lambda a, b: percentile_absolute_percentage_error(a, b, 90),
            ),
        ]:
            train_loss = loss_fn(y_train, y_pred_train)
            test_losses = [
                loss_fn(y_test, y_pred_test)
                for (_, _, y_test), y_pred_test in zip(test_sets, y_pred_tests)
            ]
            table.add_row(
                loss_name,
                f"{train_loss:.4E}",
                *[f"{test_loss:.4E}" for test_loss in test_losses],
            )

        console.print(table)

# print some samples
n_sample = 10

for test_name, X_test, y_test in test_sets:
    table = Table(title=f"Test samples - {test_name}")
    table.add_column("Real", justify="center", style="green")
    for model_name, _ in models.items():
        table.add_column(model_name, justify="center", style="yellow")

    idx_sample = np.random.choice(X_test.index, size=n_sample, replace=False)
    results = [
        np.array(y_test[y_test.index.intersection(idx_sample)].squeeze())
    ]
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

if input("Save results as baseline? (html only) (y/[n]) ") == "y":
    Path(BASELINES_DIR).mkdir(parents=True, exist_ok=True)
    filename = f"results_{encoding.value}.html"
    results_file = os.path.join(BASELINES_DIR, filename)
    console.save_html(results_file)
    console.log(f"[green]Results stored in {results_file}")


if input("Save models? (y/[n]) ") == "y":
    save_models = {
        "Random Forest": (rf_model, "random_forest_model.pkl"),
        "XGBoost": (xgb_model, "xgb_model.pkl"),
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
