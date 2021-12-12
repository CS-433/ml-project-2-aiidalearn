import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.utils import (
    Encoding,
    LogTransform,
    custom_mape,
    encode_all_structures,
)

# Set Up
DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

MODELS_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()),
    "models/log_delta_E_generalization/",
)

encoding = Encoding.ATOMIC

console = Console(record=True)

# Data Loading
with console.status("") as status:
    status.update("[bold blue]Loading data...")
    df = pd.read_csv(
        os.path.join(DATA_DIR, "data.csv"), index_col=0, na_filter=False
    )

    status.update(f"[bold blue]Encoding structures ({encoding.value})...")
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

    # Log transformation of the output delta_E
    log_transform = LogTransform(y_raw)

    # Train-Test-Split
    p = 0.2
    n_structures = df["structure"].nunique()

    species_test_set = set(
        np.random.choice(
            df["structure"].unique(),
            size=int(p * n_structures),
            replace=False,
        )
    )
    species_train_set = set(
        s for s in df["structure"].unique() if s not in species_test_set
    )
    console.print(
        Panel(
            f"Train set:\t{100*len(species_train_set) / n_structures:.0f}%\n"
            f"Test set:\t{100*len(species_test_set) / n_structures:.0f}%",
            title="[bold]Data distribution",
            style="blue",
            expand=False,
        )
    )

    train_idx = df["structure"].isin(species_train_set)
    test_idx = df["structure"].isin(species_test_set)

    assert train_idx.sum() + test_idx.sum() == len(df)

    X_train = X_raw[train_idx]
    y_train = log_transform.transform(y_raw[train_idx])

    X_test = X_raw[test_idx]
    y_test = log_transform.transform(y_raw[test_idx])
console.log("Data loaded")

# Model Definitions
# functions such that f(x) != 0 and f(+inf) = 0
functions_set_1 = [
    lambda x: np.exp(-x),
    lambda x: 1 / (1 + x),
    lambda x: 1 / (1 + x) ** 2,
    lambda x: np.cos(x) * np.exp(-x),
]

# functions such that f(x) = 0 and f(+inf) = 0
functions_set_2 = [
    lambda x: x * np.exp(-x),
    lambda x: x / (1 + x) ** 2,
    lambda x: x / (1 + x) ** 3,
    lambda x: np.sin(x) * np.exp(-x),
]

linear_augmented_model = Pipeline(
    [
        ("scaler_init", StandardScaler()),
        (
            "features",
            FeatureUnion(
                [
                    (f"fun_{j}", FunctionTransformer(lambda X: f(X[:, :3])))
                    for j, f in enumerate(functions_set_1 + functions_set_2)
                ]
                + [
                    (
                        f"fun_{j}_col_{col}_1",
                        FunctionTransformer(
                            lambda X: f(X[:, :3] * X[:, i][:, None])
                        ),
                    )
                    for j, f in enumerate(functions_set_1)
                    for i, col in enumerate(["ecutrho", "kpoints", "ecutwfc"])
                ]
                + [
                    (
                        f"fun_{j}_col_{col}_2",
                        FunctionTransformer(
                            lambda X: f(X[:, 3:] * X[:, i][:, None])
                        ),
                    )
                    for j, f in enumerate(functions_set_2)
                    for i, col in enumerate(["ecutrho", "kpoints", "ecutwfc"])
                ]
            ),
        ),
        ("scaler_final", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)

rf_model = RandomForestRegressor(n_jobs=-1, random_state=0)

xgb_model = xgb.XGBRegressor(
    max_depth=7, n_estimators=400, learning_rate=1.0, random_state=0
)

lgb_model = lgb.LGBMRegressor(
    max_depth=6,
    num_leaves=10,
    n_estimators=20000,
    learning_rate=1.0,
    n_jobs=-1,
    random_state=0,
)

# detect if gpu is usable with xgboost by training on toy data
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
    # "LightGBM": lgb_model,
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
            ("MSE - log", mean_squared_error),
            ("MAE - log", mean_absolute_error),
            ("MAPE - log", mean_absolute_percentage_error),
        ]:
            train_loss = loss_fn(y_train, y_pred_train)
            test_loss = loss_fn(y_test, y_pred_test)
            table.add_row(loss_name, f"{train_loss:.4E}", f"{test_loss:.4E}")

        # we transform the predictions back to the original form and evaluate
        y_pred_origin_train = log_transform.inverse_transform(
            y_pred_train.squeeze()
        )
        y_pred_origin_test = log_transform.inverse_transform(
            y_pred_test.squeeze()
        )

        y_origin_train = log_transform.inverse_transform(y_train.squeeze())
        y_origin_test = log_transform.inverse_transform(y_test.squeeze())

        for loss_name, loss_fn in [
            ("MSE", mean_squared_error),
            ("MAE", mean_absolute_error),
            ("MAPE", mean_absolute_percentage_error),
            ("Custom MAPE", lambda a, b: custom_mape(a, b, True)),
        ]:
            train_loss = loss_fn(y_origin_train, y_pred_origin_train)
            test_loss = loss_fn(y_origin_test, y_pred_origin_test)
            table.add_row(loss_name, f"{train_loss:.4E}", f"{test_loss:.4E}")

        console.print(table)

# print some samples
n_sample = 10

table = Table(title="Test samples")
table.add_column("Real", justify="center", style="green")
for model_name, _ in models.items():
    table.add_column(model_name, justify="center", style="yellow")

idx_sample = np.random.choice(X_test.index, size=n_sample, replace=False)
results = [
    log_transform.inverse_transform(
        np.array(y_test[y_test.index.intersection(idx_sample)].squeeze())
    )
]
for model_name, model in models.items():
    results.append(
        log_transform.inverse_transform(
            np.array(
                model.predict(
                    X_test.loc[X_test.index.intersection(idx_sample)]
                ).squeeze()
            )
        )
    )

for i in range(n_sample):
    table.add_row(*[f"{r[i]:.3E}" for r in results],)
console.print(table)


# store the terminal output
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
results_file = os.path.join(MODELS_DIR, f"results_{encoding.value}.html")
console.save_html(results_file)
console.print(f"Results stored in {results_file}")
