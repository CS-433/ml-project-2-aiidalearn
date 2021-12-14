import os
import sys
from pathlib import Path

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
from tools.data_loader import TestSet, TestSplit, data_loader
from tools.save import save_as_baseline, save_datasets, save_models
from tools.train import train_models
from tools.transform import CustomLogTargetTransformer
from tools.utils import (
    StructureEncoding,
    Target,
    check_xgboost_gpu,
    custom_mape,
    percentile_absolute_percentage_error,
)

# Set Up
DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

MODELS_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "models/log_delta_E/"
)

BASELINES_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()),
    "baselines/log_delta_E/",
)


def instantiate_models(console: Console):
    with console.status("") as status:
        status.update("[bold blue]Initializing models...")

        linear_log_augmented_model = Pipeline(
            [
                ("scaler_init", StandardScaler()),
                ("features", PolynomialFeatures(degree=2)),
                ("scaler_final", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )

        rf_log_model = RandomForestRegressor(random_state=0)

        xgb_log_model = xgb.XGBRegressor(
            max_depth=7, n_estimators=400, learning_rate=1.0, random_state=0
        )

        status.update("[bold blue]Checking GPU usability for XGBoost...")
        if check_xgboost_gpu():
            xgb_log_model.set_params(tree_method="gpu_hist")
            console.print("[italic bright_black]Using GPU for XGBoost")
        else:
            console.print("[italic bright_black]Using CPU for XGBoost")

        return {
            "Dummy - log": DummyRegressor(),
            "Linear - log": LinearRegression(),
            # "Augmented Linear - log": linear_log_augmented_model,
            "Random Forest - log": rf_log_model,
            "XGBoost - log": xgb_log_model,
        }


def evaluate_models_log(
    models,
    X_train,
    logy_train,
    test_sets,
    target_transformer,
    console: Console,
):
    with console.status("") as status:
        for model_name, model in models.items():
            status.update(f"[bold blue]Evaluating {model_name}...")

            table = Table(title=model_name)
            table.add_column("Loss name", justify="center", style="cyan")
            table.add_column("Train", justify="center", style="green")
            for name, _, _ in test_sets:
                table.add_column(
                    f"Test - {name}", justify="center", style="green"
                )

            # first we evaluate the log-transformed prediction
            logy_pred_train = model.predict(X_train)
            logy_pred_tests = [
                model.predict(X_test) for _, X_test, _ in test_sets
            ]

            for loss_name, loss_fn in [
                ("MSE - log", mean_squared_error),
                ("MAE - log", mean_absolute_error),
                ("MAPE - log", mean_absolute_percentage_error),
                ("Custom MAPE - log", custom_mape),
                (
                    "50%-APE - log",
                    lambda a, b: percentile_absolute_percentage_error(
                        a, b, 50
                    ),
                ),
                (
                    "90%-APE - log",
                    lambda a, b: percentile_absolute_percentage_error(
                        a, b, 90
                    ),
                ),
            ]:
                train_loss = loss_fn(logy_train, logy_pred_train)
                test_losses = [
                    loss_fn(logy_test, logy_pred_test)
                    for (_, _, logy_test), logy_pred_test in zip(
                        test_sets, logy_pred_tests
                    )
                ]
                table.add_row(
                    loss_name,
                    f"{train_loss:.4E}",
                    *[f"{test_loss:.4E}" for test_loss in test_losses],
                )

            # then we transform the predictions back and evaluate
            y_pred_train = target_transformer.inverse_transform(
                logy_pred_train.squeeze()
            )
            y_pred_tests = [
                target_transformer.inverse_transform(logy_pred_test.squeeze())
                for logy_pred_test in logy_pred_tests
            ]

            y_train = target_transformer.inverse_transform(
                logy_train.squeeze()
            )
            y_tests = [
                target_transformer.inverse_transform(logy_test.squeeze())
                for _, _, logy_test in test_sets
            ]

            for loss_name, loss_fn in [
                ("MSE", mean_squared_error),
                ("MAE", mean_absolute_error),
                ("MAPE", mean_absolute_percentage_error),
                ("Custom MAPE", lambda a, b: custom_mape(a, b, True)),
                (
                    "50%-APE",
                    lambda a, b: percentile_absolute_percentage_error(
                        a, b, 50
                    ),
                ),
                (
                    "90%-APE",
                    lambda a, b: percentile_absolute_percentage_error(
                        a, b, 90
                    ),
                ),
            ]:
                train_loss = loss_fn(y_train, y_pred_train)
                test_losses = [
                    loss_fn(y_test, y_pred_test)
                    for y_test, y_pred_test in zip(y_tests, y_pred_tests)
                ]
                table.add_row(
                    loss_name,
                    f"{train_loss:.4E}",
                    *[f"{test_loss:.4E}" for test_loss in test_losses],
                )

            console.print(table)


def print_test_samples_log(
    models, test_sets, target_transformer, console: Console, n_sample=10
):
    for test_name, X_test, logy_test in test_sets:
        table = Table(title=f"Test samples - {test_name}")
        table.add_column("Real", justify="center", style="green")
        for model_name, _ in models.items():
            table.add_column(model_name, justify="center", style="yellow")

        idx_sample = np.random.choice(
            X_test.index, size=n_sample, replace=False
        )
        results = [
            np.array(
                target_transformer.inverse_transform(
                    logy_test[
                        logy_test.index.intersection(idx_sample)
                    ].squeeze()
                )
            )
        ]
        for model_name, model in models.items():
            results.append(
                target_transformer.inverse_transform(
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


if __name__ == "__main__":
    console = Console(record=True)
    prompt_user = True

    encoding = StructureEncoding.ATOMIC
    target = Target.DELTA_E
    target_transformer = CustomLogTargetTransformer()
    test_sets_cfg = [
        TestSet("Parameter gen.", size=0.1, split=TestSplit.ROW),
        TestSet("Structure gen.", size=0.1, split=TestSplit.STRUCTURE),
    ]

    # Data Loading
    X_train, logy_train, test_sets = data_loader(
        target=target,
        encoding=encoding,
        data_path=DATA_PATH,
        test_sets_cfg=test_sets_cfg,
        target_transformer=target_transformer,
        console=console,
        remove_ref_rows=True,
    )

    models = instantiate_models(console)
    train_models(models, X_train, logy_train, console)
    evaluate_models_log(
        models, X_train, logy_train, test_sets, target_transformer, console
    )
    print_test_samples_log(models, test_sets, target_transformer, console)
    save_as_baseline(encoding, console, BASELINES_DIR, prompt_user)

    models_to_save = {
        "Random Forest - log": (
            models["Random Forest - log"],
            "random_forest_model.pkl",
        ),
        "XGBoost - log": (models["XGBoost - log"], "xgboost_model.pkl"),
    }
    save_models(models_to_save, encoding, console, MODELS_DIR, prompt_user)

    save_datasets(
        X_train,
        logy_train,
        test_sets,
        encoding,
        console,
        MODELS_DIR,
        prompt_user,
    )
