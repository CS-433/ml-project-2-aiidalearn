import os
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.data_loader import TestSet, TestSplit, data_loader
from tools.save import save_params
from tools.train import train_models
from tools.transform import CustomLogTargetTransformer
from tools.utils import (
    StructureEncoding,
    Target,
    custom_mape,
    percentile_absolute_percentage_error,
)

# Define global variables
DATA_DIR = os.path.join(ROOT_DIR, "data/")

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

PARAMS_DIR = os.path.join(ROOT_DIR, "hyperparameter_tuning/log_delta_E/")


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


if __name__ == "__main__":
    console = Console(record=True)
    prompt_user = False

    encodings = [StructureEncoding.ATOMIC]
    # encodings = list(StructureEncoding)
    for encoding in encodings:
        console.log(
            f"[bold green] Started training pipeline for {encoding.value} encoding"
        )
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

        base_model = RandomForestRegressor(random_state=0)

        # Number of trees in random forest
        n_estimators = [
            int(x) for x in np.linspace(start=10, stop=1000, num=20)
        ]
        # Number of features to consider at every split
        max_features = ["auto", "sqrt"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(50, 250, num=10)]
        max_depth.append(None)
        param_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
        }

        model_random = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=15,
            cv=2,
            verbose=100,
        )

        console.log("Performing randomized grid search")
        model_random.fit(X_train, logy_train)
        best_model = RandomForestRegressor(**model_random.best_params_)
        console.log("Finished randomized grid search")

        table = Table(title="Best Parameters")
        table.add_column("n_estimators", justify="center", style="white")
        table.add_column("max_features", justify="center", style="white")
        table.add_column("max_depth", justify="center", style="white")
        table.add_row(
            f"{model_random.best_params_['n_estimators']}",
            f"{model_random.best_params_['max_features']}",
            f"{model_random.best_params_['max_depth']}",
        )

        console.print(table)

        save_params(encoding, target, console, PARAMS_DIR)

        models = {"base model": base_model, "best model": best_model}
        train_models(models, X_train, logy_train, console)
        evaluate_models_log(models, X_train, logy_train, test_sets, console)
