import os
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.utils import custom_mape, percentile_absolute_percentage_error


def train_models(models, X_train, y_train, console: Console):
    with console.status("") as status:
        for model_name, model in models.items():
            status.update(f"[bold blue]Training {model_name}...")
            model.fit(X_train, y_train)
            console.log(f"[blue]Finished training {model_name}[/blue]")


def evaluate_models(models, X_train, y_train, test_sets, console: Console):
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

            y_pred_train = model.predict(X_train)
            y_pred_tests = [
                model.predict(X_test) for _, X_test, _ in test_sets
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
                    for (_, _, y_test), y_pred_test in zip(
                        test_sets, y_pred_tests
                    )
                ]
                table.add_row(
                    loss_name,
                    f"{train_loss:.4E}",
                    *[f"{test_loss:.4E}" for test_loss in test_losses],
                )

            console.print(table)


def print_test_samples(models, test_sets, console: Console, n_sample=10):
    for test_name, X_test, y_test in test_sets:
        table = Table(title=f"Test samples - {test_name}")
        table.add_column("Real", justify="center", style="green")
        for model_name, _ in models.items():
            table.add_column(model_name, justify="center", style="yellow")

        idx_sample = np.random.choice(
            X_test.index, size=n_sample, replace=False
        )
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
