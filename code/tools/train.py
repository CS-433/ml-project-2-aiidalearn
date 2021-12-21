import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.utils import custom_mape, percentile_absolute_percentage_error


def train_models(
    models: Dict[BaseEstimator],
    X_train: np.ndarray,
    y_train: np.array,
    console: Console,
) -> object:
    """Wrapper around the sklearn.BaseEstimator method 'fit' with logging functionality.


    Parameters
    ----------
    models : Dict[BaseEstimator]
        Dictionary of models
    X_train : np.ndarray
        Train set.
    y_train : np.array
        Train target set.
    console : Console
        Console object for logging purposes.

    Returns
    -------
    None.

    """
    with console.status("") as status:
        for model_name, model in models.items():
            status.update(f"[bold blue]Training {model_name}...")
            model.fit(X_train, y_train)
            console.log(f"[blue]Finished training {model_name}[/blue]")


def evaluate_models(
    models: Dict[BaseEstimator, str],
    X_train: np.ndarray,
    y_train: np.array,
    test_sets: List[Tuple[str, np.ndarray, np.array]],
    console: Console,
):
    """Wrapper around the sklearn.BaseEstimator method 'predict' and scoring metrics for REGRESSION MODELS
     with logging functionality. Works well in combination with 'tools.save.save_as_baseline'. For CLASSIFIERS see
     'evaluate_classifiers'.


    Parameters
    ----------
    models : Dict[BaseEstimator]
        Dictionary of models
    X_train : np.ndarray
        Train set.
    y_train : np.array
        Train target set.
    test_sets : List[Tuple[str, np.ndarray, np.array]]
        List of test sets on which the models should be evaluated.
    console : Console
        Console object for logging purposes.

    Returns
    -------
    None.

    """
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


def evaluate_classifiers(
    models: Dict[BaseEstimator, str],
    X_train: np.ndarray,
    y_train: np.array,
    test_sets: List[Tuple[str, np.ndarray, np.array]],
    console: Console,
):
    """Wrapper around the sklearn.BaseEstimator method 'predict' and scoring metrics for CLASSIFIERS
     with logging functionality. Works well in combination with 'tools.save.save_as_baseline'. For REGRESSION MODELS see
     'evaluate_models'.


    Parameters
    ----------
    models : Dict[BaseEstimator]
        Dictionary of models
    X_train : np.ndarray
        Train set.
    y_train : np.array
        Train target set.
    test_sets : List[Tuple[str, np.ndarray, np.array]]
        List of test sets on which the models should be evaluated.
    console : Console
        Console object for logging purposes.

    Returns
    -------
    None.

    """
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
                ("Accuracy", accuracy_score),
            ]:
                train_loss = loss_fn(y_train, y_pred_train)
                test_losses = [
                    loss_fn(y_test, y_pred_test)
                    for (_, _, y_test), y_pred_test in zip(
                        test_sets, y_pred_tests
                    )
                ]

                if loss_name == "Accuracy":
                    table.add_row(
                        loss_name,
                        f"{100*train_loss:.2f}%",
                        *[
                            f"{100*test_loss:.2f}%"
                            for test_loss in test_losses
                        ],
                    )
                else:
                    table.add_row(
                        loss_name,
                        f"{train_loss:.4E}",
                        *[f"{test_loss:.4E}" for test_loss in test_losses],
                    )

            console.print(table)


def cv_classifiers(
    models: Dict[BaseEstimator, str],
    X_train: np.ndarray,
    y_train: np.array,
    console: Console,
    ncv=5,
    shuffle=False,
):
    """Wrapper around sklearn.model_selection.cross_validate and  ..KFold to cross-validate multiple classifiers with
    logging to console. Also works with 'tools.save.save_as_baseline' to save the console output as html. Note that
    in the context of this project, shuffle=False corresponds to CV w.r.t to unseen structures while shuffle=False
    corresponds to CV w.r.t to unseen parameters. For details see report.

    Parameters
    ----------
    models : Dict[BaseEstimator]
        Dictionary of models
    X_train : np.ndarray
        Train set.
    y_train : np.array
        Train target set.
    console : Console
        Console object for logging purposes.
    ncv : int, optional
        Number of splits in cross-vaidation. The default is 5.
    shuffle : bool, optional
        Whether the rows should be shuffled before performing the splits. The default is False.

    Returns
    -------
    None.

    """
    if shuffle:
        table = Table(title=f"Cross Validation {ncv}-fold with shuffle")
    else:
        table = Table(title=f"Cross Validation {ncv}-fold without shuffle")

    table.add_column("Model", justify="center", style="white")
    table.add_column("Train mean", justify="center", style="white")
    table.add_column("Train std", justify="center", style="white")
    table.add_column("Test mean", justify="center", style="white")
    table.add_column("Test std", justify="center", style="white")

    with console.status("") as status:
        for model_name, model in models.items():
            status.update(f"[bold blue]Cross validating {model_name}...")
            if shuffle:
                cv_iterators = KFold(
                    n_splits=ncv, shuffle=True, random_state=42
                )
            else:
                cv_iterators = KFold(n_splits=ncv, shuffle=False)
            cv_res = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv_iterators,
                return_train_score=True,
            )
            train_scores = cv_res["train_score"]
            test_scores = cv_res["test_score"]
            table.add_row(
                f"{model_name}",
                f"{100*np.mean(train_scores):.2f}%",
                f"{np.std(train_scores):.6f}",
                f"{100*np.mean(test_scores):.2f}%",
                f"{np.std(test_scores):.6f}",
            )

    console.print(table)


def print_test_samples(
    models: Dict[BaseEstimator],
    test_sets: List[Tuple[str, np.ndarray, np.array]],
    console: Console,
    n_sample=10,
):
    """Print function for output of sample predictions for manual inspection.


    Parameters
    ----------
    models : Dict[BaseEstimator, str]
        Dictionary of models.
    test_sets : List[Tuple[str, np.ndarray, np.array]]
        List of test sets.
    console : Console
        Console object to print the samples to.
    n_sample : int, optional
        number of samples that should be printed. The default is 10.

    Returns
    -------
    None.

    """
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


def print_problematic_samples(
    models: Dict[BaseEstimator],
    test_sets: List[Tuple[str, np.ndarray, np.array]],
    console: Console,
    elts: List[str],
    n_sample: object = 10,
):
    """Print function for output of sample predictions for manual inspection of suspicious elements.


    Parameters
    ----------
    models : Dict[BaseEstimator, str]
        Dictionary of models.
    test_sets : List[Tuple[str, np.ndarray, np.array]]
        List of test sets.
    console : Console
        Console object to which the results are printed.
    elts : List[str]
        List of element keys, e.g. 'H', 'Fe'.
    n_sample : object, optional
        Number of samples that should be printed. The default is 10.

    Returns
    -------
    None.

    """
    for elt in elts:
        for test_name, X_test, y_test in test_sets:
            table = Table(title=f"Test samples - {test_name}")
            table.add_column("Real", justify="center", style="green")
            for model_name, _ in models.items():
                table.add_column(model_name, justify="center", style="yellow")

            elt_mask = X_test[elt].to_numpy().nonzero()[0]
            idx_sample = np.random.choice(
                X_test.iloc[elt_mask].index, size=n_sample, replace=False
            )
            results = [
                np.array(
                    y_test[y_test.index.intersection(idx_sample)].squeeze()
                )
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
