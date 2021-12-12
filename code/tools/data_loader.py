import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

from tools.utils import Encoding, LogTransform, Target, encode_all_structures

DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")


def get_generalisation_idx(df, test_size=0.2):
    n_structures = df["structure"].nunique()

    species_test_set = set(
        np.random.choice(
            df["structure"].unique(),
            size=int(test_size * n_structures),
            replace=False,
        )
    )
    species_train_set = set(
        s for s in df["structure"].unique() if s not in species_test_set
    )

    train_idx = df["structure"].isin(species_train_set)
    test_idx = df["structure"].isin(species_test_set)

    assert train_idx.sum() + test_idx.sum() == len(df)
    return train_idx, test_idx


def base_loader(
    cols_trash,
    cols_independent,
    encoding=Encoding.COLUMN_MASS,
    data_path=DATA_PATH,
):

    df = pd.read_csv(data_path, index_col=0, na_filter=False)
    df = encode_all_structures(df, encoding)

    cols_raw = list(df.columns)
    cols_drop = cols_trash + cols_independent

    cols_dependent = cols_raw.copy()
    for element in cols_drop:
        cols_dependent.remove(element)

    X_raw = df[cols_dependent][df["converged"]]
    y_raw = np.abs(df[cols_independent][df["converged"]]).squeeze()
    return X_raw, y_raw, df


def delta_E_loader(
    data_path=DATA_PATH,
    encoding=Encoding.COLUMN_MASS,
    test_size=0.2,
    log=False,
    generalization=False,
):
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

    X_raw, y_raw, df = base_loader(
        cols_trash, cols_independent, encoding=encoding, data_path=data_path
    )

    # Train-Test-Split
    if generalization:

        train_idx, test_idx = get_generalisation_idx(df, test_size)

        X_train = X_raw[train_idx]
        y_train = y_raw[train_idx]

        X_test = X_raw[test_idx]
        y_test = y_raw[test_idx]

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=42
        )

    if log:
        log_transform = LogTransform(y_raw)
        return (
            X_train,
            X_test,
            log_transform.transform(y_train),
            log_transform.transform(y_test),
        )

    else:
        return X_train, X_test, y_train, y_test


def sim_time_loader(
    data_path=DATA_PATH,
    encoding=Encoding.COLUMN_MASS,
    test_size=0.2,
    generalization=False,
):
    cols_trash = [
        "structure",
        "converged",
        "accuracy",
        "n_iterations",
        "delta_E",
        "fermi",
        "total_energy",
    ]
    cols_independent = ["time"]

    X_raw, y_raw, df = base_loader(
        cols_trash, cols_independent, encoding=encoding, data_path=data_path
    )

    # Train-Test-Split
    if generalization:

        train_idx, test_idx = get_generalisation_idx(df, test_size=test_size)

        X_train = X_raw[train_idx]
        y_train = y_raw[train_idx]

        X_test = X_raw[test_idx]
        y_test = y_raw[test_idx]

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=42
        )

    return X_train, X_test, y_train, y_test


def data_loader(
    target=Target.DELTA_E,
    encoding=Encoding.COLUMN_MASS,
    data_path=DATA_PATH,
    test_size=0.2,
    generalization=False,
):
    console = Console()
    with console.status("") as status:
        status.update("Loading data")
        if target == Target.DELTA_E:
            X_train, X_test, y_train, y_test = delta_E_loader(
                data_path=data_path,
                encoding=encoding,
                test_size=test_size,
                log=False,
                generalization=generalization,
            )
        if target == Target.LOG_DELTA_E:
            X_train, X_test, y_train, y_test = delta_E_loader(
                data_path=data_path,
                encoding=encoding,
                test_size=test_size,
                log=True,
                generalization=generalization,
            )
        if target == Target.SIM_TIME:
            X_train, X_test, y_train, y_test = sim_time_loader(
                data_path=data_path,
                encoding=encoding,
                test_size=test_size,
                generalization=generalization,
            )

    table = Table(title="Loaded Dataset", show_header=False, show_lines=False)
    table.add_row(
        f"Train set: {100*(1.0 - test_size):.0f}%",
        f"Test set: {100*test_size:.0f}%",
    )
    table.add_row(f"Target: {target.value}", f"Encoding: {encoding.value}")
    table.add_row(
        f"Gen. Training:\t{generalization}",
        f"Size: ~{sys.getsizeof(X_train)/(1-test_size)*1E-6:.1f}MB",
    )
    console.print(table)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_loader(
        target=Target.SIM_TIME,
        encoding=Encoding.COLUMN_MASS,
        data_path=DATA_PATH,
        generalization=True,
    )
