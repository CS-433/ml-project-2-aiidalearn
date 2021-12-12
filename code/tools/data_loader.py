import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from sklearn.base import TransformerMixin

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.utils import StructureEncoding, Target, encode_all_structures

DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")


class TestSplit(Enum):
    ROW = "row"
    STRUCTURE = "structure"


class TestSet:
    name: str
    size: float
    split: TestSplit

    def __init__(
        self, name: str, size: float = 0.2, split: TestSplit = TestSplit.ROW
    ):
        self.name = name
        self.size = size
        self.split = split


def train_test_structure_split(
    structure_col: pd.Series, structure_set: Set[str], n_test_structures: int
):
    np.random.seed(42)
    species_test_set = set(
        np.random.choice(
            list(structure_set), size=n_test_structures, replace=False
        )
    )
    species_train_set = set(
        s for s in structure_set if s not in species_test_set
    )

    train_mask = structure_col.isin(species_train_set)
    test_mask = structure_col.isin(species_test_set)

    return train_mask, test_mask, species_train_set


def base_loader(
    data_path,
    cols_target: List[str],
    structure_encoding: StructureEncoding = None,
    cols_trash: List[str] = None,
):
    df = pd.read_csv(data_path, index_col=0, na_filter=False)

    if structure_encoding is not None:
        df = encode_all_structures(df, structure_encoding)

    df["delta_E"] = np.abs(df["delta_E"])

    if cols_trash is None:
        cols_trash = []
    cols_drop = cols_trash + cols_target
    feature_cols = list(df.columns)
    for element in cols_drop:
        feature_cols.remove(element)

    X_raw = df[feature_cols]
    y_raw = df[cols_target].squeeze()
    return X_raw, y_raw, df["structure"], df["converged"]


def get_columns(target):
    cols_trash = [
        "structure",
        "accuracy",
        "n_iterations",
        "fermi",
        "total_energy",
    ]
    if target == Target.DELTA_E:
        cols_trash += ["converged", "time"]
        cols_target = ["delta_E"]
    elif target == Target.SIM_TIME:
        cols_trash += ["converged", "delta_E"]
        cols_target = ["time"]
    elif target == Target.CONVERGED:
        cols_trash += ["delta_E", "time"]
        cols_target = ["converged"]
    return cols_trash, cols_target


def data_loader(
    target: Target = Target.DELTA_E,
    encoding: StructureEncoding = StructureEncoding.COLUMN_MASS,
    data_path=DATA_PATH,
    test_sets_cfg: List[TestSet] = None,
    target_transformer: TransformerMixin = None,
    console: Console = None,
):
    if console is None:
        console = Console()
    with console.status("") as status:
        status.update("Loading data")
        cols_trash, cols_target = get_columns(target)

        X_raw, y_raw, structure_col, converged_col = base_loader(
            data_path=data_path,
            cols_target=cols_target,
            structure_encoding=encoding,
            cols_trash=cols_trash,
        )

        if target != Target.CONVERGED:
            # keep only converged rows
            X_raw = X_raw[converged_col]
            y_raw = y_raw[converged_col]
            structure_col = structure_col[converged_col]

        if target_transformer is not None:
            y_raw = target_transformer.fit_transform(y_raw)

        train_mask = np.ones(X_raw.shape[0], dtype=bool)
        test_sets = []
        if test_sets_cfg is not None:
            structures_train = set(structure_col.unique())
            test_sets_rows = []
            for i, cfg in enumerate(test_sets_cfg):
                if cfg.split == TestSplit.STRUCTURE:
                    (
                        train_mask,
                        test_mask,
                        structures_train,
                    ) = train_test_structure_split(
                        structure_col=structure_col,
                        structure_set=structures_train,
                        n_test_structures=int(
                            cfg.size * structure_col.nunique()
                        ),
                    )
                    test_sets.append(
                        (cfg.name, X_raw[test_mask], y_raw[test_mask])
                    )
                elif cfg.split == TestSplit.ROW:
                    test_sets.append(None)
                    test_sets_rows.append((i, cfg))

            for i, cfg in test_sets_rows:
                n_test_rows = int(cfg.size * X_raw.shape[0])
                # get randomly n_test_rows rows from train_idx true rows
                test_idx = np.random.choice(
                    np.where(train_mask)[0], size=n_test_rows, replace=False
                )
                train_mask[test_idx] = False
                test_mask = np.zeros(X_raw.shape[0], dtype=bool)
                test_mask[test_idx] = True
                test_sets[i] = (cfg.name, X_raw[test_mask], y_raw[test_mask])

        X_train, y_train = X_raw[train_mask], y_raw[train_mask]

    console.print(
        Panel(
            (
                f"Train set: {100*X_train.shape[0]/X_raw.shape[0]:.0f}%\n"
                + "\n".join(
                    [
                        f"Test set {i} ({name}): {100*X_test.shape[0]/X_raw.shape[0]:.0f}%"
                        for i, (name, X_test, _) in enumerate(test_sets)
                    ]
                )
                + f"\nTotal datapoints: {X_raw.shape[0]}"
                + f"\nSize: ~{sys.getsizeof(X_raw)*1E-6:.1f}MB"
            ),
            title="Loaded Dataset",
            expand=False,
        )
    )

    assert X_raw.shape[0] == X_train.shape[0] + sum(
        X_test.shape[0] for _, X_test, _ in test_sets
    )

    return X_train, y_train, test_sets


if __name__ == "__main__":
    X_train, y_train, _ = data_loader(
        test_sets_cfg=[
            TestSet(name="test", size=0.1),
            TestSet(name="test2", size=0.1, split=TestSplit.STRUCTURE),
        ]
    )
