import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from sklearn.base import TransformerMixin

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.utils import StructureEncoding, Target, encode_all_structures

DATA_DIR = os.path.join(ROOT_DIR, "data/")

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
    """


    Parameters
    ----------
    structure_col : pd.Series
        DESCRIPTION.
    structure_set : Set[str]
        DESCRIPTION.
    n_test_structures : int
        DESCRIPTION.

    Returns
    -------
    train_mask : TYPE
        DESCRIPTION.
    test_mask : TYPE
        DESCRIPTION.
    species_train_set : TYPE
        DESCRIPTION.

    """
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
    data_path: str,
    cols_target: List[str],
    structure_encoding: StructureEncoding = None,
    cols_trash: List[str] = None,
    remove_ref_rows: bool = False,
) -> Tuple[np.ndarray, np.array, pd.Series, pd.Series]:
    """Basic data loading function which encodes the chemical structure and identifies the dependent and independent
    variables.

    Parameters
    ----------
    data_path : str
        Path to the.csv data file.
    cols_target : List[str]
        List with the target column(s).
    structure_encoding : StructureEncoding, optional
        Encoding for the chemical structures, see tools.utils for details. The default is None.
    cols_trash : List[str], optional
        Unused columns that shoud be dropped from the dataset. The default is None.
    remove_ref_rows : bool, optional
        Whether to remove the rows containing the datapoints with the reference energy. The default is False.

    Returns
    -------
    X_raw :  np.ndarray
        raw observation data.
    y_raw : np.array
        raw target data.
    pd.Series
        Series containing the raw structure strings.
    pd.Series
        Boolean Series containing the convergence success marker.

    """
    df = pd.read_csv(data_path, index_col=0, na_filter=False)

    if structure_encoding is not None:
        df = encode_all_structures(df, structure_encoding)

    df["delta_E"] = np.abs(df["delta_E"])

    if remove_ref_rows:
        df = df.loc[df["delta_E"] != 0]

    df.reset_index(drop=True, inplace=True)

    if cols_trash is None:
        cols_trash = []
    cols_drop = cols_trash + cols_target
    feature_cols = list(df.columns)
    for element in cols_drop:
        feature_cols.remove(element)

    X_raw = df[feature_cols]
    y_raw = df[cols_target].squeeze()
    return X_raw, y_raw, df["structure"], df["converged"]


def get_columns(target: Target) -> Tuple[List[str], List[str]]:
    """Function that returns the column names of the columns that have to be dropped from the observations.

    Parameters
    ----------
    target : Target
        Marker for the target variable.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of two lists
        1) List of column names of trash columns -> can be dropped both from observations and target.
        2) List of column names of target columns

    """
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
    remove_ref_rows: bool = False,
) -> Tuple[np.ndarray, np.array, List[Tuple[str, np.ndarray, np.array]]]:
    """Data loading function returning the train set and optionally different test set. It internally calls the
    base_loader.


    Parameters
    ----------
    target : Target, optional
        Marker for the target variable. The default is Target.DELTA_E.
    encoding : StructureEncoding, optional
        Marker for the desired encoding of the chemical structures. The default is StructureEncoding.COLUMN_MASS.
    data_path : str, optional
        Path to the data directory. The default is DATA_PATH, i.e. the data directory in the repository.
    test_sets_cfg : List[TestSet], optional
        List of test set configurations. The default is None.
    target_transformer : TransformerMixin, optional
        Transformer for the target variable, e.g. log or magnitude. The default is None.
    console : Console, optional
        Console object for convenient logging. The default is None.
    remove_ref_rows : bool, optional
        Whether to remove the rows containing the datapoints with the reference energy, cf. base_loader.
        The default is False.

    Returns
    -------
    X_train : np.ndarray
        Dependent variables of the train set.
    y_train : np.array
        Independent variables of the test set.
    test_sets : List[Tuple[str, np.ndarray, np.array]]
        List of tuples for each test set with
        - namestring
        - X_test
        - y_test.

    """
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
            remove_ref_rows=remove_ref_rows,
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
                np.random.seed(42)
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
