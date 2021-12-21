import os
import pickle
import sys
from typing import Dict, Tuple, List
from pathlib import Path
from sklearn.base import BaseEstimator

import numpy as np
import pandas as pd
from rich.console import Console

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.utils import StructureEncoding, Target


def save_as_baseline(
    encoding: StructureEncoding,
    console: Console,
    baseline_dir,
    prompt_user=True,
):
    """Function to save console output to a html file to save baseline results in a reproducible way.
    

    Parameters
    ----------
    encoding : StructureEncoding
        Chosen encoding for chemical structures. The underlying string will be part of the filname to distinguish
        results from different encodings.
    console : Console
        Console to which output had been printed beforehand.
    baseline_dir : str
        Path to the directory where the baselines are supposed to be saved to.
    prompt_user : bool, optional
        Whether the user should be prompted before saving. The default is True.

    Returns
    -------
    None.

    """
    if (
        not prompt_user
        or input("Save results as baseline? (html only) (y/[n]) ") == "y"
    ):
        Path(baseline_dir).mkdir(parents=True, exist_ok=True)
        filename = f"results_{encoding.value}.html"
        results_file = os.path.join(baseline_dir, filename)
        console.save_html(results_file)
        console.log(f"[green]Results stored in {results_file}")


def save_models(
    models: Dict[Tuple[BaseEstimator, str]],
    encoding: StructureEncoding,
    console: Console,
    models_dir: str,
    prompt_user=True,
    transformer=None,
):
    """Function to save multiple models to pickle files for later restoration. Functionality to keep track of the
    chosen encoding and transformations is provided. Note that models can be large. Thus, by default, the user is
    prompted before the models are saved to the disk.

    Parameters
    ----------
    models : Dict[Tuple[BaseEstimator, str]]
        Dictionary of tuples containing a model object and a string for the filename.
    encoding : StructureEncoding
        Chosen encoding for the chemical structures. The model will be saved to a corresponding subdirectory.
    console : Console
        Console object for logging purposes.
    models_dir : str
        Path to the directory where the model should be saved.
    prompt_user : bool, optional
        Whether the user should be prompted before saving. The default is True.
    transformer : transformer object from 'transform.py' or similar, optional
        Transformer that has been applied to the data. The default is None.

    Returns
    -------
    None.

    """
    if not prompt_user or input("Save models? (y/[n]) ") == "y":
        models_save_dir = os.path.join(models_dir, encoding.value)
        Path(models_save_dir).mkdir(parents=True, exist_ok=True)
        results_file = os.path.join(models_save_dir, "results.html")
        console.save_html(results_file)
        console.log(f"Results stored in {results_file}")
        if transformer is not None:
            transformer_path = os.path.join(models_save_dir, "transformer.pkl")
            with open(transformer_path, "wb") as file:
                pickle.dump(transformer, file)
                console.log(
                    f"[green]Saved transformer to {transformer_path}[/green]"
                )
        with console.status("[bold green]Saving models..."):
            for model_name, (model, filename) in models.items():
                modelpath = os.path.join(models_save_dir, filename)
                with open(modelpath, "wb") as file:
                    pickle.dump(model, file)
                console.log(
                    f"[green]Saved {model_name} to {modelpath}[/green]"
                )


def save_datasets(
    X_train: np.ndarray,
    y_train: np.array,
    test_sets: List[Tuple[str, np.ndarray, np.array]],
    encoding: StructureEncoding,
    console: Console,
    models_dir: str,
    prompt_user=True,
):
    """Function to save the datasets (train and test) that have been used to train and evaluate a model to a
     subdirectory of the directory, where the model is saved.

    Parameters
    ----------
    X_train : np.ndarray
        Training observations set.
    y_train : np.array
        Training target set.
    test_sets : List[Tuple[str, np.ndarray, np.array]]
        List of test sets.
    encoding : StructureEncoding
        Chosen encoding for the chemical structures.
    console : Console
        Console object for logging objects.
    models_dir : str
        Directory, where model has been saved to. The datasets are then saved into a subdirectory 'datasets'.
    prompt_user : bool, optional
        Whether to prompt the user before saving the datasets. The default is True.

    Returns
    -------
    None.

    """
    if not prompt_user or input("Save datasets? (y/[n]) ") == "y":
        datasets_save_dir = os.path.join(
            models_dir, f"{encoding.value}/datasets"
        )
        Path(datasets_save_dir).mkdir(parents=True, exist_ok=True)
        with console.status("[bold green]Saving datasets..."):
            for name, X, y in [("train", X_train, y_train)] + test_sets:
                X.to_csv(os.path.join(datasets_save_dir, f"X_{name}.csv"))
                y.to_csv(os.path.join(datasets_save_dir, f"y_{name}.csv"))
                console.log(
                    f"[green]Saved {name} to {datasets_save_dir}[/green]"
                )


def save_params(
    encoding: StructureEncoding, target: Target, console: Console, params_dir: str,
):
    """Function to save console output for model parameters to an html file.
    

    Parameters
    ----------
    encoding : StructureEncoding
        Chosen encoding for the chemical structures.
    target : Target
        Target variable, e.g. ∆E.
    console : Console
        Console to which output had been printed beforehand.
    params_dir : str
        Directory to which the parameters should be saved to.

    Returns
    -------
    None.

    """
    Path(params_dir).mkdir(parents=True, exist_ok=True)
    filename = f"params_{target.value}_{encoding.value}.html"
    params_file = os.path.join(params_dir, filename)
    console.save_html(params_file)
    console.log(f"[green]Parameters stored in {params_file}")


def load_saved_datasets(
    encoding: StructureEncoding,
    console: Console,
    models_dir: str,
    prompt_user=True,
) -> Tuple[np.ndarray, np.array, List[Tuple[str, np.ndarray, np.array]]]:
    """Function to reload datasets that have been saved using 'save_datasets'.
    

    Parameters
    ----------
    encoding : StructureEncoding
        Chosen encoding for the chemical structures.
    console : Console
        Console object for logging purposes.
    models_dir : str
        Directory where the model is saved, i.e. same models_dir which has been used in 'save_datasets'.
    prompt_user : bool, optional
        Whether to prompt the user before datasets are loaded. The default is True.

    Returns
    -------
    X_train : np.ndarray
        Train observations set.
    y_train : np.array
        Train target set.
    test_sets : List[Tuple[str, np.ndarray, np.array]]
        List of test sets.

    """
    if not prompt_user or input("Load datasets? (y/[n]) ") == "y":
        test_sets = []
        datasets_load_dir = os.path.join(
            models_dir, f"{encoding.value}/datasets"
        )
        p = Path(datasets_load_dir)
        with console.status("[bold green]Loading datasets..."):
            for file in p.glob("X_*.csv"):
                name = file.name[2:-4]
                X_file = file
                y_file = os.path.join(str(p.absolute()), f"y_{name}.csv")

                X = pd.read_csv(X_file, index_col=0, na_filter=False)
                y = pd.read_csv(y_file, index_col=0, na_filter=False)
                console.log(
                    f"[green]Loaded {name} from {datasets_load_dir}[/green]"
                )
                if name == "train":
                    X_train, y_train = X, y
                else:
                    test_sets.append((name, X, y))
        return X_train, y_train, test_sets
