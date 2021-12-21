import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, TransformerMixin

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.utils import (
    PERIODIC_TABLE_INFO,
    PTC_COLNAMES,
    StructureEncoding,
    get_structure_encoding,
)

DELTA_E_MODELS_DIR = os.path.join(ROOT_DIR, "models/delta_E/")
LOG_DELTA_E_MODELS_DIR = os.path.join(ROOT_DIR, "models/log_delta_E/")
SIM_TIME_MODELS_DIR = os.path.join(ROOT_DIR, "models/sim_time/")

DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "data.csv")


def load_models(
    delta_E_model_name: str = None,
    log_delta_E_model_name: str = None,
    sim_time_model_name: str = None,
):
    """Function to load the pretrained models from their pickle files.


    Parameters
    ----------
    delta_E_model_name : str, optional
        Path to delta_E_model relative to directory 'models'. The default is None.
    log_delta_E_model_name : str, optional
        Path to log_delta_E_model relative to directory 'models'. The default is None.
    sim_time_model_name : str, optional
        Path to sim_time relative to directory 'models'. The default is None.

    Raises
    ------
    FileNotFoundError
        when models cannot be found in the specified directories.

    Yields
    ------
    BaseEstimator
        Models loaded from pickle files.

    """
    for model_name, model_dir in [
        (delta_E_model_name, DELTA_E_MODELS_DIR),
        (log_delta_E_model_name, LOG_DELTA_E_MODELS_DIR),
        (sim_time_model_name, SIM_TIME_MODELS_DIR),
    ]:
        if model_name is None:
            continue
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model file "
                + model_path
                + " does not exist. Please run the training script first."
            )
        with open(model_path, "rb") as file:
            yield pickle.load(file)


def get_features_name(encoding: StructureEncoding) -> List[str]:
    """Helper function to retrieve the column names of the data frame with a specified encoding. Simplifies the
    construction of inputs to the models in 'delta_E_prediction' and 'sim_time_prediction'.

    Parameters
    ----------
    encoding : StructureEncoding
        Chosen encoding for the chemical structures.

    Returns
    -------
    List[str]
        Column names of the data frame with the specified encoding.

    """
    res = ["ecutrho", "k_density", "ecutwfc"]
    if encoding == StructureEncoding.ATOMIC:
        res += list(PERIODIC_TABLE_INFO.keys())
    elif encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        res += PTC_COLNAMES
    return res + ["total_atoms"]


def sanitize_input(x: np.array) -> np.array:
    """Rounds values in an array of floats to the nearest integer.

    Parameters
    ----------
    x : np.array
        array containing arbitrary floating point numbers.

    Returns
    -------
    np.array
        array containing rounded values to the nearest integer.

    """
    return np.array([int(round(x_i)) for x_i in x])


def delta_E_prediction(
    x: np.array,
    model: BaseEstimator,
    structure_encoding: np.array,
    delta_E_features: List[str],
) -> float:
    """Wrapper around BaseModel.predict() for the predictor of ∆E. This function is used to fix the structure and the
    encoding but vary the parameters ["ecutrho", "k_density", "ecutwfc"].


    Parameters
    ----------
    x : np.array
        Array containing the simulation parameters ["ecutrho", "k_density", "ecutwfc"].
    model : BaseEstimator
        sklearn model predicting ∆E for a specific encoding (StructureEncoding).
    structure_encoding : np.array
        Array containing an encoded chemical structure with the encoding on which the model has been trained.
    delta_E_features : List[str]
        List of column names of the train set on which the model has been trained.

    Returns
    -------
    float
        prediction of ∆E.

    """
    input_value = np.concatenate([x, structure_encoding])
    input_value = pd.DataFrame(
        input_value.reshape(1, -1), columns=delta_E_features
    )
    return model.predict(input_value)[0]


def sim_time_prediction(
    x: np.array,
    model: BaseEstimator,
    structure_encoding: np.array,
    sim_time_features: List[str],
) -> float:
    """Wrapper around BaseModel.predict() for the predictor of sim_time. This function is used to fix the structure
     and the encoding but vary the parameters ["ecutrho", "k_density", "ecutwfc"].


    Parameters
    ----------
    x : np.array
        Array containing the simulation parameters ["ecutrho", "k_density", "ecutwfc"].
    model : BaseEstimator
        sklearn model predicting sim_time for a specific encoding (StructureEncoding).
    structure_encoding : np.array
        Array containing an encoded chemical structure with the encoding on which the model has been trained.
    sim_time_features : List[str]
        List of column names of the train set on which the model has been trained.

    Returns
    -------
    float
        prediction of sim_time.

    """
    input_value = np.concatenate([x, structure_encoding])
    input_value = pd.DataFrame(
        input_value.reshape(1, -1), columns=sim_time_features
    )
    return model.predict(input_value)[0]


def get_optimal_parameters(
    structure_name: str,
    max_delta_E: float,
    encoding_delta_E: StructureEncoding,
    encoding_sim_time: StructureEncoding,
    feature_bounds: dict,
    delta_E_model: BaseEstimator = None,
    sim_time_model: BaseEstimator = None,
    log_delta_E_model: BaseEstimator = None,
    transformer: TransformerMixin = None,
    verbose: bool = False,
) -> Tuple[np.array, float, float]:
    """


    Parameters
    ----------
    structure_name : str
        Name of the chemical structure.
    max_delta_E : float
        Max. accepted value of ∆E.
    encoding_delta_E : StructureEncoding
        Encoding of the chemical structures used in the delta_E model.
    encoding_sim_time : StructureEncoding
        Encoding of the chemical structures used in the sim_time_model.
    feature_bounds : dict
        Bounds for the domain of the features ["ecutrho", "k_density", "ecutwfc"].
    delta_E_model : BaseEstimator, optional
        Model predicting ∆E. The default is None.
    sim_time_model : BaseEstimator, optional
        Model predicting the simulation time. The default is None.
    log_delta_E_model : BaseEstimator, optional
        Model predicting log(∆E). The default is None.
    transformer : Transformer, optional
        Transformer of the target value. The default is None. Necessary for log_delta_E_model.
    verbose : bool, optional
        Verbosity level of output. The default is False.

    Raises
    ------
    ValueError
        when transformer for log(∆E) model is not provided.

    Returns
    -------
    Tuple[np.array, float, float]
        Array with the optimal parameters, the predicted simulation time with the optimal parameters, predicted value
        of ∆E.

    """
    if log_delta_E_model is not None and delta_E_model is not None:
        raise ValueError(
            "Only one of delta_E_model and log_delta_E_model can be provided"
        )

    if log_delta_E_model is None and delta_E_model is None:
        raise ValueError(
            "One of delta_E_model and log_delta_E_model must be provided"
        )

    if sim_time_model is None:
        raise ValueError("sim_time_model must be provided")

    structure_encoding_delta_E = get_structure_encoding(
        structure_name, encoding_delta_E
    )
    structure_encoding_sim_time = get_structure_encoding(
        structure_name, encoding_sim_time
    )

    if delta_E_model is not None:

        def delta_E_pred_func(x):
            """Function to make a prediction of delta_E for a fixed structure
            """
            return delta_E_prediction(
                sanitize_input(x),
                delta_E_model,
                structure_encoding_delta_E,
                get_features_name(encoding_delta_E),
            )

    if log_delta_E_model is not None:
        if transformer is None:
            raise ValueError("transformer must be provided")

        def delta_E_pred_func(x):
            """Function to make a prediction of delta_E for a fixed structure
            """
            return transformer.inverse_transform(
                delta_E_prediction(
                    sanitize_input(x),
                    log_delta_E_model,
                    structure_encoding_delta_E,
                    get_features_name(encoding_delta_E),
                )
            )

    def sim_time_pred_func(x):
        """Function to make a prediction of the simulation time for a fixed structure
        """
        return sim_time_prediction(
            sanitize_input(x),
            sim_time_model,
            structure_encoding_sim_time,
            get_features_name(encoding_sim_time),
        )

    mu = 1e100  # Penalization parameter

    def pen_func(x):
        """Function to be optimizied to solve the penalized problem. Penalization can be adjusted by changing 'mu'.
        """
        return (
            sim_time_pred_func(sanitize_input(x))
            + mu
            * max(delta_E_pred_func(sanitize_input(x)) - max_delta_E, 0)
            / max_delta_E
        )

    res = differential_evolution(
        pen_func,
        bounds=[
            (feature_bounds["ecutrho"][0], feature_bounds["ecutrho"][1]),
            (feature_bounds["k_density"][0], feature_bounds["k_density"][1]),
            (feature_bounds["ecutwfc"][0], feature_bounds["ecutwfc"][1]),
        ],
        seed=0,
    )

    x_f = sanitize_input(res.x)
    sim_time = sim_time_pred_func(x_f)
    delta_E = delta_E_pred_func(x_f)

    if verbose:
        print("=======OPTIMIZATION PARAMETERS=======")
        print(f"∆E_max: {max_delta_E:.3E}")
        print(f"Structure: {structure_name}")
        print("========OPTIMIZATION RESULTS=========")
        print(res)
        print("=============PREDICTIONS=============")
        print(f"Parameters: {x_f}\tSim. time: {sim_time}\t∆E: {delta_E:.3E}")

    return x_f, sim_time, delta_E


def get_feature_bounds(data_path: str) -> Dict[Tuple[float, float]]:
    """Helper function that determines the domain of the parameters ["ecutrho", "k_density", "ecutwfc"] in the dataset.
    This is necessary since Random Forest Model by definition cannot extrapolate. Therefore, the optimizer can only
    search for optimal parameters within these bounds.

    Parameters
    ----------
    data_path : str
        Path to data.csv file with all the parsed data.

    Returns
    -------
    Dict[Tuple[float, float]]
        Dictionary with an entry for each parameter in["ecutrho", "k_density", "ecutwfc"] with upper and lower bound.

    """
    data = pd.read_csv(data_path, na_filter=False)
    return {
        "ecutrho": (data["ecutrho"].min(), data["ecutrho"].max()),
        "k_density": (data["k_density"].min(), data["k_density"].max()),
        "ecutwfc": (data["ecutwfc"].min(), data["ecutwfc"].max()),
    }


if __name__ == "__main__":
    console = Console()

    encoding_delta_E = StructureEncoding.ATOMIC
    encoding_sim_time = StructureEncoding.ATOMIC

    with console.status("Loading models..."):
        delta_E_model, log_delta_E_model, sim_time_model = None, None, None
        log_delta_E_model, sim_time_model = load_models(
            log_delta_E_model_name=f"{encoding_delta_E.value}/random_forest_model.pkl",
            sim_time_model_name=f"{encoding_sim_time.value}/random_forest_model.pkl",
            # log_delta_E_model_name=f"{encoding_delta_E.value}/xgboost_model.pkl",
            # sim_time_model_name=f"{encoding_sim_time.value}/xgboost_model.pkl",
        )

    transformer = None
    if log_delta_E_model is not None:
        transformer_path = os.path.join(
            LOG_DELTA_E_MODELS_DIR,
            f"{encoding_delta_E.value}/transformer.pkl",
        )
        with open(transformer_path, "rb") as file:
            transformer = pickle.load(file)

    feature_bounds = get_feature_bounds(DATA_PATH)

    # structure_list = ["Ag1Cl1", "Ba1S1"]
    # get all the structures in the data folder
    structure_list = list(
        set(
            structure
            for structure in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, structure))
        )
    )[:100]

    max_delta_E_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    predictions = []
    for structure_name in track(
        structure_list, description="Optimizing parameters...", console=console
    ):
        for max_delta_E in max_delta_E_list:
            console.print(
                f"Structure: {structure_name}\t∆E_max: {max_delta_E:.2e}"
            )
            params, sim_time_pred, delta_E_pred = get_optimal_parameters(
                # delta_E_model=delta_E_model,
                log_delta_E_model=log_delta_E_model,
                sim_time_model=sim_time_model,
                structure_name=structure_name,
                max_delta_E=max_delta_E,
                encoding_delta_E=encoding_delta_E,
                encoding_sim_time=encoding_sim_time,
                feature_bounds=feature_bounds,
                transformer=transformer,
            )
            predictions.append(
                {
                    "structure": structure_name,
                    "max_delta_E": float(max_delta_E),
                    "params": {
                        "ecutrho": int(params[0]),
                        "k_density": int(params[1]),
                        "ecutwfc": int(params[2]),
                    },
                    "sim_time_pred": float(sim_time_pred),
                    "delta_E_pred": float(delta_E_pred),
                }
            )

    # saving in json format
    with open(
        os.path.join(os.path.dirname(__file__), "optimization_results.json"),
        "w",
    ) as f:
        json.dump(predictions, f, indent=2)
