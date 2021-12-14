import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from scipy.optimize import differential_evolution

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.transform import CustomLogTargetTransformer
from tools.utils import (
    PERIODIC_TABLE_INFO,
    PTC_COLNAMES,
    StructureEncoding,
    get_structure_encoding,
)

ROOT_DIR = str(Path(__file__).parent.parent.parent.absolute())

DELTA_E_MODELS_DIR = os.path.join(ROOT_DIR, "models/delta_E/")

LOG_DELTA_E_MODELS_DIR = os.path.join(ROOT_DIR, "models/log_delta_E/")

SIM_TIME_MODELS_DIR = os.path.join(ROOT_DIR, "models/sim_time/")

DATA_PATH = os.path.join(ROOT_DIR, "data/data.csv")


def load_models(
    delta_E_model_name=None,
    log_delta_E_model_name=None,
    sim_time_model_name=None,
):
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


def get_features_name(encoding):
    res = ["ecutrho", "k_density", "ecutwfc"]
    if encoding == StructureEncoding.ATOMIC:
        res += list(PERIODIC_TABLE_INFO.keys())
    elif encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        res += PTC_COLNAMES
    return res


def sanitize_input(x):
    return np.array([int(round(x_i)) for x_i in x])


def delta_E_prediction(x, model, structure_encoding, delta_E_features):
    input = np.concatenate([x, structure_encoding])
    input = pd.DataFrame(input.reshape(1, -1), columns=delta_E_features)
    return model.predict(input)[0]


def sim_time_prediction(x, model, structure_encoding, sim_time_features):
    input = np.concatenate([x, structure_encoding])
    input = pd.DataFrame(input.reshape(1, -1), columns=sim_time_features)
    return model.predict(input)[0]


def get_optimal_parameters(
    structure_name: str,
    max_delta_E: float,
    encoding_delta_E: StructureEncoding,
    encoding_sim_time: StructureEncoding,
    feature_bounds: dict,
    delta_E_model=None,
    sim_time_model=None,
    log_delta_E_model=None,
    verbose=False,
):
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
            return delta_E_prediction(
                sanitize_input(x),
                delta_E_model,
                structure_encoding_delta_E,
                get_features_name(encoding_delta_E),
            )

    if log_delta_E_model is not None:
        transformer = CustomLogTargetTransformer()
        y = pd.read_csv(DATA_PATH, na_filter=False)["delta_E"]
        transformer.fit(y)

        def delta_E_pred_func(x):
            return transformer.inverse_transform(
                delta_E_prediction(
                    sanitize_input(x),
                    log_delta_E_model,
                    structure_encoding_delta_E,
                    get_features_name(encoding_delta_E),
                )
            )

    def sim_time_pred_func(x):
        return sim_time_prediction(
            sanitize_input(x),
            sim_time_model,
            structure_encoding_sim_time,
            get_features_name(encoding_sim_time),
        )

    mu = 1e100

    def pen_func(x):
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


def get_feature_bounds(data_path):
    data = pd.read_csv(data_path, na_filter=False)
    return {
        "ecutrho": (data["ecutrho"].min(), data["ecutrho"].max()),
        "k_density": (data["k_density"].min(), data["k_density"].max()),
        "ecutwfc": (data["ecutwfc"].min(), data["ecutwfc"].max()),
    }


if __name__ == "__main__":
    console = Console()
    with console.status("Loading models..."):
        log_delta_E_model, sim_time_model = load_models(
            log_delta_E_model_name="random_forest_model.pkl",
            sim_time_model_name="random_forest_model.pkl",
        )

    structure_list = ["AgCl", "BaS"]
    max_delta_E_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
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
                encoding_delta_E=StructureEncoding.ATOMIC,
                encoding_sim_time=StructureEncoding.ATOMIC,
                feature_bounds=get_feature_bounds(DATA_PATH),
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
