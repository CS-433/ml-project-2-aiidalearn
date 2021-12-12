#!/usr/bin/env python
# coding: utf-8


import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

sys.path.append(os.path.dirname(os.getcwd()))
from tools.utils import (
    PERIODIC_TABLE_INFO,
    PTC_COLNAMES,
    StructureEncoding,
    encode_all_structures,
)

DELTA_E_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "models/delta_E/"
)

SIM_TIME_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "models/sim_time/"
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/")


# # Loading the models
delta_E_model_name = "random_forest_model.pkl"
sim_time_model_name = "random_forest_model.pkl"

with open(os.path.join(DELTA_E_MODELS_DIR, delta_E_model_name), "rb") as file:
    delta_E_model = pickle.load(file)

with open(
    os.path.join(SIM_TIME_MODELS_DIR, sim_time_model_name), "rb"
) as file:
    sim_time_model = pickle.load(file)


# # Load data
encoding = StructureEncoding.ATOMIC

df = pd.read_csv(
    os.path.join(DATA_DIR, "data.csv"), index_col=0, na_filter=False
)
df = encode_all_structures(df, encoding)


# ## Transform to ∆E input
cols_raw = list(df.columns)
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
cols_drop = cols_trash + cols_independent

cols_dependent = cols_raw.copy()
for element in cols_drop:
    cols_dependent.remove(element)


X_raw = df[cols_dependent][df["converged"]]
y_raw = np.abs(df[cols_independent][df["converged"]]).squeeze()


# ## Transform to sim time input
cols_raw = list(df.columns)
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
cols_drop = cols_trash + cols_independent

cols_dependent = cols_raw.copy()
for element in cols_drop:
    cols_dependent.remove(element)
cols_dependent

X_raw_sim_time = df[cols_dependent][df["converged"]]
y_raw_sim_time = np.abs(df[cols_independent][df["converged"]]).squeeze()


# # Setting up optimizer
def sanitize_input(x):
    return np.array([int(round(x_i)) for x_i in x])


def delta_E_prediction(x, model, structure_encoding):
    input = np.concatenate([x, structure_encoding])
    input = pd.DataFrame(input.reshape(1, -1), columns=X_raw.columns)
    return model.predict(input)[0]


def sim_time_prediction(x, model, structure_encoding):
    input = np.concatenate([x, structure_encoding])
    input = pd.DataFrame(input.reshape(1, -1), columns=X_raw_sim_time.columns)
    return model.predict(input)[0]


def get_structure_encoding(structure_key):
    structure_data = df[df["structure"] == structure_key]

    # then find the index of the first encoding column
    if encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        enc_first_idx = next(
            (
                i
                for i, col in enumerate(structure_data.columns)
                if col in PTC_COLNAMES
            ),
            None,
        )
    elif encoding == StructureEncoding.ATOMIC:
        enc_first_idx = next(
            (
                i
                for i, col in enumerate(structure_data.columns)
                if col in PERIODIC_TABLE_INFO
            ),
            None,
        )

    # finally get the encoding
    structure_encoding = np.array(
        structure_data.iloc[0, enc_first_idx:].values, dtype=float
    )

    return structure_encoding


structure_key = "AgCl"
structure_encoding = get_structure_encoding(structure_key)


max_delta_E = 1e-3

delta_E_pred_func = lambda x: delta_E_prediction(
    sanitize_input(x), delta_E_model, structure_encoding
)
complexity_pred_func = lambda x: sim_time_prediction(
    sanitize_input(x), sim_time_model, structure_encoding
)

mu = 1e100


def pen_func(x):
    return (
        complexity_pred_func(sanitize_input(x))
        + mu
        * max(delta_E_pred_func(sanitize_input(x)) - max_delta_E, 0)
        / max_delta_E
    )


res = differential_evolution(
    pen_func,
    bounds=[
        (X_raw["ecutrho"].min(), X_raw["ecutrho"].max()),
        (X_raw["k_density"].min(), X_raw["k_density"].max()),
        (X_raw["ecutwfc"].min(), X_raw["ecutwfc"].max()),
    ],
    seed=0,
)

x_f = sanitize_input(res.x)

print("=======OPTIMIZATION PARAMETERS=======")
print(f"∆E_max: {max_delta_E:.3E}")
print(f"Structure: {structure_key}")
print("========OPTIMIZATION RESULTS=========")
print(res)
print("=============PREDICTIONS=============")
print(
    f"Parameters: {x_f}\tSim. time: {complexity_pred_func(x_f)}\t∆E: {delta_E_pred_func(x_f):.3E}"
)
