import json
import os
import re
from collections import defaultdict
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from natsort import natsorted


def load_json(filepath: str):
    with open(filepath) as file:
        data = json.load(file)
    return data


class StructureEncoding(Enum):
    ATOMIC = "atomic"
    COLUMN = "column"
    COLUMN_MASS = "column_mass"
    VALENCE_CONFIG = "valence_configuration"


class Target(Enum):
    SIM_TIME = "time"
    DELTA_E = "delta_E"
    CONVERGED = "converged"


PERIODIC_TABLE_INFO = load_json(
    os.path.join(os.path.dirname(__file__), "periodic_table_info.json",)
)
PTC_COLNAMES = natsorted(
    list(set(PERIODIC_TABLE_INFO[elt]["PTC"] for elt in PERIODIC_TABLE_INFO))
)


def extract_structure_elements(structure_name: str) -> Dict[str, int]:
    """
    Extract the structure elements from the structure name.

    Examples:

    AgCl -> {Ag: 1, Cl: 1}

    Rb2O2 -> {Rb: 2, O: 2}
    """
    elts = re.findall("[A-Z][^A-Z]*", structure_name)
    elements_nbrs = defaultdict(int)
    for elt in elts:
        atom_num = re.findall(r"\d+|\D+", elt)
        if len(atom_num) == 1:
            elements_nbrs[elt] += 1
        else:
            elements_nbrs[atom_num[0]] += int(atom_num[1])
    return dict(elements_nbrs)


def get_structure_encoding(
    structure_name: str, encoding: StructureEncoding
) -> np.ndarray:
    periodic_elt_list = list(PERIODIC_TABLE_INFO.keys())
    if encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        res = np.zeros(len(PTC_COLNAMES) + 1)
    elif encoding == StructureEncoding.ATOMIC:
        res = np.zeros(len(periodic_elt_list) + 1)

    elements_nbrs = extract_structure_elements(structure_name)
    total_atoms = sum(list(elements_nbrs.values()))
    total_mass = 0.0
    for elt, nb_elt in elements_nbrs.items():
        elt_mass = PERIODIC_TABLE_INFO[elt]["mass"]
        total_mass += nb_elt * elt_mass

    for elt, nb_elt in elements_nbrs.items():
        if encoding == StructureEncoding.COLUMN:
            ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
            ptc = ELEMENT_INFO["PTC"]
            res[PTC_COLNAMES.index(ptc)] += nb_elt / total_atoms
        elif encoding == StructureEncoding.COLUMN_MASS:
            ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
            ptc = ELEMENT_INFO["PTC"]
            elt_mass = ELEMENT_INFO["mass"]
            res[PTC_COLNAMES.index(ptc)] += nb_elt * elt_mass / total_mass
        elif encoding == StructureEncoding.ATOMIC:
            res[periodic_elt_list.index(elt)] += nb_elt / total_atoms
        elif encoding == StructureEncoding.VALENCE_CONFIG:
            ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
            valence_band_str = ELEMENT_INFO["valence_band"]
            valence_band = parse_valence_band(valence_band_str)
            blocks = ["s", "p", "d", "f"]
            for idx_block, block in enumerate(blocks):
                res[idx_block] += (
                    valence_band[block] / valence_band["outermost"]
                )
    res[-1] = total_atoms

    return res

    # # code below would be better conceptually, but several orders of magnitude slower in practice
    # df = pd.DataFrame({"structure": [structure_name]})
    # df = encode_all_structures(df, encoding)
    # return df.iloc[0, 1:].values


def encode_all_structures(
    df: pd.DataFrame, encoding: StructureEncoding,
) -> pd.DataFrame:
    if encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        for colname in PTC_COLNAMES:
            df = df.assign(**{colname: 0.0})
    elif encoding == StructureEncoding.ATOMIC:
        for element in PERIODIC_TABLE_INFO:
            df = df.assign(**{element: 0.0})

    elif encoding == StructureEncoding.VALENCE_CONFIG:
        blocks = ["s", "p", "d", "f"]
        for block in blocks:
            df = df.assign(**{block: 0.0})

    df = df.assign(**{"total_atoms": 0.0})

    for structure_name in df["structure"].unique():
        elements_nbrs = extract_structure_elements(structure_name)
        total_atoms = sum(list(elements_nbrs.values()))
        df.loc[df["structure"] == structure_name, "total_atoms"] += total_atoms
        total_mass = 0.0
        for elt, nb_elt in elements_nbrs.items():
            elt_mass = PERIODIC_TABLE_INFO[elt]["mass"]
            total_mass += nb_elt * elt_mass

        for elt, nb_elt in elements_nbrs.items():
            if encoding == StructureEncoding.COLUMN:
                ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
                ptc = ELEMENT_INFO["PTC"]
                df.loc[df["structure"] == structure_name, ptc] += (
                    nb_elt / total_atoms
                )
            elif encoding == StructureEncoding.COLUMN_MASS:
                ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
                ptc = ELEMENT_INFO["PTC"]
                elt_mass = ELEMENT_INFO["mass"]
                df.loc[df["structure"] == structure_name, ptc] += (
                    nb_elt * elt_mass / total_mass
                )
            elif encoding == StructureEncoding.ATOMIC:
                df.loc[df["structure"] == structure_name, elt] = (
                    nb_elt / total_atoms
                )
            elif encoding == StructureEncoding.VALENCE_CONFIG:
                ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
                valence_band_str = ELEMENT_INFO["valence_band"]
                valence_band = parse_valence_band(valence_band_str)
                blocks = ["s", "p", "d", "f"]
                for block in blocks:
                    df.loc[df["structure"] == structure_name, block] += (
                        valence_band[block] / valence_band["outermost"]
                    )

    return df


def parse_valence_band(valence_band_str: str) -> Dict[str, float]:
    orbitals = valence_band_str.split("-")
    valence_band = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0, "outermost": 0.0}
    for orbital_str in orbitals:
        key = orbital_str[1]
        value = int(orbital_str.split("^")[-1])
        valence_band[key] += value

    outermost_orbital = orbitals[-1]
    valence_band["outermost"] = int(outermost_orbital[0])
    return valence_band


def custom_mape(y_true: np.array, y_pred: np.array, shift=False) -> float:
    if shift:
        miny2 = sorted(set(np.array(y_true).flatten()))[:2]
        bias = -miny2[0] + (miny2[1] - miny2[0]) / 10
        y_true = y_true + bias
        y_pred = y_pred + bias
    return np.mean(
        np.divide(
            np.abs(y_true - y_pred),
            np.abs(y_true),
            where=y_true != 0,
            out=np.zeros_like(y_true),
        )
    )


def absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    epsilon = np.finfo(np.float64).eps
    return np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)


def percentile_absolute_percentage_error(
    y_true: np.array, y_pred: np.array, percentile=50
) -> float:
    ape = absolute_percentage_error(y_true, y_pred)
    return np.percentile(ape, percentile)


def check_xgboost_gpu() -> bool:
    try:
        xgb_model = xgb.XGBRegressor(tree_method="gpu_hist")
        xgb_model.fit(np.array([[1, 2, 3]]), np.array([[1]]))
        return True
    except:
        return False
