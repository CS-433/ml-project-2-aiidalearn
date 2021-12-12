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
    Extracts the structure elements from the structure name.
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


def get_structure_encoding(structure_name, encoding) -> np.ndarray:
    periodic_elt_list = list(PERIODIC_TABLE_INFO.keys())
    if encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        res = np.zeros(len(PTC_COLNAMES))
    elif encoding == StructureEncoding.ATOMIC:
        res = np.zeros(len(periodic_elt_list))

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

    return res


def encode_structure(
    df: pd.DataFrame,
    elements_nbrs: Dict[str, int],
    encoding: StructureEncoding,
):
    total_atoms = sum(list(elements_nbrs.values()))
    total_mass = 0.0
    for elt, nb_elt in elements_nbrs.items():
        elt_mass = PERIODIC_TABLE_INFO[elt]["mass"]
        total_mass += nb_elt * elt_mass

    if encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        for colname in PTC_COLNAMES:
            df = df.assign(**{colname: 0.0})
    elif encoding == StructureEncoding.ATOMIC:
        for element in PERIODIC_TABLE_INFO:
            df = df.assign(**{element: 0.0})

    for elt, nb_elt in elements_nbrs.items():
        if encoding == StructureEncoding.COLUMN:
            ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
            ptc = ELEMENT_INFO["PTC"]
            print("-----Col encoding-----")
            print(elt, " -> ", ptc)
            df[ptc] += nb_elt / total_atoms
            print("--------------------")
        elif encoding == StructureEncoding.COLUMN_MASS:
            ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
            ptc = ELEMENT_INFO["PTC"]
            elt_mass = ELEMENT_INFO["mass"]
            print("--Col mass encoding---")
            print(elt, " -> ", ptc)
            print(f"Mass of {elt}: {elt_mass:.3f}")
            df[ptc] += nb_elt * elt_mass / total_mass
            print("----------------------")
        elif encoding == StructureEncoding.ATOMIC:
            df = df.assign(**{elt: nb_elt / total_atoms})

    return df


def encode_all_structures(
    df: pd.DataFrame, encoding: StructureEncoding,
):
    if encoding in [StructureEncoding.COLUMN, StructureEncoding.COLUMN_MASS]:
        for colname in PTC_COLNAMES:
            df = df.assign(**{colname: 0.0})
    elif encoding == StructureEncoding.ATOMIC:
        for element in PERIODIC_TABLE_INFO:
            df = df.assign(**{element: 0.0})

    for structure_name in df["structure"].unique():
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

    return df


def custom_mape(y_true, y_pred, shift=False):
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


def check_xgboost_gpu():
    try:
        xgb_model = xgb.XGBRegressor(tree_method="gpu_hist")
        xgb_model.fit(np.array([[1, 2, 3]]), np.array([[1]]))
        return True
    except:
        return False
