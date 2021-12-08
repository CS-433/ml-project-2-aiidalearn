import json
import os
import re
from collections import defaultdict
from enum import Enum
from typing import Dict

import pandas as pd
from natsort import natsorted


def load_json(filepath: str):
    with open(filepath) as file:
        data = json.load(file)
    return data


class Encoding(Enum):
    ATOMIC = "atomic"
    COLUMN = "column"
    COLUMN_MASS = "column_mass"


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
            elements_nbrs[elt[0]] += int(elt[1])
    return dict(elements_nbrs)


def encode_structure(
    df: pd.DataFrame, elements_nbrs: Dict[str, int], encoding: Encoding,
):
    total_atoms = sum(list(elements_nbrs.values()))
    total_mass = 0.0
    for elt, nb_elt in elements_nbrs.items():
        elt_mass = PERIODIC_TABLE_INFO[elt]["mass"]
        total_mass += nb_elt * elt_mass

    if encoding in [Encoding.COLUMN, Encoding.COLUMN_MASS]:
        for colname in PTC_COLNAMES:
            df = df.assign(**{colname: 0.0})
    elif encoding == Encoding.ATOMIC:
        for element in PERIODIC_TABLE_INFO:
            df = df.assign(**{element: 0.0})

    for elt, nb_elt in elements_nbrs.items():
        if encoding == Encoding.COLUMN:
            ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
            ptc = ELEMENT_INFO["PTC"]
            print("-----Col encoding-----")
            print(elt, " -> ", ptc)
            df[ptc] = nb_elt / total_atoms
            print("--------------------")
        elif encoding == Encoding.COLUMN_MASS:
            ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
            ptc = ELEMENT_INFO["PTC"]
            elt_mass = ELEMENT_INFO["mass"]
            print("--Col mass encoding---")
            print(elt, " -> ", ptc)
            print(f"Mass of {elt}: {elt_mass:.3f}")
            df[ptc] += nb_elt * elt_mass / total_mass
            print("----------------------")
        elif encoding == Encoding.ATOMIC:
            df = df.assign(**{elt: nb_elt / total_atoms})

    return df


def encode_all_structures(
    df: pd.DataFrame, encoding: Encoding,
):
    if encoding in [Encoding.COLUMN, Encoding.COLUMN_MASS]:
        for colname in PTC_COLNAMES:
            df = df.assign(**{colname: 0.0})
    elif encoding == Encoding.ATOMIC:
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
            if encoding == Encoding.COLUMN:
                ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
                ptc = ELEMENT_INFO["PTC"]
                df.loc[df["structure"] == structure_name, ptc] = (
                    nb_elt / total_atoms
                )
            elif encoding == Encoding.COLUMN_MASS:
                ELEMENT_INFO = PERIODIC_TABLE_INFO[elt]
                ptc = ELEMENT_INFO["PTC"]
                elt_mass = ELEMENT_INFO["mass"]
                df.loc[df["structure"] == structure_name, ptc] = (
                    nb_elt * elt_mass / total_mass
                )
            elif encoding == Encoding.ATOMIC:
                df.loc[df["structure"] == structure_name, elt] = (
                    nb_elt / total_atoms
                )

    return df
