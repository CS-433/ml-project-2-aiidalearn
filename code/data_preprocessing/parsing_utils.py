#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import json
import os
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_json(filepath: str):
    with open(filepath) as file:
        data = json.load(file)
    return data


# LOADING ALL ELEMENT KEYS
SSSP_PATH = os.path.join(
    os.path.dirname(__file__), "SSSP_1.1.2_PBE_efficiency.json",
)
SSSP_TABLE = load_json(SSSP_PATH)

PERIODIC_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "periodic_table_info.json",
)
PERIODIC_TABLE_INFO = load_json(PERIODIC_TABLE_PATH)
PERIODIC_TABLE_KEYS = list(PERIODIC_TABLE_INFO.keys())
PTC_COLNAMES = ["PTC" + str(n) for n in range(1, 19)]


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/"
)


class Encoding(Enum):
    ATOMIC = "atomic"
    COLUMN = "column"
    COLUMN_MASS = "column_mass"


def encode_structure(
    df: pd.DataFrame,
    elements_nbrs: Dict[str, int],
    encoding: Encoding = Encoding.ATOMIC,
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
        for element in PERIODIC_TABLE_KEYS:
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


def compute_delta_E(df: pd.DataFrame):
    idx_ref = 0
    for idx, row in df.loc[df["converged"]].iterrows():
        if (
            row["ecutwfc"] > df.loc[df["converged"], "ecutwfc"][idx_ref]
            or row["ecutrho"] > df.loc[df["converged"], "ecutrho"][idx_ref]
            or row["k_density"] < df.loc[df["converged"], "k_density"][idx_ref]
        ):
            idx_ref = idx

    ref_energy = df.loc[df["converged"], "total_energy"][idx_ref]
    print(f"Ref energy: {ref_energy} (found at index {idx_ref})")

    df = df.assign(delta_E=df["total_energy"] - ref_energy)

    return df


def parse_json(
    filepath: str,
    savepath: str,
    elements_nbrs: Dict[str, int],
    encodings: List[Encoding] = [Encoding.ATOMIC],
    inv_k_density: bool = False,
):
    with open(filepath) as file:
        data = json.load(file)

    raw_df = pd.DataFrame(data)
    raw_df = compute_delta_E(raw_df)
    rel_cols = [
        "ecutrho",
        "k_density",
        "ecutwfc",
        "converged",
        "accuracy",
        "delta_E",
    ]
    df = raw_df[rel_cols]

    if inv_k_density:
        df["k_density"] = df["k_density"].apply(lambda x: int(round(1 / x)))

    for encoding in encodings:
        encode_structure(df.copy(), elements_nbrs, encoding).to_csv(
            os.path.join(savepath, f"enc_{encoding.value}.csv")
        )


if __name__ == "__main__":
    p = Path(DATA_DIR)
    for struct_dir in p.iterdir():
        if not struct_dir.is_dir():
            continue
        for file in struct_dir.glob("data.json"):
            structure_name = os.path.basename(struct_dir)

            # Parsing the structure name to get the elements and their number
            elts = re.findall("[A-Z][^A-Z]*", structure_name)
            elements_nbrs = defaultdict(int)
            for elt in elts:
                atom_num = re.findall("\d+|\D+", elt)
                if len(atom_num) == 1:
                    elements_nbrs[elt] += 1
                else:
                    elements_nbrs[elt[0]] += int(elt[1])

            # Skip Lantanides
            isLant = False
            for elt in elements_nbrs.keys():
                if PERIODIC_TABLE_INFO[elt]["PTC"] == "Lant":
                    isLant = True
            if isLant:
                continue

            parse_json(
                filepath=file,
                savepath=struct_dir,
                elements_nbrs=elements_nbrs,
                encodings=list(Encoding),
                inv_k_density=True,
            )

            print("Done!\n")
