#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import json
import os
from enum import Enum
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


def encode_structure(
    df: pd.DataFrame,
    elements_nbrs: Dict[str, int],
    encoding: Encoding = Encoding.ATOMIC,
):
    total_atoms = sum(list(elements_nbrs.values()))

    if encoding == Encoding.COLUMN:
        for colname in PTC_COLNAMES:
            df = df.assign(**{colname: 0.0})
    elif encoding == Encoding.ATOMIC:
        for element in PERIODIC_TABLE_KEYS:
            df = df.assign(**{element: 0.0})

    for elt, nb_elt in elements_nbrs.items():
        if encoding == Encoding.COLUMN:
            ptc = PERIODIC_TABLE_INFO[elt]
            print(elt, " -> ", ptc)
            df[ptc] = nb_elt / total_atoms
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
    name: str,
    elements_nbrs: Dict[str, int],
    encodings: List[Encoding] = [Encoding.ATOMIC],
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

    for encoding in encodings:
        encode_structure(df.copy(), elements_nbrs, encoding).to_csv(
            os.path.join(savepath, f"{name}_{encoding.value}.csv")
        )


if __name__ == "__main__":
    for filename in os.listdir(DATA_DIR):
        ext = ".json"
        if filename.endswith(ext):
            structure_name = filename[: -len(ext)]
            print(f"Parsing {structure_name}...")

            parse_json(
                filepath=os.path.join(DATA_DIR, filename),
                savepath=DATA_DIR,
                name=structure_name,
                elements_nbrs={
                    elt.split("-")[0]: int(elt.split("-")[1])
                    for elt in structure_name.split("_")
                },
                encodings=list(Encoding),
            )

            print("Done!\n")
