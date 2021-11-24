#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import json
import os
from typing import Dict

import pandas as pd
import requests

# LOADING ALL ELEMENT KEYS
URL_TABLE = requests.get(
    "https://archive.materialscloud.org/record/file?record_id=862&filename=SSSP_1.1.2_PBE_efficiency.json&file_id=a5642f40-74af-4073-8dfd-706d2c7fccc2"
)
TEXT_TABLE = URL_TABLE.text
SSSP_TABLE = json.loads(TEXT_TABLE)
PERIODIC_TABLE_KEYS = list(SSSP_TABLE.keys())
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/"
)


def encode_structure(df: pd.DataFrame, elements_nbrs: Dict[str, int]):
    total_atoms = sum(list(elements_nbrs.values()))

    for elt, nb_elt in elements_nbrs.items():
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


def parse_json(filepath: str, savepath: str, elements_nbrs: Dict[str, int]):
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

    for element in PERIODIC_TABLE_KEYS:
        df = df.assign(**{element: 0.0})

    df = encode_structure(df, elements_nbrs)

    df.to_csv(savepath)


if __name__ == "__main__":
    for filename in os.listdir(DATA_DIR):
        ext = ".json"
        if filename.endswith(ext):
            structure_name = filename[: -len(ext)]
            print(f"Parsing {structure_name}...")

            parse_json(
                filepath=os.path.join(DATA_DIR, filename),
                savepath=os.path.join(DATA_DIR, structure_name + ".csv"),
                elements_nbrs={
                    elt.split("-")[0]: int(elt.split("-")[1])
                    for elt in structure_name.split("_")
                },
            )

            print("Done!\n")
