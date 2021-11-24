#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import pandas as pd
import json
import os
import requests

# LOADING ALL ELEMENT KEYS
URL_TABLE = requests.get(
    "https://archive.materialscloud.org/record/file?record_id=862&filename=SSSP_1.1.2_PBE_efficiency.json&file_id=a5642f40-74af-4073-8dfd-706d2c7fccc2"
)
TEXT_TABLE = URL_TABLE.text
SSSP_TABLE = json.loads(TEXT_TABLE)
PERIODIC_TABLE_KEYS = list(SSSP_TABLE.keys())


def encode_structure(df, elements_nbrs):
    total_atoms = sum(list(elements_nbrs.values()))

    elements = list(elements_nbrs.keys())

    for element in elements:
        df[element] = elements_nbrs[element] / total_atoms

    return df


def compute_delta_E(df):
    converged_rows = df["converged"] == True
    idx_ref = 0
    for idx, row in df.loc[converged_rows].iterrows():
        if (
            row["ecutrho"] > df.loc[converged_rows, "ecutrho"][idx_ref]
            or row["k_density"] < df.loc[converged_rows, "k_density"][idx_ref]
            or row["ecutwfc"] > df.loc[converged_rows, "ecutwfc"][idx_ref]
        ):
            idx_ref = idx

    ref_energy = df.loc[converged_rows, "total_energy"][idx_ref]
    print(f"Ref energy: {ref_energy} (found at index {idx_ref})")

    df["delta_E"] = df["total_energy"] - ref_energy

    return df


def parse_json(filepath, savepath, elements_nbrs):
    with open(filepath) as file:
        data = json.load(file)

    raw_df = pd.DataFrame(data)
    rel_cols = [
        "ecutrho",
        "k_density",
        "ecutwfc",
        "converged",
        "accuracy",
        "total_energy",
    ]
    df = raw_df[rel_cols]

    df = compute_delta_E(df)

    for element in PERIODIC_TABLE_KEYS:
        df[element] = 0.0

    df = encode_structure(df, elements_nbrs)

    df.to_csv(savepath)


if __name__ == "__main__":
    struct_name_list = ["Ge-1_Se-1", "Ge-1_Te-1"]
    for structure_name in struct_name_list:
        elements_nbrs = {
            elt.split("-")[0]: int(elt.split("-")[1])
            for elt in structure_name.split("_")
        }

        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/" + structure_name + ".json",
        )
        savepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/" + structure_name + ".csv",
        )

        parse_json(filepath, savepath, elements_nbrs)

        print(structure_name + " done!")
