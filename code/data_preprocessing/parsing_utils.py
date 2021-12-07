#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from tools.utils import (
    PERIODIC_TABLE_INFO,
    PTC_COLNAMES,
    extract_structure_elements,
    load_json,
)

DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
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
    data = load_json(filepath)

    df = pd.DataFrame(data)
    df = compute_delta_E(df)

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
            elements_nbrs = extract_structure_elements(structure_name)

            parse_json(
                filepath=file,
                savepath=struct_dir,
                elements_nbrs=elements_nbrs,
                encodings=list(Encoding),
                inv_k_density=True,
            )

            print("Done!\n")
