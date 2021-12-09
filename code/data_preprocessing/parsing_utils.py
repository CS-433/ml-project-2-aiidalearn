#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from tools.utils import (
    Encoding,
    encode_structure,
    extract_structure_elements,
    load_json,
)

DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)


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


def parse_all_json(
    data_dir: str, savepath: str, inv_k_density: bool = False,
) -> pd.DataFrame:
    list_df = []
    p = Path(data_dir)
    console = Console()
    p = Path(DATA_DIR)
    with console.status("[bold blue] Parsing structures...") as status:
        for struct_dir in p.iterdir():
            if not struct_dir.is_dir():
                continue
            for file in struct_dir.glob("data.json"):
                structure_name = os.path.basename(struct_dir)

                data = load_json(file)

                df = pd.DataFrame(data)
                df = compute_delta_E(df)

                if inv_k_density:
                    df["k_density"] = df["k_density"].apply(
                        lambda x: int(round(1 / x))
                    )

                df["structure"] = structure_name
                df = df[["structure"] + df.columns.tolist()[:-1]]
                list_df.append(df)

                console.print(f'[blue]Parsed {structure_name}')

    res = pd.concat(list_df, ignore_index=True)
    res.to_csv(os.path.join(savepath, "data.csv"))


if __name__ == "__main__":
    parse_all_json(
        DATA_DIR, DATA_DIR, inv_k_density=True,
    )
    exit()

    # Code below parses JSON individually and with specific encodings
    console = Console()
    p = Path(DATA_DIR)
    with console.status("[bold blue] Parsing structures...") as status:
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

                console.print(f'[blue]Parsed {structure_name}')
