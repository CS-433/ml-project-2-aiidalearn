#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import os
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from tools.utils import load_json

DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)
DATA_CSV = os.path.join(DATA_DIR, "data.csv")


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

    df = df.assign(delta_E=df["total_energy"] - ref_energy)

    return df, ref_energy


def check_parsing(data_dir: str, savepath: str) -> bool:
    console = Console()
    p = Path(data_dir)
    structure_names = set()
    for struct_dir in p.iterdir():
        if not struct_dir.is_dir():
            continue
        for _ in struct_dir.glob("data.json"):
            structure_name = os.path.basename(struct_dir)
            structure_names.add(structure_name)

    # check if savepath exists
    if not os.path.exists(savepath):
        console.print(f"{savepath} not found")
        return False

    df = pd.read_csv(savepath, na_filter=False)

    # check missing structures
    missing_structures = structure_names - set(df["structure"].unique())
    if len(missing_structures) != 0:
        console.print(
            Panel(
                "\n".join(str(elt) for elt in missing_structures),
                title="Structures not parsed",
                expand=False,
                style="bold yellow",
            )
        )

    # check potential ill-parsed structures (i.e. "NaN")
    structures_ill_parsed = set(df["structure"].unique()) - structure_names
    if len(structures_ill_parsed) != 0:
        console.print(
            Panel(
                "\n".join(
                    str(elt) + str(type(elt)) for elt in structures_ill_parsed
                ),
                title="Structures not parsed correctly",
                expand=False,
                style="bold red",
            )
        )

    if len(missing_structures) == 0 and len(structures_ill_parsed) == 0:
        console.print("All the structures are parsed")

    return True


def print_data_summary(df: pd.DataFrame = None):
    # print a summary on the collected data
    console = Console()
    if df is None:
        df = pd.read_csv(DATA_CSV)
    console.print(
        Panel(
            "[blue]"
            f"{df.structure.nunique()} unique structures\n"
            f"{df.shape[0]} total simulations\n"
            f"{df.loc[df['converged']].shape[0]} converged simulations",
            title="Data summary",
            expand=False,
            style="bold green",
        )
    )


def parse_all_data_json(
    data_dir: str, savepath: str, inv_k_density: bool = False,
) -> pd.DataFrame:
    list_df = []
    p = Path(data_dir)
    nb_folders = len(os.listdir(data_dir))
    console = Console()
    for struct_dir in track(
        p.iterdir(),
        description="Parsing structures...",
        total=nb_folders,
        console=console,
    ):
        if not struct_dir.is_dir():
            continue
        for file in struct_dir.glob("data.json"):
            structure_name = os.path.basename(struct_dir)

            df = pd.DataFrame(load_json(file))
            df, ref_energy = compute_delta_E(df)

            if inv_k_density:
                df["k_density"] = df["k_density"].apply(
                    lambda x: int(round(1 / x))
                )

            df["structure"] = structure_name
            df = df[["structure"] + df.columns.tolist()[:-1]]
            list_df.append(df)

            console.print(
                Panel(
                    "[blue]"
                    f"Reference energy: {ref_energy:.2f}\n"
                    f"Nb simulations: {len(df)}",
                    title=structure_name,
                    expand=False,
                )
            )

    res = pd.concat(list_df, ignore_index=True)

    with console.status("") as status:
        # save the data
        status.update(f"Saving parsed data to {savepath}...")
        res.to_csv(savepath)
        console.log(f"Data stored in {savepath}")

    return res


if __name__ == "__main__":
    if not (
        check_parsing(DATA_DIR, DATA_CSV)
        and input("Reparse data? (y/[n]) ") != "y"
    ):
        df = parse_all_data_json(DATA_DIR, DATA_CSV, inv_k_density=True)
        print_data_summary(df)
