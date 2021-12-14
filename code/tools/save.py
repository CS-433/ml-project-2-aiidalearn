import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.utils import StructureEncoding


def save_as_baseline(
    encoding: StructureEncoding,
    console: Console,
    baseline_dir,
    prompt_user=True,
):
    if (
        not prompt_user
        or input("Save results as baseline? (html only) (y/[n]) ") == "y"
    ):
        Path(baseline_dir).mkdir(parents=True, exist_ok=True)
        filename = f"results_{encoding.value}.html"
        results_file = os.path.join(baseline_dir, filename)
        console.save_html(results_file)
        console.log(f"[green]Results stored in {results_file}")


def save_models(
    models,
    encoding: StructureEncoding,
    console: Console,
    models_dir,
    prompt_user=True,
):
    if not prompt_user or input("Save models? (y/[n]) ") == "y":
        models_save_dir = os.path.join(models_dir, encoding.value)
        Path(models_save_dir).mkdir(parents=True, exist_ok=True)
        results_file = os.path.join(models_save_dir, "results.html")
        console.save_html(results_file)
        console.print(f"Results stored in {results_file}")
        with console.status("[bold green]Saving models..."):
            for model_name, (model, filename) in models.items():
                modelpath = os.path.join(models_save_dir, filename)
                with open(modelpath, "wb") as file:
                    pickle.dump(model, file)
                console.log(
                    f"[green]Saved {model_name} to {modelpath}[/green]"
                )


def save_datasets(
    X_train,
    y_train,
    test_sets,
    encoding: StructureEncoding,
    console: Console,
    models_dir,
    prompt_user=True,
):
    if not prompt_user or input("Save datasets? (y/[n]) ") == "y":
        datasets_save_dir = os.path.join(
            models_dir, f"{encoding.value}/datasets"
        )
        Path(datasets_save_dir).mkdir(parents=True, exist_ok=True)
        with console.status("[bold green]Saving datasets..."):
            for name, X, y in [("train", X_train, y_train)] + test_sets:
                X.to_csv(os.path.join(datasets_save_dir, f"X_{name}.csv"))
                y.to_csv(os.path.join(datasets_save_dir, f"y_{name}.csv"))
                console.log(
                    f"[green]Saved {name} to {datasets_save_dir}[/green]"
                )


def load_saved_datasets(
    encoding: StructureEncoding,
    console: Console,
    models_dir,
    prompt_user=True,
):
    if not prompt_user or input("Load datasets? (y/[n]) ") == "y":
        test_sets = []
        datasets_load_dir = os.path.join(
            models_dir, f"{encoding.value}/datasets"
        )
        p = Path(datasets_load_dir)
        with console.status("[bold green]Loading datasets..."):
            for file in p.glob("X_*.csv"):
                name = file.name[2:-4]
                X_file = file
                y_file = os.path.join(str(p.absolute()), f"y_{name}.csv")

                X = pd.read_csv(X_file, index_col=0, na_filter=False)
                y = pd.read_csv(y_file, index_col=0, na_filter=False)
                console.log(
                    f"[green]Loaded {name} from {datasets_load_dir}[/green]"
                )
                if name == "train":
                    X_train, y_train = X, y
                else:
                    test_sets.append((name, X, y))
        return X_train, y_train, test_sets
