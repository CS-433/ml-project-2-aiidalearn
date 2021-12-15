import os
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from rich.console import Console
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.save import save_as_baseline, save_datasets, save_models
from tools.train import evaluate_models, print_test_samples, train_models
from tools.utils import (
    StructureEncoding,
    check_xgboost_gpu,
    encode_all_structures,
)

# Define global variables
DATA_DIR = os.path.join(ROOT_DIR, "data/")

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

MODELS_DIR = os.path.join(ROOT_DIR, "models/ref_energy/")

BASELINES_DIR = os.path.join(ROOT_DIR, "baselines/ref_energy/")


def data_loader_ref_energy(
    encoding: StructureEncoding, console: Console, data_path=DATA_PATH
):
    with console.status("[bold blue]Loading data..."):
        df = pd.read_csv(data_path, na_filter=False)
        df = encode_all_structures(df, encoding)
        X = df.drop(columns=["structure", "total_energy"])
        y = df["total_energy"].squeeze()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        test_sets = [("Structure gen.", X_test, y_test)]

    return X_train, y_train, test_sets


def instantiate_models(console: Console):
    with console.status("") as status:
        status.update("[bold blue]Initializing models...")

        rf_model = RandomForestRegressor(random_state=0, bootstrap=True)

        xgb_model = xgb.XGBRegressor(
            max_depth=7, n_estimators=400, learning_rate=1.0, random_state=0
        )

        status.update("[bold blue]Checking GPU usability for XGBoost...")
        if check_xgboost_gpu():
            xgb_model.set_params(tree_method="gpu_hist")
            console.print("[italic bright_black]Using GPU for XGBoost")
        else:
            console.print("[italic bright_black]Using CPU for XGBoost")

        return {
            "Dummy": DummyRegressor(),
            "Linear": LinearRegression(),
            "Random Forest": rf_model,
            "XGBoost": xgb_model,
        }


if __name__ == "__main__":
    encoding = StructureEncoding.ATOMIC
    console = Console(record=True)
    prompt_user = True

    X_train, y_train, test_sets = data_loader_ref_energy(encoding, console)
    models = instantiate_models(console)
    train_models(models, X_train, y_train, console)
    evaluate_models(models, X_train, y_train, test_sets, console)
    print_test_samples(models, test_sets, console)
    save_as_baseline(encoding, console, BASELINES_DIR, prompt_user)

    models_to_save = {
        "Random Forest": (models["Random Forest"], "random_forest_model.pkl"),
        "XGBoost": (models["XGBoost"], "xgboost_model.pkl"),
    }
    save_models(models_to_save, encoding, console, MODELS_DIR, prompt_user)

    save_datasets(
        X_train, y_train, test_sets, encoding, console, MODELS_DIR, prompt_user
    )
