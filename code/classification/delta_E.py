import os
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rich.console import Console
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.data_loader import TestSet, TestSplit, data_loader
from tools.save import save_as_baseline, save_models, save_datasets
from tools.train import (
    evaluate_classifiers,
    print_test_samples,
    cv_classifiers,
    train_models,
)
from tools.utils import StructureEncoding, Target, check_xgboost_gpu
from tools.transform import TargetMagnitudeTransformer


# Define global variables
DATA_DIR = os.path.join(ROOT_DIR, "data/")

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

MODELS_DIR = os.path.join(ROOT_DIR, "models/delta_E_magnitude/")

BASELINES_DIR = os.path.join(ROOT_DIR, "baselines/delta_E_magnitude/")


def instantiate_models(console: Console):
    with console.status("") as status:
        status.update("[bold blue]Initializing models...")

        rf_params = {
            "n_estimators": 218,
            "max_features": "sqrt",
            "max_depth": 205,
            "random_state": 0,
        }
        rf_model = RandomForestClassifier(**rf_params)
        console.log(f"[green] Initialized {rf_model}")

        xgb_model = xgb.XGBClassifier(random_state=0)
        console.log(f"[green] Initialized {xgb_model}")

        status.update("[bold blue]Checking GPU usability for XGBoost...")
        if check_xgboost_gpu():
            xgb_model.set_params(tree_method="gpu_hist")
            console.print("[italic bright_black]Using GPU for XGBoost")
        else:
            console.print("[italic bright_black]Using CPU for XGBoost")

        return {
            # "Dummy" : DummyClassifier(),
            "Random Forest": rf_model,
            # "XGBoost": xgb_model,
        }


if __name__ == "__main__":
    console = Console(record=True)
    prompt_user = False

    encodings = [StructureEncoding.ATOMIC]
    # encodings = list(StructureEncoding)
    for encoding in encodings:
        console.log(
            f"[bold green]Started training pipeline for {encoding.value} encoding"
        )
        target = Target.DELTA_E
        target_transformer = TargetMagnitudeTransformer()
        test_sets_cfg = [
            TestSet("Parameter gen.", size=0.1, split=TestSplit.ROW),
            TestSet("Structure gen.", size=0.1, split=TestSplit.STRUCTURE),
        ]

        # Data Loading
        X_train, y_train, test_sets = data_loader(
            target=target,
            encoding=encoding,
            data_path=DATA_PATH,
            test_sets_cfg=test_sets_cfg,
            target_transformer=target_transformer,
            console=console,
            remove_ref_rows=True,
        )
        models = instantiate_models(console)
        train_models(models, X_train, y_train, console)
        evaluate_classifiers(models, X_train, y_train, test_sets, console)
        # cv_classifiers(models, X_train, y_train, console, shuffle=False)
        # cv_classifiers(models, X_train, y_train, console, shuffle=True)

        # print_test_samples(models, test_sets, console)
        save_as_baseline(encoding, console, BASELINES_DIR, prompt_user)

        models_to_save = {
            "Random Forest": (
                models["Random Forest"],
                "random_forest_model.pkl",
            ),
            # "XGBoost": (models["XGBoost"], "xgboost_model.pkl"),
        }
        save_models(models_to_save, encoding, console, MODELS_DIR, prompt_user)

        save_datasets(
            X_train,
            y_train,
            test_sets,
            encoding,
            console,
            MODELS_DIR,
            prompt_user,
        )
