import os
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rich.console import Console
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.data_loader import TestSet, TestSplit, data_loader
from tools.save import save_as_baseline, save_datasets, save_models
from tools.train import evaluate_models, print_test_samples, train_models, print_problematic_samples
from tools.utils import StructureEncoding, Target, check_xgboost_gpu

# Define global variables
DATA_DIR = os.path.join(ROOT_DIR, "data/")

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

MODELS_DIR = os.path.join(ROOT_DIR, "models/delta_E/")

BASELINES_DIR = os.path.join(ROOT_DIR, "baselines/delta_E/")


def instantiate_models(console: Console):
    with console.status("") as status:
        status.update("[bold blue]Initializing models...")

        # functions such that f(x) != 0 and f(+inf) = 0
        functions_set_1 = [
            lambda x: np.exp(-x),
            lambda x: 1 / (1 + x),
            lambda x: 1 / (1 + x) ** 2,
            lambda x: np.cos(x) * np.exp(-x),
        ]

        # functions such that f(x) = 0 and f(+inf) = 0
        functions_set_2 = [
            lambda x: x * np.exp(-x),
            lambda x: x / (1 + x) ** 2,
            lambda x: x / (1 + x) ** 3,
            lambda x: np.sin(x) * np.exp(-x),
        ]

        linear_augmented_model = Pipeline(
            [
                ("scaler_init", StandardScaler()),
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                f"fun_{j}",
                                FunctionTransformer(lambda X: f(X[:, :3])),
                            )
                            for j, f in enumerate(
                                functions_set_1 + functions_set_2
                            )
                        ]
                        + [
                            (
                                f"fun_{j}_col_{col}_1",
                                FunctionTransformer(
                                    lambda X: f(X[:, :3] * X[:, i][:, None])
                                ),
                            )
                            for j, f in enumerate(functions_set_1)
                            for i, col in enumerate(
                                ["ecutrho", "kpoints", "ecutwfc"]
                            )
                        ]
                        + [
                            (
                                f"fun_{j}_col_{col}_2",
                                FunctionTransformer(
                                    lambda X: f(X[:, 3:] * X[:, i][:, None])
                                ),
                            )
                            for j, f in enumerate(functions_set_2)
                            for i, col in enumerate(
                                ["ecutrho", "kpoints", "ecutwfc"]
                            )
                        ]
                    ),
                ),
                ("scaler_final", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )

        rf_model = RandomForestRegressor(random_state=0)

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
            # "Augmented Linear": linear_augmented_model,
            "Random Forest": rf_model,
             "XGBoost": xgb_model,
        }


if __name__ == "__main__":
    console = Console(record=True)
    prompt_user = False

    encodings = [StructureEncoding.ATOMIC]
    # encodings = list(StructureEncoding)
    for encoding in encodings:
        console.log(f"[bold green]Started pipeline for {encoding.value} encoding")
        target = Target.DELTA_E
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
            console=console,
            remove_ref_rows=True,
        )

        models = instantiate_models(console)
        train_models(models, X_train, y_train, console)
        evaluate_models(models, X_train, y_train, test_sets, console)
        print_test_samples(models, test_sets, console)
        save_as_baseline(encoding, console, BASELINES_DIR, prompt_user)

        models_to_save = {
            "Random Forest": (
                models["Random Forest"],
                "random_forest_model.pkl",
            ),
            "XGBoost": (models["XGBoost"], "xgboost_model.pkl"),
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

