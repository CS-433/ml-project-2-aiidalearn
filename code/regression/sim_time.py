import os
import sys
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from rich.console import Console
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from tools.data_loader import TestSet, TestSplit, data_loader
from tools.save import save_as_baseline, save_datasets, save_models
from tools.train import evaluate_models, print_test_samples, train_models
from tools.utils import StructureEncoding, Target, check_xgboost_gpu

# Set Up
DATA_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "data/"
)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

MODELS_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "models/sim_time/"
)

BASELINES_DIR = os.path.join(
    str(Path(__file__).parent.parent.parent.absolute()), "baselines/sim_time/"
)


def instantiate_models(console: Console):
    with console.status("") as status:
        status.update("[bold blue]Initializing models...")

        linear_augmented_model = Pipeline(
            [
                ("scaler_init", StandardScaler()),
                ("poly_features", PolynomialFeatures(degree=2)),
                ("scaler_final", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )

        # best parameters from randomized grid search in corr. notebook
        rf_params = {
            "n_estimators": 230,
            "max_features": "auto",
            "max_depth": 40,
        }
        rf_model = RandomForestRegressor(random_state=0, **rf_params)

        # best parameters from randomized grid search in corr. notebook
        xgb_params = {
            "n_estimators": 500,
            "min_split_loss": 0.2,
            "max_depth": 12,
            "learning_rate": 0.1,
            "lambda": 1,
            "booster": "gbtree",
            "alpha": 0.5,
        }

        xgb_model = xgb.XGBRegressor(random_state=0, **xgb_params)

        lgbm_model = lgb.LGBMRegressor(
            n_estimators=5000, learning_rate=0.05, random_state=0
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
            # "LightGBM": lgbm_model,
        }


if __name__ == "__main__":
    console = Console(record=True)
    prompt_user = True

    encoding = StructureEncoding.ATOMIC
    target = Target.SIM_TIME
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
        "Random Forest": (models["Random Forest"], "random_forest_model.pkl"),
        "XGBoost": (models["XGBoost"], "xgboost_model.pkl"),
    }
    save_models(models_to_save, encoding, console, MODELS_DIR, prompt_user)

    save_datasets(
        X_train, y_train, test_sets, encoding, console, MODELS_DIR, prompt_user
    )
