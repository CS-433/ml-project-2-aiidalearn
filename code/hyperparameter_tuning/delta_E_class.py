import os
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.data_loader import TestSet, TestSplit, data_loader
from tools.save import save_params
from tools.train import evaluate_classifiers, train_classifiers
from tools.transform import TargetMagnitudeTransformer
from tools.utils import StructureEncoding, Target

# Define global variables
DATA_DIR = os.path.join(ROOT_DIR, "data/")

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

PARAMS_DIR = os.path.join(ROOT_DIR, "hyperparameter_tuning/delta_E_magnitude/")


if __name__ == "__main__":
    console = Console(record=True)
    prompt_user = False

    encodings = [StructureEncoding.ATOMIC]
    # encodings = list(StructureEncoding)
    for encoding in encodings:
        console.log(
            f"[bold green]Started pipeline for {encoding.value} encoding"
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

        base_model = RandomForestClassifier(random_state=0)

        # Number of trees in random forest
        n_estimators = [
            int(x) for x in np.linspace(start=10, stop=1000, num=20)
        ]
        # Number of features to consider at every split
        max_features = ["auto", "sqrt"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(50, 250, num=10)]
        max_depth.append(None)
        param_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
        }

        model_random = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=15,
            cv=2,
            verbose=100,
        )
        console.log("Performing randomized grid search")
        model_random.fit(X_train, y_train)
        console.log("Finished randomized grid search")

        best_model = RandomForestClassifier(**model_random.best_params_)

        table = Table(title="Best Parameters")
        table.add_column("n_estimators", justify="center", style="white")
        table.add_column("max_features", justify="center", style="white")
        table.add_column("max_depth", justify="center", style="white")
        table.add_row(
            f"{model_random.best_params_['n_estimators']}",
            f"{model_random.best_params_['max_features']}",
            f"{model_random.best_params_['max_depth']}",
        )

        console.print(table)

        save_params(encoding, target, console, PARAMS_DIR)

        models = {"base model": base_model, "best model": best_model}
        train_classifiers(models, X_train, y_train, console)
        evaluate_classifiers(models, X_train, y_train, test_sets, console)
