import os
import numpy as np
import pandas as pd
import sys
from time import sleep
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import xgboost as xgb

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

# Set up
sys.path.append(os.path.dirname(os.getcwd()))
from tools.utils import encode_all_structures, Encoding

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "data/"
)

# Loading Data
encoding = Encoding.COLUMN_MASS

df = pd.read_csv(os.path.join(DATA_DIR, "data.csv"), index_col=0, na_filter=False)
df = encode_all_structures(df, encoding)

cols_raw = list(df.columns)
cols_trash = ["structure", 'converged', "n_iterations", "delta_E", "fermi", "total_energy"]
cols_independent = ['time']
cols_drop = cols_trash + cols_independent

cols_dependent = cols_raw.copy()
for element in cols_drop:
    cols_dependent.remove(element)

X_raw = df[cols_dependent][df["converged"]]
y_raw = np.abs(df[cols_independent][df["converged"]]).squeeze()

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw,
    test_size=0.2,
    random_state=42
)

# Model definition
linear_augmented_model = Pipeline([
    ('scaler_init', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2)),
    ('scaler_final', StandardScaler()),
    ('regressor', LinearRegression()),
])

rf_model = RandomForestRegressor(random_state=0)

gb_model = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05, random_state=0)

xgb_model = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.05, random_state=0)

models = {
    "Augmented Linear Regression": linear_augmented_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model,
}

# Model training

console = Console()
with console.status("[bold blue]Training models...") as status:
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        console.log(f"[blue]Finished training {model_name}[/blue]")


# Model evaluation

with console.status("[bold blue] Evaluating models...") as status:
    for model_name, model in models.items():
        sleep(0.5)
        console.log(f"[blue]Evaluating {model_name}[/blue]")

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mse_test = mean_squared_error(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)

        titlestr = f'MSE\n'
        contentstr = f"train:{mse_train:.4E}\ttest:{mse_test:.4E}"
        p1 = Panel(titlestr+contentstr)

        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        titlestr = f'MAE\n'
        contentstr = f"train:{mae_train:.4E}\ttest:{mae_test:.4E}"
        p2 = Panel(titlestr + contentstr)

        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
        mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

        titlestr = f'MAPE\n'
        contentstr = f"train:{mape_train:.4E}\ttest:{mape_test:.4E}"
        p3 = Panel(titlestr+contentstr)

        renderables = [p1, p2, p3]
        console.print(Columns(renderables))

        # mape_test = custom_mape(y_test, y_pred_test)
        # mape_train = custom_mape(y_train, y_pred_train)
        # print(f"Custom MAPE:\ttrain:{mape_train:.4E}\ttest:{mape_test:.4E}")

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "models/sim_time/"
)

save_models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model,
}

models_filenames = [
    "random_forest_model.pkl",
    "gb_model.pkl",
    "xgb_model.pkl",
]

with console.status("[bold green] Saving models...") as status:
    for filename, (model_name, model) in zip(models_filenames, save_models.items()):
        sleep(1)
        modelpath = MODELS_DIR + filename
        with open(modelpath, "wb") as file:
            pickle.dump(model, file)
        console.log(f"[green]Saved {model_name} to {modelpath}[/green]")
