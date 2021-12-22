import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import os
import pickle
import sys
from rich.console import Console
from pathlib import Path

from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(
    os.path.dirname(str(Path(os.path.abspath('')).parent.absolute())))

PLOT_ROOT = os.path.join(str(Path(os.path.abspath('')).parent.parent.parent.absolute()), "plots/decision_boundaries")

sys.path.append(os.path.join(ROOT_DIR, "code"))
from tools.transform import magnitude_transform
from tools.utils import get_structure_encoding, StructureEncoding, Target


def load_model(target=Target.DELTA_E_MAGNITUDE,
               encoding=StructureEncoding.ATOMIC, model_name="random_forest_model.pkl"):

    MODELS_DIR = ROOT_DIR + os.sep + "models" + os.sep + target.value + os.sep + encoding.value + os.sep

    with open(MODELS_DIR + model_name, 'rb') as file:
        model = pickle.load(file)

    return model, MODELS_DIR


def load_data_sets(MODELS_DIR, test_set_type="Parameter gen."):
    X_TRAIN_NAME = "X_train.csv"
    y_TRAIN_NAME = "y_train.csv"

    X_train = pd.read_csv(MODELS_DIR + "/datasets/" + X_TRAIN_NAME, index_col=0)
    y_train = pd.read_csv(MODELS_DIR + "/datasets/" + y_TRAIN_NAME, index_col=0)

    X_TEST_NAME = f"X_{test_set_type}.csv"
    y_TEST_NAME = f"y_{test_set_type}.csv"

    X_test = pd.read_csv(MODELS_DIR + "/datasets/" + X_TEST_NAME, index_col=0)
    y_test = pd.read_csv(MODELS_DIR + "/datasets/" + y_TEST_NAME, index_col=0)

    return X_train, y_train, X_test, y_test


def filter_data(X, y, structure_name):
    elt1 = structure_name[0:2]
    elt2 = structure_name[2:]
    mask = (X[elt1] * X[elt2]).to_numpy().nonzero()[0]

    return X.iloc[mask], y.iloc[mask]


def initialize_grid(X_train, X_test, h=0.1):
    x_min_train, x_max_train = X_train['ecutwfc'].min() - 1, X_train['ecutwfc'].max() + 1
    y_min_train, y_max_train = X_train['k_density'].min() - 1, X_train['k_density'].max() + 1

    x_min_test, x_max_test = X_test['ecutwfc'].min() - 1, X_test['ecutwfc'].max() + 1
    y_min_test, y_max_test = X_test['k_density'].min() - 1, X_test['k_density'].max() + 1

    x_max = max(x_max_train, x_max_test)
    y_max = max(y_max_train, y_max_test)

    x_min = min(x_min_train, x_min_test)
    y_min = min(y_min_train, y_min_test)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def assemble_features(xx, yy, rho_factor=4):
    return np.vstack((rho_factor*xx.T, yy.T, xx.T))


def magnitude_prediction(xx, yy, estimator, structure_encoding, rho_factor=4):
    structure_block = np.stack([structure_encoding for _ in range(len(xx))])
    input = np.vstack((assemble_features(xx, yy), structure_block.T)).T
    return estimator.predict(input)


def plot_boundary(structures=[{"name":"NaCl", "rho_factor":4}], target=Target.DELTA_E_MAGNITUDE,
                  encoding=StructureEncoding.ATOMIC, model_name="random_forest_model.pkl",
                  test_set_type="Parameter gen.", verbose=False):

    console = Console()

    with console.status("") as status:
        status.update("Loading model")
        model, MODELS_DIR = load_model(target, encoding, model_name)
        console.log(f'[green]Loaded model from {MODELS_DIR}')

        status.update("Loading data")
        X_train, y_train, X_test, y_test = load_data_sets(MODELS_DIR, test_set_type)

        train_score = accuracy_score(y_train, model.predict(X_train))
        test_score = accuracy_score(y_test, model.predict(X_test))

        if verbose:
            console.print(f"[italic]Train score of loaded model: {100*train_score:.2f}%")
            console.print(f"[italic]Test score of leaded model: {100 * test_score:.2f}%")

        for structure in structures:
            structure_name = structure['name']
            rho_factor = structure['rho_factor']
            console.log(f"[green]Started plotting of {structure_name} with rho factor {rho_factor}")
            status.update("Filtering data")
            X_train_fitered, y_train_fitered = filter_data(X_train, y_train, structure_name)
            X_test_fitered, y_test_fitered = filter_data(X_test, y_test, structure_name)

            if len(y_train_fitered) == 0:
                console.log(f"[bold red] Train subset for {structure_name} empty!")
                # continue

            if len(y_test_fitered) == 0:
                console.log(f"[bold red] Test subset for {structure_name} empty!")
                # continue

            if verbose:
                console.print(f"[italic]Size of train subset for {structure_name}: ", len(y_train_fitered))
                console.print(f"[italic]Size of test subset for {structure_name}: ", len(y_test_fitered))

            xx, yy = initialize_grid(X_train, X_test)

            structure_encoding = get_structure_encoding(structure_name, encoding)

            cm = plt.cm.plasma
            vmax = max(max(y_train['delta_E']), max(y_test['delta_E']))
            vmin = min(min(y_train['delta_E']), min(y_test['delta_E']))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            plt.rcParams['text.usetex'] = True
            fs = 12 #fontsize

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            status.update("Plotting decision boundary")
            predictor = lambda xx, yy: magnitude_prediction(xx, yy, model, structure_encoding)
            Z = predictor(xx.ravel(), yy.ravel())
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm)

            plt.title(structure_name, fontsize=fs+2)

            sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
            fig.colorbar(sm, label="neg. order of magnitude")

            ax.set_xlabel("ecutwfc", fontsize=fs)
            ax.set_ylabel("k_density", fontsize=fs)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())

            status.update("Saving figure")
            PLOT_PATH = PLOT_ROOT + os.sep + encoding.value + os.sep + structure_name + ".png"
            plt.savefig(PLOT_PATH)
            console.log(f"[green]Saved figure to {PLOT_PATH}")
            plt.show()
            console.log(f"[green]Finished plotting of {structure_name}")


if __name__ == "__main__":
    structures = [
        {"name": "NaCl", "rho_factor": 4},
        {"name": "GeTe", "rho_factor": 4},
        {"name": "AgCl", "rho_factor": 4}
        ]
    plot_boundary(structures=structures, verbose=True)
