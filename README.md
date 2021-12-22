# README

## Project Description

This project aims at predicting both the energy accuracy and the simulation duration of a DFT simulation for a given set of input parameters and a prescribed chemical structure. Furthermore, we have implemented an approach to solve the inverse problem, i.e. to generate a set of computationally optimal input parameters for a prescribed chemical structure and energy accuracy.

This repository contains all the code used during the course of the project, including:

- scripts to prepare and launch DFT simulations on the EPFL cluster Fidis (Note that access credentials as well as computing time budget are required to launch them)
- scripts to parse the data from the JSON file and assemble it to the final raw dataset, i.e. with the chemical structures only given as strings
- scripts to assemble the final dataset for different encodings of the chemical structures. These datasets are not saved, but only assembled ad-hoc when they are needed. However, the data loading routines are factored out into a python model (see `code/tools/`), such that it is simple to retrieve them for other purposes.
- scripts to train our models
- notebooks to explore the data and analyse our models
- scripts to create the plots in the report

## Requirements

All the requirements to run this project can be found in `requirements.txt`.

To install them using pip, run the following command:

    pip3 install -r requirements.txt

## Reproducing report results

Please place you at the root of the project before running the following commands.

### Parsing the data (optional)

This command generates one `data.csv` with all `data.json` files from the subfolders inside `data`:

    python3 code/data_preprocessing/parsing_utils.py

Note: since all the data are already parsed, running the previous command is not necessary.

### Training models

After the following commands are run, the trained models and the datasets they were training/testing on are saved in `models` folder.

#### Regression models

    python3 code/regression/delta_E.py
    python3 code/regression/log_delta_E.py
    python3 code/regression/sim_time.py

#### Classification models

    python3 code/classification/delta_E.py

Note: the terminal output of all these commands is automatically saved in `baselines` folder in HTML format.

### Finetuning models

    python3 code/hyperparameter_tuning/delta_E_class.py
    python3 code/hyperparameter_tuning/delta_E_reg.py
    python3 code/hyperparameter_tuning/log_delta_E.py
    python3 code/hyperparameter_tuning/sim_time.py

Note: the terminal output of all these commands is automatically saved in `hyperparameter_tuning` folder in HTML format.

### Optimizing simulation parameters

Before executing any commands in this section, please make sure you trained and saved the models using the commands in the training models section.

The results of the optimization are saved in `code/optimization/optimization_results.json` file.

    python3 code/optimization/optimization.py

## Folders and Files

- `baselines`: html files with training results for different target models and different structure encoding methods.
- `code`
  - `code/data_scraping`:
    - julia scripts for launching simulations
  - `code/data_preprocessing`: scripts to preprocess the data
    - julia scripts
      - parse simulation results to json files
    - python scripts
      - parse simulation results in json files to one csv file `data/data.csv`
  - `code/eda`
    - notebooks for exploratory data analysis
  - `code/tools`: toolbox containing base methods for data preprocessing, model training and evaluation
    - `code/tools/encoding_periodic_table.ipynb`: generate `code/tools/periodic_table_info.json` file containing information about the periodic table and used in some encoding methods
    - `code/tools/data_loader.py`: methods to load data from `data/data.csv`
  - `code/sandbox`: notebooks for testing new ideas and debugging
  - `code/regression`: scripts training, evaluating and saving models for the different regression targets defined for this project.
  - `code/classification`: scripts training, evaluating and saving models for $\Delta E$ classification target (order of magnitude).
  - `code/hyperparameter_tuning`: scripts for hyperparameter tuning for the different models. Use RandomizedSearchCV from the sklearn library to find the best hyperparameters for the different targets. Note that a standard machine might not be able to handle the computational effort of many iterations.
  - `code/model_analysis`: notebooks for analyzing the predictions of the different models.
    - `code/model_analysis/baseline_analysis.ipynb`: analysis of the predictions of the different regression models.
    - `code/model_analysis/classification_decision_boundaries.ipynb`: generate figures displaying the decision boundaries of the $\Delta E$ classifer.
  - `code/optimization`: contains a script implementing the simulation parameter optimization procedure described in the report.
- `data`: contains all the data files. In this folder you may find subfolders with the name of structures which contain simulation results. You may also find 3 csv files:
  - `data/data.csv`: contains the data used for the project. This dataset is built using all the data from the structure folders.
  - `data/ref_energy.csv`: contains the reference energy for each structure.
- `models`: folder in which models trained are saved (python scripts from `code/regression` save their models in this folder).
- `plots`: some plots are saved here.

## Collaborators

You may contact us about the project via the following e-mail adresses:

- Martin Uhrin: martin.uhrin@epfl.ch (supervisor)
- Louis Ponet: louis.ponet@epfl.ch (co-supervisor)
- Auguste Poiroux: auguste.poiroux@epfl.ch
- Nataliya Paulish: nataliya.paulish@epfl.ch
- Philipp Weder: philipp.weder@epfl.ch

## License and Copyright

Licensed under the [MIT License](LICENSE)

© 2021 anp
