# TODO

- training baselines

  - linear regression on raw data
  - ...

- hyperparameter tuning

  - all the models trained until now aren't hyperparameter tuned

- feature augmentations

  - beside our linear regression model, all our models predict directly on the raw data.

- testing new encodings

  - electronic structure idea
  - (neural networks) we can potentially use embeddings + aggregation (pooling layers) to encode structures
  - ...

- testing generalization performance (on unseen data)

  - to do with several encodings in order to measure which ones allow better generalization

- evaluating results from rf_optimization.ipynb
  - we can create a dataset generated with several structures and delta thresholds,
    then launch simulations with the parameters predicted and compare.
