# hproj - projections of histopathology embeddings into lower dimensions

This is a reimplementation and extension of the experiments run for "Nine Dimensions Are Enough for Histopathology Foundation Model Embeddings". It is expanded with more projectors, multiple classifiers, intrinsic dimensionality estimators, and spatial charactisation measurements.

This version of the experiment has the following stages:
1. Embedding
2. Data Splitting
3. Hyperparameter Calibration
4. Dimension-Performance Curse Estimation
5. Threshold Identification
6. Generalisation Evaluation

## Embeddings
The embeddings are loaded as Cupy arrays.
Each datasets embeddings have train and test splits.
Each each split has an array of feature and an array of labels.

## Data Splitting
For each dataset, the training set is split into *k-folds*.
This can use from cuml.model_selection import KFold.
This generates indices that can be used to index into the training split.

## Projector Hyperparameter Calibration
For each of projectors generate a set of hyperparameter configurations by the grid.
At each of the *callibration dimensions* and for each of the projector, for each hyperparameter configuration, for each fold of the data:
1. fit on the train of the fold
2. transform the train and valid of the fold
3. return the k-nn accuracy for the valid. Note - The k-nn *score* is computed of *values of k* and averaged.
The mean of the folds to find the k-nn score cross validation value for that configuration at that dimension.
Select configuration that performs best across all dimensions and across all datasets (take the mean score over all dimensions for each configurations).
Repeat this *n-times* for stochastic projectors and take the best of the n runs. This is using paired seeds, so have a set of seed defined upfront and reuse them multiple times.

## Classifier Hyperparameter Calibration
With the projector hyperparameters fixed, classifier hyperparameters are tuned at the same calibration dimenions and folds. The procedure is almost the same as above.
1. fit the projector using the training fold
2. transform the train and valid of the fold
3. for each classifier and hyperparameter configuration for that classifier, for each of the k-folds:
    - train on the training set
    - evaluate the on the validation set using *accuracy*.

The mean of the folds to find the accuracy cross validation value for that configuration at that dimension.
Select configuration that performs best across all dimensions (take the mean score over all dimensions for each configurations).

Note - logistic regression is not stochastic because the loss function is convex. So you do stochastic gradient decent but you always end up at the same place.

## Design
What are the inputs and outputs of each process?
Where are they stored in the file system?
How do we make it re-enterant?

FeatureSpace
    features: cupy array
    labels: cupy array

# Setting your .env file
```sh
EMBEDDINGS_ROOT='<add path to the embeddings root here>/'
DATA_ROOT='<add path to the data root here>/'
```

# File Layout
{EMBEDDINGS_ROOT}/
    cache-{dataset}-{encoder} 
    label_map.json 
    train/ 
        meta.json 
        emb_stats.pt 
        emb_{shard}.pt 
        labels_{shard}.pt 
        paths_{shard}.json 
        ... 
    test/ 
        ...

runs/{run_id}/
    log.txt
    config.yaml  # paird seed,
    manifest.json  # run id, git commit hash, timings
    datasets/{dataset}/
        folds/folds_k{k}_seed{seed}.npz
        encoders/{encoder}/
            calibration/
                projectors/{projector}/
                    hyperparameter_configs.json
                    {params_id}/
                        run_seed_{seed}/
                            # params config log and scores over all folds
                            {dim}.json
                    best_parames.json
                classifiers/{classifier}/
                    hyperparameter_configs.json
                    {params_id}/
                        run_seed_{seed}/
                            # params config log and scores over all folds
                            {dim}.json
                    best_parames.json
            curves/{projector}/{classifer}/
                {dim}.json
            evalutation/{projector}/{classifer}/
                hyperparameter_configs.json
                {params_id}/
                    run_seed_{seed}/
                        # params config log and scores over all folds
                        {dim}.json
                best_parames.json
