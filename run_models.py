import numpy as np
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.utils.validation import check_n_jobs
from aeon.transformations.collection.interval_based import QUANTTransformer
import numpy as np
import polars as pl
from aeon.classification.base import BaseClassifier
from aeon.classification.feature_based import (
    Catch22Classifier,
)
import os
from aeon.transformations.collection.convolution_based import Rocket
from aeon.datasets.tsc_datasets import univariate
from sklearn.base import clone
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.convolution_based import RocketClassifier
from sklearn.metrics import accuracy_score
from aeon.classification.interval_based import QUANTClassifier
from autotsc import utils, models, transformers
from tqdm import tqdm
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from aeon.pipeline import make_pipeline as aeon_make_pipeline
from aeon.transformations.collection import Normalizer
import seaborn as sns
import matplotlib.pyplot as plt
from run_stacking import *
import random
import itertools
from aeon.benchmarking import resampling

def get_model(model_names, random_state):
    if model_names == 'rocket':
        return RocketClassifier(n_jobs=-1, random_state=random_state)
    elif model_names == 'multirockethydra':
        return MultiRocketHydraClassifier(n_jobs=-1, random_state=random_state)
    elif model_names == 'catch22':
        return Catch22Classifier(n_jobs=-1, random_state=random_state)
    elif model_names == 'quant':
        return QUANTClassifier(random_state=random_state)
    elif model_names == 'downsample-0.5-multirockethydra':
        return aeon_make_pipeline(
            transformers.DownsampleTransformer(proportion=0.5),
            MultiRocketHydraClassifier(n_jobs=-1, random_state=random_state)
        )
    else:
        raise ValueError(f"Unknown model name: {model_names}")

if '__main__' == __name__:
    write_dir = "experiments/automl_oof_all_classifiers_shuffled/"
    os.makedirs(write_dir, exist_ok=True)

    datasets = univariate
    seeds = range(5)
    model_names = ['rocket', 'catch22', 'quant', 'multirockethydra', 'downsample-0.5-multirockethydra']
    resample_seeds = [0, 1, 2]

    pairs = list(itertools.product(datasets, seeds, model_names, resample_seeds))
    random.shuffle(pairs)

    for dataset, run, model_name, resample_seed in pairs:
        try:
            stats = {
                'dataset': dataset,
                'model': model_name,
                'run': run,
            }

            hash_val = pl.DataFrame([stats]).hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3).item()
            file = f"{write_dir}/{hash_val}.parquet"

            if os.path.exists(file):
                print(f"Skipping {dataset} with {model_name} on run {run}, already exists.")
                continue
            else:
                print(f"Processing {dataset} with {model_name} on run {run}.")

            X_train, y_train, X_test, y_test = utils.load_dataset(dataset)
            X_train, y_train, X_test, y_test = resampling.stratified_resample_data(X_train, y_train, X_test, y_test,random_state=resample_seed)

            model = get_model(model_name, random_state=run)
            m = CrossValidationWrapper(model.clone(), k_folds=10, n_repetitions=1, random_state=run)
            prob = m.fit_predict_proba(X_train, y_train)
            test_prob = m.predict_proba(X_test)

            cmodel = model.clone()
            cmodel.fit(X_train, y_train)
            prob_single = cmodel.predict_proba(X_test)

            stats['oof_pred'] = prob.tolist()
            stats['oof_true'] = y_train.tolist()
            stats['test_ensemble_pred'] = test_prob.tolist()
            stats['test_single_pred'] = prob_single.tolist()
            stats['test_true'] = y_test.tolist()
            stats['classes'] = m.classes_.tolist()

            df_stat = pl.DataFrame([stats])
            df_stat.write_parquet(file, mkdir=True)
        except Exception as e:
            print(f"Error processing {dataset} with {model_name} on run {run}: {e}")



    write_dir = "experiments/automl_oof_all_classifiers/"
    os.makedirs(write_dir, exist_ok=True)

    datasets = univariate
    seeds = range(5)
    model_names = ['rocket', 'catch22', 'quant', 'multirockethydra', 'downsample-0.5-multirockethydra']

    pairs = list(itertools.product(datasets, seeds, model_names))
    random.shuffle(pairs)

    for dataset, run, model_name in pairs:
        try:
            stats = {
                'dataset': dataset,
                'model': model_name,
                'run': run,
            }

            hash_val = pl.DataFrame([stats]).hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3).item()
            file = f"{write_dir}/{hash_val}.parquet"

            if os.path.exists(file):
                print(f"Skipping {dataset} with {model_name} on run {run}, already exists.")
                continue
            else:
                print(f"Processing {dataset} with {model_name} on run {run}.")

            X_train, y_train, X_test, y_test = utils.load_dataset(dataset)
            model = get_model(model_name, random_state=run)
            m = CrossValidationWrapper(model.clone(), k_folds=10, n_repetitions=1, random_state=run)
            prob = m.fit_predict_proba(X_train, y_train)
            test_prob = m.predict_proba(X_test)

            cmodel = model.clone()
            cmodel.fit(X_train, y_train)
            prob_single = cmodel.predict_proba(X_test)

            stats['oof_pred'] = prob.tolist()
            stats['oof_true'] = y_train.tolist()
            stats['test_ensemble_pred'] = test_prob.tolist()
            stats['test_single_pred'] = prob_single.tolist()
            stats['test_true'] = y_test.tolist()
            stats['classes'] = m.classes_.tolist()

            df_stat = pl.DataFrame([stats])
            df_stat.write_parquet(file, mkdir=True)
        except Exception as e:
            print(f"Error processing {dataset} with {model_name} on run {run}: {e}")