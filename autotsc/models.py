from time import perf_counter

import numpy as np
import polars as pl
import ray
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import (
    MiniRocketClassifier,
    MultiRocketClassifier,
    RocketClassifier,
)
from aeon.classification.dummy import DummyClassifier
from aeon.classification.feature_based import (
    Catch22Classifier,
    SummaryClassifier,
)
from aeon.classification.interval_based import (
    QUANTClassifier,
)
from aeon.classification.sklearn import SklearnClassifierWrapper
from aeon.pipeline import make_pipeline as aeon_make_pipeline
from aeon.transformations.collection import DownsampleTransformer
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from autotsc import transformers, utils


class RidgeClassifierCVWithProba(RidgeClassifierCV):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)


def default_model_creators(model_n_jobs=4, type="all"):
    if type == "catch22":
        return [
            lambda: Catch22Classifier(n_jobs=model_n_jobs),
            lambda: Catch22Classifier(
                n_jobs=model_n_jobs,
                estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
            ),
        ]
    if type != "rocket-catch22":
        return [
            lambda: RocketClassifier(n_jobs=model_n_jobs),
            lambda: Catch22Classifier(n_jobs=model_n_jobs),
        ]

    return [
        # === Priority 1: Fast Baselines (always run first) ===
        lambda: aeon_make_pipeline(
            DownsampleTransformer(downsample_by="proportion", proportion=0.2),
            MultiRocketClassifier(
                n_jobs=model_n_jobs,
                random_state=6,
                n_kernels=500,
                estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
            ),
        ),
        lambda: aeon_make_pipeline(
            DownsampleTransformer(downsample_by="proportion", proportion=0.5),
            MiniRocketClassifier(
                n_jobs=model_n_jobs,
                random_state=6,
                n_kernels=500,
                estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
            ),
        ),
        lambda: DummyClassifier(),  # Baseline reference
        lambda: SklearnClassifierWrapper(
            make_pipeline(
                StandardScaler(), RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
            )
        ),
        lambda: SummaryClassifier(),  # Fast feature-based
        lambda: SklearnClassifierWrapper(
            RandomForestClassifier(n_estimators=100, n_jobs=model_n_jobs)
        ),
        # === Priority 2: Core ROCKET Models (best quality/speed tradeoff) ===
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=0,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MiniRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=0,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        # === Priority 3: Ensemble Diversity (multiple seeds) ===
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=2,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MiniRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=2,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=3,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MiniRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=3,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=4,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MiniRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=4,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=5,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MiniRocketClassifier(
            n_jobs=model_n_jobs,
            random_state=5,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        lambda: MiniRocketClassifier(
            n_jobs=model_n_jobs,
            n_kernels=500,
            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        ),
        # === Priority 4: Transformed Features ===
        lambda: aeon_make_pipeline(transformers.Difference(), SummaryClassifier()),
        lambda: aeon_make_pipeline(
            transformers.Difference(),
            MultiRocketClassifier(
                n_jobs=model_n_jobs,
                random_state=6,
                n_kernels=500,
                estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
            ),
        ),
        lambda: aeon_make_pipeline(
            transformers.Difference(),
            MiniRocketClassifier(
                n_jobs=model_n_jobs,
                random_state=6,
                n_kernels=500,
                estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
            ),
        ),
        # === Priority 5: Feature-Based Models ===
        lambda: Catch22Classifier(n_jobs=model_n_jobs),
        lambda: aeon_make_pipeline(transformers.Difference(), Catch22Classifier()),
        lambda: QUANTClassifier(random_state=1),
        lambda: QUANTClassifier(
            random_state=2, estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
        ),
        # === Priority 6: CumSum Transformations ===
        lambda: aeon_make_pipeline(
            transformers.CumSum(),
            MultiRocketClassifier(
                n_jobs=model_n_jobs,
                random_state=6,
                n_kernels=500,
                estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
            ),
        ),
        lambda: aeon_make_pipeline(
            transformers.CumSum(),
            MiniRocketClassifier(
                n_jobs=model_n_jobs,
                random_state=6,
                n_kernels=500,
                estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
            ),
        ),
        lambda: aeon_make_pipeline(transformers.CumSum(), Catch22Classifier()),
        # === Priority 7: Expensive/Experimental Models (lowest priority) ===
        lambda: SklearnClassifierWrapper(
            RandomForestClassifier(n_estimators=100, n_jobs=model_n_jobs, ccp_alpha=0.001)
        ),
        lambda: SklearnClassifierWrapper(
            RandomForestClassifier(n_estimators=100, n_jobs=model_n_jobs, ccp_alpha=0.01)
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs, estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs, estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs, estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
        ),
        lambda: MultiRocketClassifier(
            n_jobs=model_n_jobs,
            estimator=ExtraTreesClassifier(n_estimators=400, n_jobs=model_n_jobs),
        ),
        # === Very expensive models (commented out, uncomment if needed) ===
        # lambda: DrCIFClassifier(n_jobs=model_n_jobs, time_limit_in_minutes=0.5),
        # lambda: FreshPRINCEClassifier(n_jobs=model_n_jobs, default_fc_parameters="minimal"),
        # lambda: SklearnClassifierWrapper(TabPFNClassifier(n_preprocessing_jobs=model_n_jobs)),
        # lambda: TimeSeriesForestClassifier(n_jobs=model_n_jobs, n_estimators=50, n_intervals='sqrt-div'),
        # lambda: ProximityForest(n_jobs=model_n_jobs, max_depth=3, n_trees=5),
        # lambda: TemporalDictionaryEnsemble(n_jobs=model_n_jobs),
    ]


@ray.remote(num_cpus=4)
def train_fold(model_id, classifier, fold_id, X, y, folds):
    selected_fold = folds.filter(pl.col("fold") == fold_id).to_dicts()[0]
    train_idx = selected_fold["train_idx"]
    test_idx = selected_fold["test_idx"]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]

    start_time = perf_counter()
    classifier.fit(X_train, y_train)
    end_time = perf_counter()
    training_time = end_time - start_time

    y_pred = classifier.predict(X_test)
    y_pred_zip = zip(test_idx, y_pred.tolist())

    y_prob = classifier.predict_proba(X_test)
    y_prob_zip = zip(test_idx, y_prob.tolist())

    return model_id, classifier, y_pred_zip, y_prob_zip, training_time


class AutoTSCModel(BaseClassifier):
    # TODO: change capability tags
    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def set_use_models(self, model_ids):
        self.use_models = model_ids

    def __init__(
        self,
        n_jobs=-1,
        n_gpus=-1,
        n_folds=8,
        verbose=0,
        model_n_jobs=4,
        model_types="all",
        use_stacking=True,
    ):
        self.n_jobs = n_jobs
        self.n_gpus = n_gpus
        self.n_folds = n_folds
        self.verbose = verbose
        self.model_n_jobs = model_n_jobs
        self.use_stacking = use_stacking

        self.models_ = {}
        self.summary_ = []
        self.model_types = model_types
        # self._owns_ray_cluster = False  # Track if we created the Ray cluster

        # Get model creators in priority order
        model_creators = default_model_creators(model_n_jobs=model_n_jobs, type=model_types)

        self.models = []
        for creator in model_creators:
            try:
                model = creator()
                self.models.append(model)
            except Exception as e:
                model_name = str(e).split("'")[1] if "'" in str(e) else type(e).__name__
                if self.verbose > 0:
                    print(f"Skipping model due to failed initialization: {model_name}")

        super().__init__()

    def build_metamodel(self, classifier, model_id):
        preds = []
        models = []
        for fold_id in range(self.n_folds):
            selected_fold = self.folds_.filter(pl.col("fold") == fold_id).to_dicts()[0]
            train_idx = selected_fold["train_idx"]
            test_idx = selected_fold["test_idx"]

            X = []
            X_proba = []
            y = []
            for row in (
                self.summary()
                .filter(pl.col("stacking_level") == 0)
                .sort("model_id")
                .iter_rows(named=True)
            ):
                X.append(row["fold_predictions"])
                y = row["true_labels"]
                X_proba.append(np.array(row["fold_probabilities"]))
            X = np.array(X).T
            y = np.array(y)
            X_proba = np.hstack(X_proba)
            X = X_proba

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]

            pipeline = clone(classifier)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            preds.extend(zip(test_idx, y_pred.tolist()))
            models.append(pipeline)

        # Compute OOF predictions and accuracy for this meta-model
        fold_predictions = sorted(preds)
        fold_predictions = [p[1] for p in fold_predictions]
        acc = accuracy_score(self.y_, fold_predictions)

        self.summary_.append(
            {
                "model_id": model_id,
                "classifier": repr(classifier).replace("\n", "").replace(" ", ""),
                "fold_predictions": fold_predictions,
                "true_labels": self.y_.tolist(),
                "validation_accuracy": acc,
                "stacking_level": 1,
            }
        )
        self.model_id += 1
        return models

    def save_X_y(self, X, y):
        self.X_, self.y_ = X, y

    def _fit(self, X, y):
        self.save_X_y(X=X, y=y)
        self.folds_ = utils.generate_fold_indices(X, y, n_folds=self.n_folds, shuffle=True)
        self.cpus_available_, self.cpus_to_use_, self.gpus_available_, self.gpus_to_use_ = (
            utils.get_resource_config(self.n_jobs, self.n_gpus)
        )
        if self.verbose > 0:
            utils.print_fit_start_info(
                self.X_,
                self.y_,
                self.cpus_to_use_,
                self.cpus_available_,
                self.gpus_to_use_,
                self.gpus_available_,
            )
        with utils.ray_init_or_reuse(num_cpus=self.cpus_to_use_, num_gpus=self.gpus_to_use_):
            # Put data in Ray object store ONCE (huge performance improvement!)
            X_ref = ray.put(self.X_)
            y_ref = ray.put(self.y_)
            folds_ref = ray.put(self.folds_)

            model_predictions = {}
            model_probabilities = {}
            model_classfiers = {}
            model_training_time = {}

            tasks = []
            self.model_id = 0
            for model_ in self.models:
                model_predictions[self.model_id] = []
                model_probabilities[self.model_id] = []
                model_classfiers[self.model_id] = []
                model_training_time[self.model_id] = []

                for fold in range(self.n_folds):
                    task = train_fold.remote(
                        self.model_id,
                        clone(model_),
                        fold,
                        X_ref,  # Pass reference instead of data
                        y_ref,  # Pass reference instead of data
                        folds_ref,  # Pass reference instead of data
                    )
                    tasks.append(task)
                self.model_id += 1

            remaining_tasks = tasks.copy()

            while remaining_tasks:
                finished, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
                for task in finished:
                    r = ray.get(task)
                    # print(f'Completed task for model_id: {r[0]}')
                    # results.append(r)
                    model_id, model, y_pred, y_prob, model_duration = r
                    model_classfiers[model_id].append(model)
                    model_training_time[model_id].append(model_duration)
                    model_predictions[model_id].extend(y_pred)
                    model_probabilities[model_id].extend(y_prob)

                    if len(model_classfiers[model_id]) == self.n_folds:
                        fold_predictions = sorted(model_predictions[model_id])
                        fold_probabilities = sorted(model_probabilities[model_id])
                        fold_predictions = [p[1] for p in fold_predictions]
                        fold_probabilities = [p[1] for p in fold_probabilities]

                        self.models_[model_id] = tuple(model_classfiers[model_id])
                        acc = accuracy_score(self.y_, fold_predictions)

                        if self.verbose > 0:
                            print(
                                f"Model {model_id} fitted, accuracy: {acc:.4f}, time: {np.mean(model_training_time[model_id]):.2f}s"
                            )

                        self.summary_.append(
                            {
                                "model_id": model_id,
                                "classifier": repr(self.models[model_id])
                                .replace("\n", "")
                                .replace(" ", ""),
                                "fold_predictions": fold_predictions,
                                "fold_probabilities": fold_probabilities,
                                "true_labels": self.y_.tolist(),
                                "validation_accuracy": acc,
                                "training_time_seconds": np.mean(model_training_time[model_id]),
                                "stacking_level": 0,
                            }
                        )

            self.meta_models = {}

            from sklearn.decomposition import PCA

            pipeline = Pipeline(
                [
                    ("pca", PCA(n_components=0.95)),  # 95% variance
                    ("classifier", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
                ]
            )

            if self.use_stacking:
                # Build meta-models with different classifiers
                m1 = self.build_metamodel(RidgeClassifierCV(), "ridge")
                m2 = self.build_metamodel(
                    RandomForestClassifier(n_estimators=500, n_jobs=self.n_jobs), "rf"
                )
                m3 = self.build_metamodel(pipeline, "pca-ridge")
                m4 = self.build_metamodel(HistGradientBoostingClassifier(), "hgb")

                self.meta_models["ridge"] = tuple(m1)
                self.meta_models["rf"] = tuple(m2)
                self.meta_models["pca-ridge"] = tuple(m3)
                self.meta_models["hgb"] = tuple(m4)

            self.use_models = self.models_.keys()

            return self

    def get_avaiable_models(self):
        return self.summary()["model_id"].to_list()

    def summary(self):
        return pl.DataFrame(self.summary_).sort("validation_accuracy")

    def most_common_label(self, all_predictions):
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[counts.argmax()])
        return np.array(final_predictions)

    def predict_per_base_model(self, X):
        if len(self.models_) == 0:
            raise ValueError("No models trained yet. Call fit().")

        with utils.ray_init_or_reuse(num_cpus=self.cpus_to_use_, num_gpus=self.gpus_to_use_):
            tasks = []

            for model_id, models in self.models_.items():
                for model in models:
                    task = make_prediction.remote(model_id, model, X)
                    tasks.append(task)

            all_model_predictions = {}
            results = ray.get(tasks)
            for model_id, predictions in results:
                if model_id not in all_model_predictions:
                    all_model_predictions[model_id] = []
                all_model_predictions[model_id].append(list(predictions))

            for model_id in all_model_predictions:
                model_preds = np.array(all_model_predictions[model_id])
                all_model_predictions[model_id] = self.most_common_label(np.array(model_preds))

            return all_model_predictions

    def predict_proba_per_model(self, X):
        with utils.ray_init_or_reuse(num_cpus=self.cpus_to_use_, num_gpus=self.gpus_to_use_):
            tasks = []

            for model_id, models in self.models_.items():
                for model in models:
                    task = make_prediction.remote(model_id, model, X, return_proba=True)
                    tasks.append(task)

            all_model_predictions = {}
            results = list(ray.get(tasks))
            for model_id, predictions, probabilities in results:
                if model_id not in all_model_predictions:
                    all_model_predictions[model_id] = []
                all_model_predictions[model_id].append(list(probabilities))
            for model_id in all_model_predictions:
                model_probs = np.array(all_model_predictions[model_id])
                avg_probs = np.mean(model_probs, axis=0)
                all_model_predictions[model_id] = avg_probs
            return all_model_predictions

    def predict_per_model(self, X):
        base_model_predictions = self.predict_per_base_model(X)
        base_model_probabilities = self.predict_proba_per_model(X)

        base_models = base_model_predictions.keys()
        base_model = sorted(base_models)

        X = []
        for model_id in base_model:
            X.append(base_model_probabilities[model_id])
        X = np.hstack(X)

        meta_model_predictions = {}

        for model_id in self.meta_models.keys():
            meta_model_predictions[model_id] = []
            for m in self.meta_models[model_id]:
                meta_model_pred = m.predict(X)
                meta_model_predictions[model_id].append(list(meta_model_pred))
            meta_model_predictions[model_id] = self.most_common_label(
                np.array(meta_model_predictions[model_id])
            )
            base_model_predictions[model_id] = meta_model_predictions[model_id]

        return base_model_predictions

    def _predict(self, X):
        if len(self.models_) == 0:
            raise ValueError("No models trained yet. Call fit().")

        sorted_classes = sorted(list(np.unique(self.y_)))

        with utils.ray_init_or_reuse(num_cpus=self.cpus_to_use_, num_gpus=self.gpus_to_use_):
            tasks = []
            c = 0
            for model_id, models in self.models_.items():
                # train all models if use_models is None
                # if not None, only use models in the list
                if model_id not in self.use_models:
                    continue
                for model in models:
                    task = make_prediction.remote(c, model, X, return_proba=True)
                    tasks.append(task)
                    c += 1

            all_predictions = []
            all_probabilities = []
            results = ray.get(tasks)
            results = sorted(results, key=lambda x: x[0])
            for order, predictions, prob in results:
                all_predictions.append(predictions)
                all_probabilities.append(prob)
            all_probabilities = np.array(all_probabilities)
            all_probabilities = np.mean(all_probabilities, axis=0)
            final_predictions = []
            for probs in all_probabilities:
                final_predictions.append(sorted_classes[np.argmax(probs)])
            return np.array(final_predictions)

            # all_predictions = np.array(all_predictions)
            # return self.most_common_label(all_predictions)


@ray.remote(num_cpus=4)
def make_prediction(order, model, X, return_proba=False):
    start_time = perf_counter()
    pred = model.predict(X)
    prob = model.predict_proba(X)
    end_time = perf_counter()
    prediction_time = end_time - start_time
    if return_proba:
        return order, pred, prob
    else:
        return order, pred


# class AutoTSCModel(BaseClassifier):
#
#    # TODO: change capability tags
#    _tags = {
#        "capability:multivariate": True,
#        "capability:train_estimate": True,
#        "capability:contractable": True,
#        "capability:multithreading": True,
#        "algorithm_type": "convolution",
#    }
#
#    def __init__(self, n_jobs=-1, n_gpus=-1, n_folds=8, verbose=0):
#        self.n_jobs = n_jobs
#        self.n_gpus = n_gpus
#        self.step_counter_ = 0
#        self.X_ = None
#        self.y_ = None
#        self.X_features_ = None
#        self.feature_transformers_ = []
#        self.models_ = []
#        self.n_folds = n_folds
#        self.verbose = verbose
#        self.summary_ = []
#        super().__init__()
#
#    def _fit(self, X, y):
#        if self.verbose > 0:
#            n_cpus_available = os.cpu_count() or 1
#            n_cpus_to_use = n_cpus_available if self.n_jobs == -1 else self.n_jobs
#            print(f"CPUs: {n_cpus_to_use}/{n_cpus_available}")
#
#            n_gpus_available = len(tf.config.list_physical_devices('GPU'))
#            n_gpus_to_use = n_gpus_available if self.n_jobs == -1 else min(self.n_jobs, n_gpus_available)
#            print(f"GPUs: {n_gpus_to_use}/{n_gpus_available}")
#
#        self.X_ = X
#        self.y_ = y
#        X_features = X.reshape(X.shape[0], -1)
#        self.X_features_ = pl.DataFrame(X_features, schema=[f"raw|step_{i}" for i in range(X_features.shape[1])])
#
#        self.folds_ = []
#        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
#        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
#            self.folds_.append({
#                'fold': i,
#                'train_idx': train_idx,
#                'test_idx': test_idx,
#            })
#        self.folds_ = pl.DataFrame(self.folds_)
#
#        if self.verbose > 0:
#            print(f"Created {len(self.folds_)} stratified folds for training.")
#        return self
#
#    def step(self):
#        self.add_random_features()
#
#        feature_subset = self.select_random_features()
#        classifier = self.get_random_tabular_model()
#
#        model_training_start_time = time.time()
#        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
#            delayed(self._train_fold)(
#                clone(classifier),
#                fold,
#                feature_subset
#            )
#            for fold in range(self.n_folds)
#        )
#        model_training_end_time = time.time()
#        training_duration = model_training_end_time - model_training_start_time
#        print(f"Trained models for {self.n_folds} folds in {training_duration:.2f} seconds")
#
#        fold_predictions = []
#        fold_models = []
#        for model, y_pred in results:
#            y_pred = list(y_pred)
#            fold_models.append(model)
#            fold_predictions.extend(y_pred)
#
#        fold_predictions = sorted(fold_predictions)
#        fold_predictions = [p[1] for p in fold_predictions]
#        self.models_.append(tuple(fold_models))
#        acc = accuracy_score(self.y_, fold_predictions)
#
#        self.summary_.append({
#            'step': self.step_counter_,
#            'classifier': repr(classifier),
#            'training_time_seconds': training_duration,
#            'fold_predictions': fold_predictions,
#            'true_labels': self.y_.tolist(),
#            'validation_accuracy': acc,
#        })
#        self.step_counter_ += 1
#
#    def _train_fold(self, classifier, fold_id, feature_subset):
#        selected_fold = self.folds_.filter(pl.col("fold") == fold_id).to_dicts()[0]
#        train_idx = selected_fold['train_idx']
#        test_idx = selected_fold['test_idx']
#
#        features_train = self.X_features_.select(feature_subset)
#        features_test = self.X_features_.select(feature_subset)
#
#        features_train = features_train.with_row_index("idx").filter(pl.col("idx").is_in(train_idx)).drop("idx")
#        features_test = features_test.with_row_index("idx").filter(pl.col("idx").is_in(test_idx)).drop("idx")
#
#        y_train = self.y_[train_idx]
#
#        classifier.fit(features_train, y_train)
#
#        y_pred = classifier.predict(features_test)
#        y_pred_zip = zip(test_idx, y_pred.tolist())
#        return classifier, y_pred_zip
#
#    def get_random_tabular_model(self):
#        if np.random.rand() < 0.5:
#            alphas = np.logspace(-4, 4, np.random.randint(9, 16))
#            classifier = Pipeline([
#                ('scaler', StandardScaler(with_mean=False)),
#                ('classifier', RidgeClassifierCV(alphas=alphas))
#            ])
#            return classifier
#        else:
#            ccp_alphas = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
#            selected = np.random.choice(ccp_alphas)
#            classifier = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, ccp_alpha=selected)
#            return classifier
#
#    def get_random_feature_generator(self):
#        rocket_types = [
#            Rocket(n_kernels=500, n_jobs=self.n_jobs, random_state=None),
#            MiniRocket(n_kernels=500, n_jobs=self.n_jobs, random_state=None),
#            MultiRocket(n_jobs=self.n_jobs),
#            QUANTTransformer(),
#            #Catch22Transformer(),
#        ]
#        random_index = np.random.randint(len(rocket_types))
#        return rocket_types[random_index]
#
#    def add_random_features(self):
#        generator = self.get_random_feature_generator()
#        generator_name = generator.__class__.__name__.lower()
#
#        start_time = time.time()
#        new_features = generator.fit_transform(self.X_)
#        feature_extraction_time = time.time() - start_time
#        self.feature_transformers_.append(generator)
#
#        start_idx = self.X_features_.shape[1]
#        feature_cols = [f"{generator_name}|feat_{start_idx + i}" for i in range(new_features.shape[1])]
#        new_features_df = pl.DataFrame(new_features, schema=feature_cols)
#        self.X_features_ = pl.concat([self.X_features_, new_features_df], how="horizontal")
#
#        if self.verbose > 0:
#            print(f"Generated {new_features.shape[1]} {generator_name} features in {feature_extraction_time:.2f}s")
#
#    def select_random_features(self, max_features=20000):
#        n_features = self.X_features_.shape[1]
#        all_feature_names = self.X_features_.columns
#
#        if n_features > max_features:
#            rng = np.random.RandomState(None)
#            feature_indices = rng.choice(n_features, max_features, replace=False)
#            selected_features = [all_feature_names[i] for i in feature_indices]
#        else:
#            selected_features = all_feature_names
#
#        return selected_features
#
#    def _predict(self, X):
#        if len(self.models_) == 0:
#            raise ValueError("No models trained yet. Call fit()/step().")
#
#        # Start with raw features
#        X_features_flat = X.reshape(X.shape[0], -1)
#        X_features = pl.DataFrame(X_features_flat, schema=[f"raw|step_{i}" for i in range(X_features_flat.shape[1])])
#
#        # Add transformed features with same naming convention as in add_random_features
#        feature_dfs = [X_features]
#        start_idx = X_features.shape[1]
#
#        for rocket in self.feature_transformers_:
#            generator_name = rocket.__class__.__name__.lower()
#            new_features = rocket.transform(X)
#
#            feature_cols = [f"{generator_name}|feat_{start_idx + i}" for i in range(new_features.shape[1])]
#            new_features_df = pl.DataFrame(new_features, schema=feature_cols)
#            feature_dfs.append(new_features_df)
#            start_idx += new_features.shape[1]
#
#        X_features = pl.concat(feature_dfs, how="horizontal")
#
#        models_to_use = self.models_
#        models_to_use = [m for i, m in enumerate(self.models_) if i in self.best_models]
#
#        all_predictions = []
#        for model_group in models_to_use:
#            for model in model_group:
#                features = X_features.select(model.feature_names_in_)
#                predictions = model.predict(features)
#                all_predictions.append(predictions)
#
#        all_predictions = np.array(all_predictions)
#        return self.most_common_label(all_predictions)
#
#    def most_common_label(self, all_predictions):
#        final_predictions = []
#        for i in range(all_predictions.shape[1]):
#            votes = all_predictions[:, i]
#            unique, counts = np.unique(votes, return_counts=True)
#            final_predictions.append(unique[counts.argmax()])
#        return np.array(final_predictions)
#
#    def summary(self):
#        return pl.DataFrame(self.summary_)
#
#    def build_ensemble(self):
#        top_3_models = pl.DataFrame(self.summary_).sort("validation_accuracy").tail(len(self.summary_) // 3)
#        self.best_models = top_3_models['step'].to_list()
#        print(f"selected models ({len(self.best_models)}/{len(self.summary_)}): {self.best_models}")
#        pass
#
#
# if __name__ == "__main__":
#    X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")
#    n_jobs = -1
#    model = AutoTSCModel(verbose=1, n_jobs=n_jobs)
#    model.fit(X_train, y_train)
#
#    for _ in range(18):
#        model.step()
#
#    model.build_ensemble()
#
#    pred = model.predict(X_test)
#    acc = accuracy_score(y_test, pred)
#    print(f"AutoTSCModel Test accuracy: {acc}")
#    print(model.summary())
#
#    model = MultiRocketClassifier(n_jobs=n_jobs)
#    model.fit(X_train, y_train)
#    pred = model.predict(X_test)
#    acc = accuracy_score(y_test, pred)
#    print(f"RocketClassifier Test accuracy: {acc}")
