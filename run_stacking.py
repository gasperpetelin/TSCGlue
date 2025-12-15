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
from aeon.pipeline import make_pipeline as aeon_make_pipeline

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

def generate_folds(X, y, n_splits=5, n_repetitions=5, random_state=0):
    all_folds = []
    for i in range(n_repetitions):
        folds = utils.get_folds(X, y, n_splits=n_splits, random_state=random_state+i)
        all_folds.extend(folds)
    return all_folds

class CrossValidationWrapper(BaseClassifier):
    def __init__(self, model, k_folds=10, n_repetitions=1, random_state=None):
        super().__init__()
        self.model = model
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None
        self.predict_time_ = []
        self.predict_time_mean_ = None
        self.cv_splits = None
        self.k_folds = k_folds
        self.n_repetitions = n_repetitions
        self.random_state = random_state

    def _fit(self, X, y):
        raise NotImplementedError()

    def _predict_proba(self, X):
        predictions = []
        for model in self.trained_models_:
            proba = model.predict_proba(X)
            predictions.append(proba)
        avg_proba = np.mean(predictions, axis=0)
        return avg_proba

    def _predict(self, X):
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def get_all_oof_proba(self):
        return pl.DataFrame(self.oof_proba).sort('index', maintain_order=True)

    def _fit_predict_proba(self, X, y):
        if self.cv_splits is None:
            self.cv_splits = generate_folds(X, y, n_splits=self.k_folds, n_repetitions=self.n_repetitions, random_state=self.random_state)
        self.oof_proba = []
        for train_idx, val_idx in tqdm(self.cv_splits):
            model_clone = self.model.clone()
            #print(model_clone)
            #print('RS:', model_clone.random_state)
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, _ = X[val_idx], y[val_idx]
            model_clone.fit(X_train, y_train)
            self.trained_models_.append(model_clone)
            proba = model_clone.predict_proba(X_valid)
            #print(val_idx, train_idx)
            #print(proba)
            prob_columns = []
            for idx, p in zip(val_idx, proba):
                d = {
                    'index': idx,
                }
                for scls, prob in zip(model_clone.classes_, p):
                    k = f'proba_class_{scls}'
                    d[k] = prob.item()
                    if k not in prob_columns:
                        prob_columns.append(k)
                self.oof_proba.append(d)
        return pl.DataFrame(self.oof_proba).group_by('index').mean().sort('index').select(prob_columns).to_numpy()
    
from sklearn.base import BaseEstimator as SKLearnBaseEstimator

class TabCrossValidationWrapper(SKLearnBaseEstimator):
    def __init__(self, model, k_folds=10, n_repetitions=3):
        super().__init__()
        self.model = model
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None
        self.predict_time_ = []
        self.predict_time_mean_ = None
        self.cv_splits = None
        self.k_folds = k_folds
        self.n_repetitions = n_repetitions 

    def _fit(self, X, y):
        raise NotImplementedError()

    def _predict_proba(self, X):
        predictions = []
        for model in self.trained_models_:
            proba = model.predict_proba(X)
            predictions.append(proba)
        avg_proba = np.mean(predictions, axis=0)
        return avg_proba

    def _predict(self, X):
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def get_all_oof_proba(self):
        return pl.DataFrame(self.oof_proba).sort('index', maintain_order=True)

    def fit_predict_proba(self, X, y):
        return self._fit_predict_proba(X, y)

    def _fit_predict_proba(self, X, y):
        self.classes_ = sorted(list(np.unique(y)))
        if self.cv_splits is None:
            self.cv_splits = generate_folds(X, y, n_splits=self.k_folds, n_repetitions=self.n_repetitions, random_state=0)
        self.oof_proba = []
        for train_idx, val_idx in tqdm(self.cv_splits):
            model_clone = self.model.clone()
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, _ = X[val_idx], y[val_idx]
            model_clone.fit(X_train, y_train)
            self.trained_models_.append(model_clone)
            proba = model_clone.predict_proba(X_valid)
            prob_columns = []
            for idx, p in zip(val_idx, proba):
                d = {
                    'index': idx,
                }
                for scls, prob in zip(model_clone.classes_, p):
                    k = f'proba_class_{scls}'
                    d[k] = prob.item()
                    if k not in prob_columns:
                        prob_columns.append(k)
                self.oof_proba.append(d)
        return pl.DataFrame(self.oof_proba).group_by('index').mean().sort('index').select(prob_columns).to_numpy()
    
class Stacker(BaseClassifier):
    def __init__(self, random_state=None, n_repetitions = 1):
        super().__init__()
        self.n_repetitions = n_repetitions
        k_folds=10
        self.random_state = random_state
        self.m1 = CrossValidationWrapper(MultiRocketHydraClassifier(n_jobs=-1, random_state=random_state), k_folds=k_folds, n_repetitions=n_repetitions, random_state=random_state)
        self.m2 = CrossValidationWrapper(QUANTClassifier(random_state=random_state), k_folds=k_folds, n_repetitions=n_repetitions, random_state=random_state)
        self.m3 = CrossValidationWrapper(RDSTClassifier(n_jobs=-1, random_state=random_state), k_folds=k_folds, n_repetitions=n_repetitions, random_state=random_state)
        
        self.use_caruana = False # !!!!!!!!!!!!!!!!!!!!!!
        #model = CatBoostClassifier(
        #    iterations=10000,
        #    early_stopping_rounds=50,
        #    learning_rate=0.0005,
        #    verbose=0
        #)
#
        #self.mm1 = TabCrossValidationWrapper(model=model, k_folds=k_folds, n_repetitions=n_repetitions)

    def _fit(self, X, y):
        def add_argmax_label(df: pl.DataFrame, label_col="label"):
            numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]

            return df.with_columns(
                pl.struct(numeric_cols)
                .map_elements(lambda row: max(row, key=row.get))
                .alias(label_col)
            )

        self.train_pred1 = self.m1.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred1, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('MR vall acc', acc)

        self.train_pred2 = self.m2.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred2, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('QUANT vall acc', acc)

        self.train_pred3 = self.m3.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred3, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('RDST vall acc', acc)

        #self.train_predmm1 = self.mm1.fit_predict_proba(np.hstack([self.train_pred1, self.train_pred2, self.train_pred3]), y)
        #preds = pl.DataFrame(self.train_predmm1, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        #acc = accuracy_score(y, preds)
        #print('Meta CatBoost vall acc', acc)


        if self.use_caruana:
            # reshuffle self.train_pred1 to check if Caruana works correctly
            X = np.random.rand(self.train_pred1.shape[0], self.train_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)
            model_predictions = {
                'MR': self.train_pred1,
                'QUANT': self.train_pred2,
                'RDST': self.train_pred3,
                'TEST': X,
            }
            def accuracy(y_true, y_pred):
                # Convert to integers (0,1,2,3) for argmax comparison
                # but *does not* affect predictions
                unique = sorted(set(y_true))
                mapping = {label: i for i, label in enumerate(unique)}
                y_idx = np.array([mapping[x] for x in y_true])

                y_hat = np.argmax(y_pred, axis=1)
                return np.mean(y_hat == y_idx)

            from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
            self.weights, traj, final_pred = weighted_ensemble_caruana(
                model_predictions=model_predictions,
                targets=y,
                size=50,                                  # ensemble size / num of draws
                metric=accuracy,
                select=max,
            )
            print(self.weights)
            
        else:
            X_meta = np.hstack([self.train_pred1, self.train_pred2, self.train_pred3])
            self.meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifierCV(alphas=np.logspace(-4, 4, 10)))
            ])
            self.meta_model.fit(X_meta, y)

    def _predict(self, X):
        self.test_pred1 = self.m1._predict_proba(X)
        self.test_pred2 = self.m2._predict_proba(X)
        self.test_pred3 = self.m3._predict_proba(X)

        # self.test_predmm1 = self.mm1._predict_proba(np.hstack([self.test_pred1, self.test_pred2, self.test_pred3]))

        if self.use_caruana:
            X = np.random.rand(self.test_pred1.shape[0], self.test_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)

            model_predictions = {
                'MR': self.test_pred1,
                'QUANT': self.test_pred2,
                'RDST': self.test_pred3,
                'TEST': X,
            }

            final_probs = np.zeros((len(X), len(self.classes_)), dtype=float)
            for model_id, weight in self.weights.items():
                final_probs += weight * model_predictions[model_id]
            predicted_indices = np.argmax(final_probs, axis=1)
            return self.classes_[predicted_indices]
        else:
            X_meta = np.hstack([self.test_pred1, self.test_pred2, self.test_pred3])
            return self.meta_model.predict(X_meta)

class FeatureCrossValidationWrapper(BaseClassifier):
    def __init__(self, features, model, k_folds=10, n_repetitions=1, random_state=None):
        super().__init__()
        self.features = features.clone()
        self.model = model
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None
        self.predict_time_ = []
        self.predict_time_mean_ = None
        self.cv_splits = None
        self.k_folds = k_folds
        self.n_repetitions = n_repetitions
        self.random_state = random_state

    def _fit(self, X, y):
        raise NotImplementedError()

    def _fit_predict_proba(self, X, y):
        if self.cv_splits is None:
            self.cv_splits = generate_folds(X, y, n_splits=self.k_folds, n_repetitions=self.n_repetitions, random_state=self.random_state)
        self.oof_proba = []
        Xt = self.features.fit_transform(X)
        for train_idx, val_idx in tqdm(self.cv_splits):
            model_clone = clone(self.model)
            #print(model_clone)
            # print('RS:', model_clone.random_state)
            X_train, y_train = Xt[train_idx], y[train_idx]
            X_valid, _ = Xt[val_idx], y[val_idx]
            model_clone.fit(X_train, y_train)
            self.trained_models_.append(model_clone)
            proba = model_clone.predict_proba(X_valid)
            #print(val_idx, train_idx)
            #print(proba)
            #break
            prob_columns = []
            for idx, p in zip(val_idx, proba):
                d = {
                    'index': idx,
                }
                for scls, prob in zip(model_clone.classes_, p):
                    k = f'proba_class_{scls}'
                    d[k] = prob.item()
                    if k not in prob_columns:
                        prob_columns.append(k)
                self.oof_proba.append(d)
        return pl.DataFrame(self.oof_proba).group_by('index').mean().sort('index').select(prob_columns).to_numpy()

    def _predict_proba(self, X):
        Xt = self.features.transform(X)
        predictions = []
        for model in self.trained_models_:
            proba = model.predict_proba(Xt)
            predictions.append(proba)
        avg_proba = np.mean(predictions, axis=0)
        return avg_proba

    def _predict(self, X):
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

class SparseScaler:
    """Sparse Scaler for hydra transform (NumPy version)."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X, y=None):
        # clamp(0) → clip to minimum 0
        X = np.clip(X, 0, None)
        X = np.sqrt(X)

        # epsilon = mean((X == 0)) ** exponent + 1e-8
        zero_mask = (X == 0).astype(float)
        self.epsilon = zero_mask.mean(axis=0) ** self.exponent + 1e-8

        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + self.epsilon

        return self

    def transform(self, X, y=None):
        X = np.clip(X, 0, None)
        X = np.sqrt(X)

        if self.mask:
            mask = (X != 0).astype(float)
            return ((X - self.mu) * mask) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

from sklearn.base import BaseEstimator, TransformerMixin
class DualScaler(BaseEstimator, TransformerMixin):
    """
    Scales columns based on prefix:
    - 'hydra': hydra_scaler (default: SparseScaler)
    - 'multirocket': rocket_scaler (default: StandardScaler)
    - Other: raises ValueError
    """
    def __init__(self, hydra_scaler, rocket_scaler):
        self.hydra_scaler = hydra_scaler
        self.rocket_scaler = rocket_scaler
    
    def fit(self, X, y=None):
        # Initialize scalers
        self.hydra_scaler_ = self.hydra_scaler
        self.rocket_scaler_ = self.rocket_scaler
        
        # Separate columns by prefix
        self.hydra_cols_ = [col for col in X.columns if col.startswith('hydra')]
        self.multirocket_cols_ = [col for col in X.columns if col.startswith('multirocket')]
        
        # Check for invalid columns
        valid_cols = set(self.hydra_cols_ + self.multirocket_cols_)
        invalid_cols = [col for col in X.columns if col not in valid_cols]
        if invalid_cols:
            raise ValueError(f"Invalid column prefixes found: {invalid_cols}. Only 'hydra' and 'multirocket' prefixes are allowed.")
        
        # Fit scalers
        if self.hydra_cols_:
            self.hydra_scaler_.fit(X.select(self.hydra_cols_).to_numpy())
        if self.multirocket_cols_:
            self.rocket_scaler_.fit(X.select(self.multirocket_cols_).to_numpy())
        
        return self
    
    def transform(self, X):
        # Transform each group
        parts = []
        if self.hydra_cols_:
            parts.append(self.hydra_scaler_.transform(X.select(self.hydra_cols_).to_numpy()))
        if self.multirocket_cols_:
            parts.append(self.rocket_scaler_.transform(X.select(self.multirocket_cols_).to_numpy()))
        
        return np.hstack(parts)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names in output order."""
        return np.array(self.hydra_cols_ + self.multirocket_cols_)

from aeon.transformations.collection import BaseCollectionTransformer, Normalizer
class MultiRocketHydra(BaseCollectionTransformer):
    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "X_inner_type": "numpy3D",
    }
    def __init__(
        self,
        n_jobs=1,
        random_state=None,
    ):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.multirocket_ = MultiRocket(random_state=self.random_state, n_jobs=self.n_jobs)
        self.hydra_ = HydraTransformer(random_state=self.random_state, n_jobs=self.n_jobs)
        self.n_multirocket_features_ = None
        self.n_hydra_features_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self.hydra_.fit(X)
        self.multirocket_.fit(X)
        return self

    def _transform(self, X, y=None):
        X_hydra_t = self.hydra_.transform(X)
        X_multirocket_t = self.multirocket_.transform(X)
        Xt = np.concatenate((X_hydra_t, X_multirocket_t), axis=1)
        self.n_multirocket_features_ = X_multirocket_t.shape[1]
        self.n_hydra_features_ = X_hydra_t.shape[1]

        schema = [
            f'hydra_{x}' for x in range(self.n_hydra_features_)
        ]+[
            f'multirocket_{x}' for x in range(self.n_multirocket_features_)
        ]
        return pl.DataFrame(Xt, schema=schema)

class StackerV2(BaseClassifier):
    def __init__(self, random_state=None, n_repetitions = 'auto', k_folds='auto'):
        super().__init__()
        self.n_repetitions = n_repetitions
        self.k_folds = k_folds
        self.random_state = random_state

        self.use_caruana = False # !!!!!!!!!!!!!!!!!!!!!!

    def _fit(self, X, y):
        n_samples = X.shape[0]

        if self.n_repetitions == 'auto':
            if n_samples < 100:
                self.computes_n_repetitions = 4
            elif n_samples < 500:
                self.computes_n_repetitions = 2
            else:
                self.computes_n_repetitions = 1

        if self.k_folds == 'auto':
            if n_samples < 35:
                self.computes_k_folds = n_samples
            elif n_samples < 150:
                self.computes_k_folds = 10
            else:
                self.computes_k_folds = 8

        print(f'Using {self.computes_k_folds} folds and {self.computes_n_repetitions} repetitions')

        #self.m1 = CrossValidationWrapper(
        #    MultiRocketHydraClassifier(n_jobs=-1, random_state=self.random_state), 
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        #)

        #self.m4 = CrossValidationWrapper(
        #    aeon_make_pipeline(
        #        transformers.RankTransform(),
        #        MultiRocketHydraClassifier(n_jobs=-1, random_state=self.random_state+1)
        #    ), 
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        #)

        #self.m2 = CrossValidationWrapper(
        #    QUANTClassifier(random_state=self.random_state), 
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        #)

        #self.m5 = CrossValidationWrapper(
        #    aeon_make_pipeline(
        #        transformers.RankTransform(),
        #        QUANTClassifier(random_state=self.random_state+1)
        #    ), 
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        #)

        #self.m3 = CrossValidationWrapper(
        #    RDSTClassifier(n_jobs=-1, random_state=self.random_state), 
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        #)
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import ExtraTreesClassifier
        from aeon.transformations.collection.shapelet_based import (
            RandomDilatedShapeletTransform,
        )

        features1 = MultiRocketHydra(n_jobs=-1, random_state=self.random_state)
        models1 = make_pipeline(
            DualScaler(hydra_scaler=SparseScaler(), rocket_scaler=StandardScaler()),
            models.RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
        )
        features2 =  QUANTTransformer()
        models2 = ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.1,
            criterion="entropy",
            random_state=self.random_state,
        )

        features3 = RandomDilatedShapeletTransform(
            max_shapelets=10000,
            shapelet_lengths=None,
            proba_normalization=0.8,
            threshold_percentiles=None,
            alpha_similarity=0.5,
            use_prime_dilations=False,
            n_jobs=-1,
            random_state=self.random_state,
        )
        models3 = make_pipeline(
            StandardScaler(with_mean=True),
            models.RidgeClassifierCVWithProba(
                alphas=np.logspace(-4, 4, 20),
            ),
        )

        features4 = aeon_make_pipeline(
            transformers.RankTransform(),
            MultiRocketHydra(n_jobs=-1, random_state=self.random_state)
        )
        models4 = make_pipeline(
            StandardScaler(),
            models.RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
        )
        features5 = aeon_make_pipeline(
            transformers.RankTransform(),
            QUANTTransformer()
        )
        models5 = ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.1,
            criterion="entropy",
            random_state=self.random_state,
        )


        self.m1 = FeatureCrossValidationWrapper(
            features=features1,
            model=models1,
            k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        )

        self.m2 = FeatureCrossValidationWrapper(
            features=features2,
            model=models2,
            k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        )    
        self.m3 = FeatureCrossValidationWrapper(
            features=features3,
            model=models3,
            k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        )

        self.m4 = FeatureCrossValidationWrapper(
            features=features4,
            model=models4,
            k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        )
        self.m5 = FeatureCrossValidationWrapper(
            features=features5,
            model=models5,
            k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        )

        

        def add_argmax_label(df: pl.DataFrame, label_col="label"):
            numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]

            return df.with_columns(
                pl.struct(numeric_cols)
                .map_elements(lambda row: max(row, key=row.get))
                .alias(label_col)
            )

        self.train_pred1 = self.m1.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred1, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('MR vall acc', acc)

        self.train_pred2 = self.m2.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred2, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('QUANT vall acc', acc)

        self.train_pred3 = self.m3.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred3, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('RDST vall acc', acc)

        self.train_pred4 = self.m4.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred4, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('Ranked MR vall acc', acc)

        self.train_pred5 = self.m5.fit_predict_proba(X, y)
        preds = pl.DataFrame(self.train_pred5, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        acc = accuracy_score(y, preds)
        print('Ranked QUANT vall acc', acc)


        if self.use_caruana:
            X = np.random.rand(self.train_pred1.shape[0], self.train_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)
            model_predictions = {
                'MR': self.train_pred1,
                'QUANT': self.train_pred2,
                'RDST': self.train_pred3,
                'TEST': X,
            }
            def accuracy(y_true, y_pred):
                unique = sorted(set(y_true))
                mapping = {label: i for i, label in enumerate(unique)}
                y_idx = np.array([mapping[x] for x in y_true])

                y_hat = np.argmax(y_pred, axis=1)
                return np.mean(y_hat == y_idx)

            from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
            self.weights, traj, final_pred = weighted_ensemble_caruana(
                model_predictions=model_predictions,
                targets=y,
                size=50,                                  # ensemble size / num of draws
                metric=accuracy,
                select=max,
            )
            print(self.weights)
            
        else:
            X_meta = np.hstack([self.train_pred1, self.train_pred2, self.train_pred3, self.train_pred4, self.train_pred5])
            self.meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifierCV(alphas=np.logspace(-4, 4, 10)))
            ])
            self.meta_model.fit(X_meta, y)

    def _predict(self, X):
        self.test_pred1 = self.m1._predict_proba(X)
        self.test_pred2 = self.m2._predict_proba(X)
        self.test_pred3 = self.m3._predict_proba(X)
        self.test_pred4 = self.m4._predict_proba(X)
        self.test_pred5 = self.m5._predict_proba(X)

        # self.test_predmm1 = self.mm1._predict_proba(np.hstack([self.test_pred1, self.test_pred2, self.test_pred3]))

        if self.use_caruana:
            X = np.random.rand(self.test_pred1.shape[0], self.test_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)

            model_predictions = {
                'MR': self.test_pred1,
                'QUANT': self.test_pred2,
                'RDST': self.test_pred3,
            }

            final_probs = np.zeros((len(X), len(self.classes_)), dtype=float)
            for model_id, weight in self.weights.items():
                final_probs += weight * model_predictions[model_id]
            predicted_indices = np.argmax(final_probs, axis=1)
            return self.classes_[predicted_indices]
        else:
            X_meta = np.hstack([self.test_pred1, self.test_pred2, self.test_pred3, self.test_pred4, self.test_pred5])
            return self.meta_model.predict(X_meta)

def get_model(model_name, random_state):
    if model_name == 'mr-hydra':
        return MultiRocketHydraClassifier(n_jobs=-1, random_state=random_state)
    elif model_name == 'quant':
        return QUANTClassifier(random_state=random_state)
    elif model_name == 'rdst':
        return RDSTClassifier(n_jobs=-1, random_state=random_state)
    elif model_name == 'mixed':
        return Stacker(random_state=random_state, n_repetitions=3)
    elif model_name == 'mixed-v2':
        return StackerV2(random_state=random_state)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

if __name__ == '__main__':

    write_dir = "experiments/stacking_run_v1"
    os.makedirs(write_dir, exist_ok=True)

    for dataset in tqdm(univariate):
        for model_name in ['mr-hydra', 'quant', 'rdst', 'mixed', 'mixed-v2']:
            for run in [100, 200, 300]:
                try:
                    print(f'Running {dataset} with model {model_name} run {run}')

                    stats = {
                        'dataset': dataset,
                        'model': model_name,
                        'run': run,
                    }

                    hash_val = pl.DataFrame([stats]).hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3).item()
                    file = f"{write_dir}/{hash_val}.parquet"

                    if os.path.exists(file):
                        print(f"Skipping: Dataset={dataset}, Run={run}, Model={model_name}")
                        continue
                    else:
                        print(f"Processing: Dataset={dataset}, Run={run}, Model={model_name}")

                    X_train, y_train, X_test, y_test = utils.load_dataset(dataset)


                    model = get_model(model_name, random_state=run)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)

                    stats['test_accuracy'] = acc

                    df_stat = pl.DataFrame([stats])
                    df_stat.write_parquet(file, mkdir=True)
                except Exception as e:
                    print(f"Error processing Dataset={dataset}, Run={run}, Model={model_name}: {e}")