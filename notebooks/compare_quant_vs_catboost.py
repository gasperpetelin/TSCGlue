import numpy as np
from aeon.datasets import load_classification
from aeon.classification.interval_based import QUANTClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)

dataset = 'FaceAll'
X_train, y_train = load_classification(dataset, split="train")
X_test, y_test = load_classification(dataset, split="test")

quant_default = QUANTClassifier(
    random_state=SEED,
)

quant_default.fit(X_train, y_train)
acc_quant = accuracy_score(y_test, quant_default.predict(X_test))

cat = CatBoostClassifier(
    task_type="GPU",
    devices="0",
    random_seed=SEED,
    verbose=True,
)

quant_cat = QUANTClassifier(
    estimator=cat,
    random_state=SEED,
)

quant_cat.fit(X_train, y_train)
acc_cat = accuracy_score(y_test, quant_cat.predict(X_test))

print(f"QUANT (default estimator) accuracy : {acc_quant:.4f}")
print(f"QUANT (CatBoost estimator) accuracy: {acc_cat:.4f}")
