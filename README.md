# TSCGlue

Automatic Time Series Classification library built on top of aeon and scikit-learn.

## Installation

```bash
pip install tscglue
```

## Quick Start

```python
from tscglue import utils
from tscglue.models import TSCGlue
from sklearn.metrics import accuracy_score

# Load a time series classification dataset
X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

# Create and train the model
model = TSCGlue(
    random_state=270,
    k_folds=10,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```
