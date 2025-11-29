"""Test if Ray's serialization makes SVM arrays read-only"""
import numpy as np
import ray
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Test 1: Train SVM locally (no Ray)
print("=" * 60)
print("Test 1: SVM trained locally (no Ray)")
print("=" * 60)
X_train, y_train = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
X_test, _ = make_classification(n_samples=10, n_features=20, n_classes=2, random_state=43)

local_svm = SVC(kernel="linear", probability=True)
local_svm.fit(X_train, y_train)

print(f"support_vectors_ writable: {local_svm.support_vectors_.flags['WRITEABLE']}")
print(f"_dual_coef_ writable: {local_svm._dual_coef_.flags['WRITEABLE']}")
print(f"_probA writable: {local_svm._probA.flags['WRITEABLE']}")
print(f"_probB writable: {local_svm._probB.flags['WRITEABLE']}")

try:
    proba = local_svm.predict_proba(X_test)
    print(f"✓ predict_proba works! Shape: {proba.shape}")
except Exception as e:
    print(f"✗ predict_proba failed: {e}")

# Test 2: Train SVM in Ray task
print("\n" + "=" * 60)
print("Test 2: SVM trained in Ray task")
print("=" * 60)

@ray.remote(num_cpus=1)
def train_svm_in_ray(X, y):
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X, y)
    return svm

ray.init(num_cpus=4, ignore_reinit_error=True)

ray_svm_ref = train_svm_in_ray.remote(X_train, y_train)
ray_svm = ray.get(ray_svm_ref)

print(f"support_vectors_ writable: {ray_svm.support_vectors_.flags['WRITEABLE']}")
print(f"_dual_coef_ writable: {ray_svm._dual_coef_.flags['WRITEABLE']}")
print(f"_probA writable: {ray_svm._probA.flags['WRITEABLE']}")
print(f"_probB writable: {ray_svm._probB.flags['WRITEABLE']}")

print("\nTrying to call predict_proba on X_test array...")
try:
    proba = ray_svm.predict_proba(X_test)
    print(f"✓ predict_proba works! Shape: {proba.shape}")
except Exception as e:
    print(f"✗ predict_proba failed: {type(e).__name__}: {e}")

print("\nTrying to make X_test writable and call predict_proba...")
X_test_writable = np.array(X_test, dtype=np.float64, order='C', copy=True)
X_test_writable.setflags(write=True)
print(f"X_test_writable flags: {X_test_writable.flags['WRITEABLE']}, C_CONTIGUOUS: {X_test_writable.flags['C_CONTIGUOUS']}")

try:
    proba = ray_svm.predict_proba(X_test_writable)
    print(f"✓ predict_proba works with writable X! Shape: {proba.shape}")
except Exception as e:
    print(f"✗ predict_proba still failed: {type(e).__name__}: {e}")

# Test 3: Fix SVM internal arrays after Ray deserialization
print("\n" + "=" * 60)
print("Test 3: Fix SVM internal arrays after Ray")
print("=" * 60)

def make_svm_writable(svm):
    """Make all internal arrays of SVM writable"""
    if hasattr(svm, 'support_vectors_'):
        svm.support_vectors_ = np.array(svm.support_vectors_, copy=True)
        svm.support_vectors_.setflags(write=True)
    if hasattr(svm, '_dual_coef_'):
        svm._dual_coef_ = np.array(svm._dual_coef_, copy=True)
        svm._dual_coef_.setflags(write=True)
    if hasattr(svm, '_probA'):
        svm._probA = np.array(svm._probA, copy=True)
        svm._probA.setflags(write=True)
    if hasattr(svm, '_probB'):
        svm._probB = np.array(svm._probB, copy=True)
        svm._probB.setflags(write=True)
    return svm

ray_svm_fixed = make_svm_writable(ray_svm)

print(f"After fix - support_vectors_ writable: {ray_svm_fixed.support_vectors_.flags['WRITEABLE']}")
print(f"After fix - _dual_coef_ writable: {ray_svm_fixed._dual_coef_.flags['WRITEABLE']}")
print(f"After fix - _probA writable: {ray_svm_fixed._probA.flags['WRITEABLE']}")
print(f"After fix - _probB writable: {ray_svm_fixed._probB.flags['WRITEABLE']}")

try:
    proba = ray_svm_fixed.predict_proba(X_test)
    print(f"✓ predict_proba works after fixing SVM! Shape: {proba.shape}")
except Exception as e:
    print(f"✗ predict_proba still failed: {type(e).__name__}: {e}")

ray.shutdown()
print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
