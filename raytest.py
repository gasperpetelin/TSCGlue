import ray
import sys

ray.init(
    address=None,           # local
    ignore_reinit_error=True,
    runtime_env={
        "working_dir": None,  # Disable auto-packaging of local module
    }
)

@ray.remote
def test():
    import sys
    import autotsc.utils
    return sys.executable

print(ray.get(test.remote()))
