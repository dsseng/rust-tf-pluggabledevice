# Rust TensorFlow PluggableDevice

Ported over from https://github.com/tensorflow/community/pull/352

The `tfp-bindings` crate mostly provides *unsafe* bindings to PluggableDevice API `_pywrap_tensorflow_internal.so` library (yes, and currently in only supports Linux and Python 3.10)

Plugin itself (`tfp-plugin`) should link to that library via its `build.rs` script and implement `SE_InitPlugin` and `TF_InitKernel` functions with proper types (these are excluded from bindgen).

There is a safe abstraction over kernel registration functions, which is mostly because:
1. There is quite a limited set of features to be easily supported by an OOP API.
2. This would help with stateful kernels, which benefit from types.

## Try it out

```bash
python3.10 -m venv venv
source venv/bin/activate
pip3 install --no-cache-dir tf-nightly-cpu
# https://github.com/tensorflow/tensorflow/issues/55497
mv venv/lib64 venv/_lib64
cargo build
mkdir venv/lib/python3.10/site-packages/tensorflow-plugins/
cp target/debug/libtfp.so venv/lib/python3.10/site-packages/tensorflow-plugins/
python3 test.py
```

```python3
import tensorflow as tf
tf.config.list_physical_devices()
```

## Running tests

Specifying `LD_LIBRARY_PATH` manually is necessary as of https://github.com/rust-lang/cargo/issues/4044

```
LD_LIBRARY_PATH=$(pwd)/venv/lib/python3.10/site-packages/tensorflow cargo test
````