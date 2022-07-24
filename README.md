# Rust TensorFlow PluggableDevice

Ported over from https://github.com/tensorflow/community/pull/352

The `tfp-bindings` crate mostly provides *unsafe* bindings to all TensorFlow C APIs required by plug-ins via `libtensorflow_framework.so.2` library (yes, and currently in only supports Linux and Python 3.10). It also has some *should-be-safe* bindings to main functions required by compute kernels and their registration. Its safety mostly relies on validating data, checking result codes and asserts in debug mode to prevent common plug-in issues like unterminated strings.

Plugin itself (`tfp-plugin`) should link to that library via its `build.rs` script and implement `SE_InitPlugin` and `TF_InitKernel` functions with proper types (these are excluded from bindgen). It may also include optimizer, but it is not yet implemented in this repository.

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
