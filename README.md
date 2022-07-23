# Rust TensorFlow PluggableDevice

Ported over from https://github.com/tensorflow/community/pull/352

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
