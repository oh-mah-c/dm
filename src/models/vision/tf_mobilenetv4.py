#!/usr/bin/env python3
"""TensorFlow end-to-end MobileNetV4-style trainer for dm.

This module trains all CNN parameters with TensorFlow autodiff.  The C99
MobileNet tiny module remains useful for lightweight inference/operator tests;
this backend is the full end-to-end training path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def require_tensorflow():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        print(
            "error: TensorFlow backend is not installed.\n"
            "Install it with: python3 -m pip install tensorflow\n"
            "The TensorFlow source is tracked as third_party/tensorflow submodule, "
            "but training uses the official Python package runtime.",
            file=sys.stderr,
        )
        raise SystemExit(127) from exc
    return tf, np


def read_manifest(path: str) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    base = Path(path).resolve().parent
    with open(path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected '<image> <label>'")
            image = Path(parts[0])
            if not image.is_absolute() and not image.exists():
                image = base / image
            rows.append((str(image), int(parts[1])))
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def _ppm_token(data: bytes, pos: int) -> Tuple[bytes, int]:
    n = len(data)
    while pos < n:
        ch = data[pos]
        if ch == 35:
            while pos < n and data[pos] not in (10, 13):
                pos += 1
        elif ch in (9, 10, 13, 32):
            pos += 1
        else:
            break
    start = pos
    while pos < n and data[pos] not in (9, 10, 13, 32):
        pos += 1
    return data[start:pos], pos


def load_image_np(path: bytes, size: int):
    tf, np = require_tensorflow()
    del tf
    p = path.decode("utf-8") if isinstance(path, (bytes, bytearray)) else str(path)
    raw = Path(p).read_bytes()
    suffix = Path(p).suffix.lower()
    if suffix == ".ppm" or raw.startswith(b"P6"):
        magic, pos = _ppm_token(raw, 0)
        if magic != b"P6":
            raise ValueError(f"{p}: only binary P6 PPM is supported by the built-in loader")
        width_b, pos = _ppm_token(raw, pos)
        height_b, pos = _ppm_token(raw, pos)
        maxval_b, pos = _ppm_token(raw, pos)
        width, height, maxval = int(width_b), int(height_b), int(maxval_b)
        if maxval <= 0 or maxval > 255:
            raise ValueError(f"{p}: unsupported PPM max value {maxval}")
        if pos < len(raw) and raw[pos] in (9, 10, 13, 32):
            pos += 1
        expected = width * height * 3
        arr = np.frombuffer(raw[pos : pos + expected], dtype=np.uint8)
        if arr.size != expected:
            raise ValueError(f"{p}: truncated PPM payload")
        arr = arr.reshape((height, width, 3)).astype("float32") / float(maxval)
    else:
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("non-PPM images require Pillow: python3 -m pip install pillow") from exc
        with Image.open(p) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = tf.image.resize(arr, (size, size), method="bilinear").numpy()
    return arr.astype("float32")


def make_dataset(rows: Sequence[Tuple[str, int]], size: int, batch: int, shuffle: bool):
    tf, np = require_tensorflow()
    paths = np.asarray([p for p, _ in rows], dtype=object)
    labels = np.asarray([y for _, y in rows], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(rows), seed=123, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.numpy_function(lambda x: load_image_np(x, size), [path], tf.float32)
        image.set_shape((size, size, 3))
        label = tf.cast(label, tf.int32)
        return image, label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


tf, np = require_tensorflow()
_tf_for_layer = tf


@_tf_for_layer.keras.utils.register_keras_serializable(package="dm")
class SharedKVAttention(_tf_for_layer.keras.layers.Layer):
    def __init__(self, heads: int, key_dim: int, sr_stride: int, out_dim: int = None, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.key_dim = key_dim
        self.sr_stride = sr_stride
        self.out_dim = out_dim

        # Instantiate sub-layers in __init__ for proper variable name scoping in Keras v3
        self.q_dense = _tf_for_layer.keras.layers.Dense(self.heads * self.key_dim, use_bias=False, name="q_dense")
        self.k_dense = _tf_for_layer.keras.layers.Dense(self.key_dim, use_bias=False, name="k_dense")
        self.v_dense = _tf_for_layer.keras.layers.Dense(self.key_dim, use_bias=False, name="v_dense")
        
        if self.out_dim is not None:
            self.out_dense = _tf_for_layer.keras.layers.Dense(self.out_dim, use_bias=False, name="out_dense")
        else:
            self.out_dense = None

        if self.sr_stride > 1:
            self.sr_dw = _tf_for_layer.keras.layers.DepthwiseConv2D(3, strides=self.sr_stride, padding="same", use_bias=False, name="sr_dw")
            self.sr_bn = _tf_for_layer.keras.layers.BatchNormalization(momentum=0.0, name="sr_bn")
            self.sr_relu = _tf_for_layer.keras.layers.ReLU(max_value=6.0, name="sr_relu")
        else:
            self.sr_dw = None
            self.sr_bn = None
            self.sr_relu = None

    def build(self, input_shape):
        if self.out_dim is None:
            self.out_dim = int(input_shape[-1])
            self.out_dense = _tf_for_layer.keras.layers.Dense(self.out_dim, use_bias=False, name="out_dense")
        super().build(input_shape)

    def call(self, x, training=None):
        tf = _tf_for_layer
        kv = x
        if self.sr_dw is not None:
            kv = self.sr_dw(kv)
            kv = self.sr_bn(kv, training=training)
            kv = self.sr_relu(kv)
        q = self.q_dense(x)
        k = self.k_dense(kv)
        v = self.v_dense(kv)
        q_shape = tf.shape(q)
        kv_shape = tf.shape(k)
        q = tf.reshape(q, (q_shape[0], q_shape[1] * q_shape[2], self.heads, self.key_dim))
        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.reshape(k, (kv_shape[0], kv_shape[1] * kv_shape[2], self.key_dim))
        v = tf.reshape(v, (kv_shape[0], kv_shape[1] * kv_shape[2], self.key_dim))
        scores = tf.einsum("bhqd,bkd->bhqk", q, k) / math.sqrt(float(self.key_dim))
        weights = tf.nn.softmax(scores, axis=-1)
        ctx = tf.einsum("bhqk,bkd->bhqd", weights, v)
        ctx = tf.transpose(ctx, (0, 2, 1, 3))
        ctx = tf.reshape(ctx, (q_shape[0], q_shape[1], q_shape[2], self.heads * self.key_dim))
        out = self.out_dense(ctx)
        return x + out

    def get_config(self):
        config = super().get_config()
        config.update({
            "heads": self.heads,
            "key_dim": self.key_dim,
            "sr_stride": self.sr_stride,
            "out_dim": self.out_dim
        })
        return config


def conv_bn_relu(x, filters: int, kernel: int, stride: int, name: str):
    tf, _ = require_tensorflow()
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0, name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)


def pw_bn_relu(x, filters: int, name: str, activate: bool = True):
    tf, _ = require_tensorflow()
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_pw")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0, name=f"{name}_bn")(x)
    if activate:
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)
    return x


def dw_bn_relu(x, kernel: int, stride: int, name: str):
    tf, _ = require_tensorflow()
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0, name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)


def fused_ib(x, expanded: int, out_c: int, kernel: int, stride: int, name: str):
    x = conv_bn_relu(x, expanded, kernel, stride, f"{name}_expand")
    return pw_bn_relu(x, out_c, f"{name}_project", activate=True)


def uib(x, kind: str, expanded: int, out_c: int, k1: int, k2: int, stride: int, name: str):
    tf, _ = require_tensorflow()
    residual = x
    use_dw1 = kind in ("convnext", "extradw")
    use_dw2 = kind in ("ib", "extradw")
    if use_dw1:
        x = dw_bn_relu(x, k1, stride, f"{name}_start")
    x = pw_bn_relu(x, expanded, f"{name}_expand", activate=True)
    if use_dw2:
        x = dw_bn_relu(x, k2, 1 if use_dw1 else stride, f"{name}_middle")
    x = pw_bn_relu(x, out_c, f"{name}_project", activate=False)
    if int(residual.shape[-1]) == out_c and stride == 1:
        x = tf.keras.layers.Add(name=f"{name}_residual")([residual, x])
    return x


def build_model(classes: int, image_size: int, width: float, dropout: float):
    tf, _ = require_tensorflow()

    def ch(v: int) -> int:
        return max(8, int(round(v * width / 8.0)) * 8)

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = fused_ib(inputs, ch(16), ch(16), 3, 2, "stem_fused_ib")
    x = uib(x, "extradw", ch(64), ch(24), 3, 3, 2, "uib_extradw")
    x = uib(x, "ib", ch(96), ch(24), 3, 3, 1, "uib_ib")
    x = uib(x, "convnext", ch(96), ch(32), 5, 3, 2, "uib_convnext")
    x = SharedKVAttention(heads=4, key_dim=max(4, ch(32) // 4), sr_stride=2, out_dim=ch(32), name="mobile_mqa")(x)
    x = pw_bn_relu(x, ch(64), "final_expand", activate=True)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name="dm_mobilenetv4_tiny_e2e")


class TrainableModel(tf.Module):
    def __init__(self, model, opt):
        super().__init__()
        # Store model and opt without dependency tracking so tf.saved_model.save does not auto-trace them
        self.model = self._no_dependency(model)
        self.opt = self._no_dependency(opt)
        
        # Explicitly track all model variables as direct attributes
        self.model_vars = []
        for i, v in enumerate(model.variables):
            val = v if isinstance(v, tf.Variable) else v.value
            setattr(self, f"model_var_{i}", val)
            self.model_vars.append(val)
            
        # Collect model trainable variables specifically for optimizer operations
        self.model_trainable_vars = [
            v if isinstance(v, tf.Variable) else v.value
            for v in model.trainable_variables
        ]
            
        # Explicitly track all optimizer variables as direct attributes
        self.opt_vars = []
        for i, v in enumerate(opt.variables):
            val = v if isinstance(v, tf.Variable) else v.value
            setattr(self, f"opt_var_{i}", val)
            self.opt_vars.append(val)

    def train(self, image, label):
        with tf.GradientTape() as tape:
            probs = self.model(image, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(label, probs))
        grads = tape.gradient(loss, self.model_trainable_vars)
        self.opt.apply_gradients(zip(grads, self.model_trainable_vars))
        pred = tf.argmax(probs, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))
        return {'loss': loss, 'accuracy': acc}

    def evaluate(self, image, label):
        probs = self.model(image, training=False)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(label, probs))
        pred = tf.argmax(probs, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))
        return {'loss': loss, 'accuracy': acc}

    def predict(self, image):
        probs = self.model(image, training=False)
        return {'probs': probs}

    def save(self, path):
        vars_list = list(self.model_vars) + list(self.opt_vars)
        
        # Build unique names to avoid SaveV2 duplicate key errors
        seen = {}
        names = []
        for v in vars_list:
            base = getattr(v, 'path', v.name.split(':')[0])
            if base in seen:
                seen[base] += 1
                names.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                names.append(base)

        slices = ['' for _ in vars_list]
        op = tf.raw_ops.SaveV2(
            prefix=path,
            tensor_names=names,
            shape_and_slices=slices,
            tensors=vars_list
        )
        with tf.control_dependencies([op]):
            return {'status': tf.identity(tf.constant('OK'))}

    def load(self, path):
        vars_list = list(self.model_vars) + list(self.opt_vars)
        
        # Reconstruct the exact same unique names in identical order
        seen = {}
        names = []
        for v in vars_list:
            base = getattr(v, 'path', v.name.split(':')[0])
            if base in seen:
                seen[base] += 1
                names.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                names.append(base)

        slices = ['' for _ in vars_list]
        dtypes = [v.dtype for v in vars_list]
        restored = tf.raw_ops.RestoreV2(
            prefix=path,
            tensor_names=names,
            shape_and_slices=slices,
            dtypes=dtypes
        )
        assign_ops = [tf.raw_ops.AssignVariableOp(resource=v.handle, value=r) for v, r in zip(vars_list, restored)]
        with tf.control_dependencies(assign_ops):
            return {'status': tf.identity(tf.constant('OK'))}


def cmd_init_savedmodel(args) -> int:
    tf, _ = require_tensorflow()
    model = build_model(args.classes, args.size, args.width, args.dropout)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
    # 1. Build the optimizer variables eagerly in global scope to prevent uninitialized optimizer resource errors
    opt.build(model.trainable_variables)
    
    # 2. Instantiate TrainableModel (which tracks all model and opt variables directly)
    trainable = TrainableModel(model, opt)

    # 3. Wrap signatures dynamically with tf.function specifying input signatures (traced exactly once during save)
    train_sig = tf.function(
        trainable.train,
        input_signature=[
            tf.TensorSpec(shape=[None, args.size, args.size, 3], dtype=tf.float32, name='image'),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name='label')
        ]
    )
    eval_sig = tf.function(
        trainable.evaluate,
        input_signature=[
            tf.TensorSpec(shape=[None, args.size, args.size, 3], dtype=tf.float32, name='image'),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name='label')
        ]
    )
    predict_sig = tf.function(
        trainable.predict,
        input_signature=[
            tf.TensorSpec(shape=[None, args.size, args.size, 3], dtype=tf.float32, name='image')
        ]
    )
    save_sig = tf.function(
        trainable.save,
        input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.string, name='path')
        ]
    )
    load_sig = tf.function(
        trainable.load,
        input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.string, name='path')
        ]
    )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(trainable, args.output, signatures={
        'train': train_sig,
        'evaluate': eval_sig,
        'predict': predict_sig,
        'save': save_sig,
        'load': load_sig
    })

    from tensorflow.python.tools import saved_model_utils
    meta_graph = saved_model_utils.get_meta_graph_def(args.output, 'serve')
    signature_def = meta_graph.signature_def

    metadata = {}
    for sig_name in ['train', 'evaluate', 'predict', 'save', 'load']:
        if sig_name in signature_def:
            sig = signature_def[sig_name]
            metadata[sig_name] = {
                'inputs': {k: v.name for k, v in sig.inputs.items()},
                'outputs': {k: v.name for k, v in sig.outputs.items()}
            }

    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"initialized trainable SavedModel in {args.output}")
    return 0


def split_rows(rows: Sequence[Tuple[str, int]], validation_split: float):
    if validation_split <= 0.0:
        return list(rows), []
    n_val = max(1, int(round(len(rows) * validation_split)))
    return list(rows[:-n_val]), list(rows[-n_val:])


def cmd_train(args) -> int:
    tf, _ = require_tensorflow()
    rows = read_manifest(args.manifest)
    train_rows, val_rows = split_rows(rows, args.validation_split)
    model = build_model(args.classes, args.size, args.width, args.dropout)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = []
    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.checkpoint, save_best_only=True, monitor="val_accuracy" if val_rows else "accuracy"))
    history = model.fit(
        make_dataset(train_rows, args.size, args.batch, shuffle=True),
        validation_data=make_dataset(val_rows, args.size, args.batch, shuffle=False) if val_rows else None,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    metrics = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(args.output + ".metrics.json", "w", encoding="utf-8") as fp:
        json.dump({"model": model.name, "classes": args.classes, "image_size": args.size, "history": metrics}, fp, indent=2)
    print(f"saved={args.output}")
    return 0


def cmd_eval(args) -> int:
    tf, _ = require_tensorflow()
    model = tf.keras.models.load_model(args.model, custom_objects={"SharedKVAttention": SharedKVAttention})
    rows = read_manifest(args.manifest)
    loss, acc = model.evaluate(make_dataset(rows, args.size, args.batch, shuffle=False), verbose=0)
    print(f"loss={float(loss):.8f}")
    print(f"accuracy={float(acc):.8f}")
    print(f"samples={len(rows)}")
    return 0


def cmd_predict(args) -> int:
    tf, np = require_tensorflow()
    model = tf.keras.models.load_model(args.model, custom_objects={"SharedKVAttention": SharedKVAttention})
    image = load_image_np(args.image.encode("utf-8"), args.size)
    probs = model.predict(np.expand_dims(image, 0), verbose=0)[0]
    order = np.argsort(-probs)[: args.top]
    print(f"model={args.model}")
    print(f"input={args.image}")
    for rank, cls in enumerate(order, 1):
        print(f"rank{rank}_class={int(cls)} prob={float(probs[cls]):.8f}")
    return 0


def cmd_export(args) -> int:
    tf, _ = require_tensorflow()
    model = tf.keras.models.load_model(args.model, custom_objects={"SharedKVAttention": SharedKVAttention})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.format == "keras":
        model.save(args.output)
    elif args.format == "savedmodel":
        model.export(args.output)
    elif args.format == "tflite":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if args.float16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        Path(args.output).write_bytes(converter.convert())
    print(f"exported={args.output}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="dm TensorFlow MobileNetV4-style end-to-end CNN backend")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--manifest", required=True)
    t.add_argument("-o", "--output", required=True)
    t.add_argument("--classes", type=int, required=True)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--size", type=int, default=128)
    t.add_argument("--width", type=float, default=0.5)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--validation-split", type=float, default=0.2)
    t.add_argument("--checkpoint")
    t.set_defaults(func=cmd_train)

    init_sm = sub.add_parser("init-savedmodel")
    init_sm.add_argument("-o", "--output", required=True)
    init_sm.add_argument("--classes", type=int, required=True)
    init_sm.add_argument("--size", type=int, default=128)
    init_sm.add_argument("--width", type=float, default=0.5)
    init_sm.add_argument("--dropout", type=float, default=0.1)
    init_sm.add_argument("--lr", type=float, default=1e-3)
    init_sm.set_defaults(func=cmd_init_savedmodel)

    e = sub.add_parser("eval")
    e.add_argument("--manifest", required=True)
    e.add_argument("-m", "--model", required=True)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--size", type=int, default=128)
    e.set_defaults(func=cmd_eval)

    q = sub.add_parser("predict")
    q.add_argument("-m", "--model", required=True)
    q.add_argument("-i", "--image", required=True)
    q.add_argument("--size", type=int, default=128)
    q.add_argument("--top", type=int, default=5)
    q.set_defaults(func=cmd_predict)

    x = sub.add_parser("export")
    x.add_argument("-m", "--model", required=True)
    x.add_argument("-o", "--output", required=True)
    x.add_argument("--format", choices=("keras", "savedmodel", "tflite"), default="tflite")
    x.add_argument("--float16", action="store_true")
    x.set_defaults(func=cmd_export)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
