"""training/train_cnn.py

Builds, trains, validates leave-one-caster-out, and int8-quantises the spell
classifier, then emits the C array the firmware links against.

    python train_cnn.py                      # real data, refuses if data/raw is empty
    python train_cnn.py --allow-synthetic    # pipeline smoke test, output stamped

Architecture rationale is in report section 6.  In short: a depthwise-separable
1D CNN, ~7k parameters, chosen because the discriminative content of these
gestures is *local temporal shape* (a flick is a 150 ms event) composed into a
*global ordering* (loop-then-flick vs flick-then-loop), which is precisely what a
small stack of 1D convolutions followed by a flatten represents efficiently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

import wand_config as C
from augment import augment_dataset
from dataset import load_dataset, loco_folds, to_arrays

SEED = 1337


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
def build_model(n_classes: int = C.NUM_CLASSES) -> keras.Model:
    """(64, 8) -> (n_classes,) softmax.

    Layer-by-layer reasoning:

    Conv1D(16, k=5, stride=2)
        A 5-sample kernel spans 50 ms, which is roughly the shortest meaningful
        wrist event.  Stride 2 halves the sequence immediately: at 100 Hz with a
        44 Hz anti-alias filter there is very little real information above 25 Hz,
        so the second half of the spectrum is noise we are paying to convolve.
        This one stride is the largest single latency saving in the network.

    Two SeparableConv1D(32, k=3) blocks, each followed by MaxPool(2)
        Separable = depthwise then pointwise.  A dense Conv1D(32, k=3) over 32
        channels costs 3*32*32 = 3072 multiplies per step; separable costs
        3*32 + 32*32 = 1120, a 2.7x saving for a negligible accuracy difference at
        this scale.  Pooling drives 32 -> 16 -> 8 steps, so the final receptive
        field covers most of the gesture while the tensor stays tiny.

    Flatten, NOT GlobalAveragePooling
        This is a deliberate and load-bearing choice.  GAP would discard *when*
        each feature occurred, and when is exactly what separates Alohomora
        (backward loop THEN flick) from Wingardium Leviosa (swish THEN flick) --
        both contain a loop-ish sweep and a flick, differing in order, direction
        and relative timing.  A GAP model confuses them; a flatten model can see
        the ordering.  The cost is 256*16 = 4096 extra weights, which on a chip
        with 520 KB of SRAM is not a real cost.

    BatchNorm before the activation
        Folded into the preceding convolution's weights at conversion time, so it
        costs nothing at inference but materially stabilises training on a small,
        heavily augmented dataset.

    Dropout 0.3 / 0.2
        The dataset will be ~600 real captures.  Without aggressive dropout the
        network memorises casters, and LOCO accuracy collapses while random-split
        accuracy stays flattering.
    """
    inp = keras.Input(shape=(C.N_RESAMPLE, C.N_CHANNELS), name="gesture")

    x = keras.layers.Conv1D(16, 5, strides=2, padding="same", use_bias=False)(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv1D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.SeparableConv1D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(n_classes, activation="softmax", name="spell")(x)

    return keras.Model(inp, out, name="wand_cnn")


def compile_model(m: keras.Model, lr: float = 2e-3) -> keras.Model:
    m.compile(
        optimizer=keras.optimizers.Adam(lr),
        # Label smoothing 0.05: stops the network driving softmax to 1.0 on the
        # training casters.  That over-confidence is what makes a rejection
        # threshold useless -- an over-confident model is confidently wrong on a
        # judge's hand, and every probability lands above any threshold you set.
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return m


# -----------------------------------------------------------------------------
# class weights
# -----------------------------------------------------------------------------
def class_weights(y: np.ndarray) -> dict[int, float]:
    """NOISE is cheap to record and will be over-represented; inverse-frequency
    weighting stops the network taking the easy win of predicting NOISE often.
    The NOISE weight is then nudged UP by 1.25 anyway, because in this scoring
    scheme a missed spell costs a point and a misfire costs more."""
    counts = np.bincount(y, minlength=C.NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    w[int(C.WandGestureClass.NOISE)] *= 1.25
    return {i: float(v) for i, v in enumerate(w)}


# -----------------------------------------------------------------------------
# training / LOCO
# -----------------------------------------------------------------------------
def train_fold(Xtr, ytr, Xte, yte, epochs=60, aug_factor=8, verbose=0):
    Xa, ya = augment_dataset(Xtr, ytr, factor=aug_factor, seed=SEED)
    Ya = keras.utils.to_categorical(ya, C.NUM_CLASSES)
    Yte = keras.utils.to_categorical(yte, C.NUM_CLASSES)

    m = compile_model(build_model())
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                      restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]
    m.fit(Xa, Ya, validation_data=(Xte, Yte), epochs=epochs, batch_size=64,
          class_weight=class_weights(ya), callbacks=cb, verbose=verbose)
    return m


def confusion(y_true, y_pred, n=C.NUM_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def cascade_predict(probs: np.ndarray, kin_rows: np.ndarray | None = None) -> np.ndarray:
    """Apply the same Stage 2/3 decision rule the firmware uses, in Python, so the
    reported accuracy is the accuracy of the SHIPPED system rather than of a bare
    argmax the device never performs."""
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    p1 = probs[np.arange(len(probs)), order[:, -1]]
    p2 = probs[np.arange(len(probs)), order[:, -2]]
    pnoise = probs[:, int(C.WandGestureClass.NOISE)]

    fire = (top1 != int(C.WandGestureClass.NOISE)) & \
           (p1 >= C.REJ_TAU_CONF) & \
           ((p1 - p2) >= C.REJ_TAU_MARGIN) & \
           (pnoise < C.REJ_TAU_NOISE)
    return np.where(fire, top1, int(C.WandGestureClass.NOISE))


# -----------------------------------------------------------------------------
# quantisation and export
# -----------------------------------------------------------------------------
def quantise(model: keras.Model, X_rep: np.ndarray) -> bytes:
    """Full-integer int8 quantisation, int8 in and int8 out.

    Why full-integer rather than float16 or dynamic-range:
      * TFLite-Micro on an ESP32 (Xtensa LX6, no FPU-accelerated SIMD for this)
        runs int8 kernels several times faster than float.
      * int8 in/out means the firmware never allocates a float input buffer --
        preprocessing writes quantised bytes straight into the arena.
      * Weights are 4x smaller, though on this chip that is a bonus, not the point.

    The representative dataset MUST come from the real training distribution.  It
    is what fixes the activation scales; feeding it synthetic or augmented-only
    data sets scales that do not match what the sensor produces, and the model
    then saturates on the device while looking perfect in Python.  This is the
    quiet failure mode that the parity check in P8 exists to catch.
    """
    def rep_gen():
        idx = np.random.default_rng(SEED).permutation(len(X_rep))[:400]
        for i in idx:
            yield [X_rep[i][None, ...].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def export_c_array(tflite: bytes, path: Path, var: str = "g_wand_model") -> None:
    """Emit model_data.cc.  Aligned to 16 bytes because TFLM's flatbuffer reader
    requires alignment and an unaligned model is an immediate hard fault on
    Xtensa -- a crash that looks like a corrupt model rather than a build flag."""
    lines = [f"// AUTO-GENERATED by train_cnn.py -- do not edit",
             f"// {len(tflite)} bytes",
             '#include "model_data.h"', "",
             f"alignas(16) const unsigned char {var}[] = {{"]
    for i in range(0, len(tflite), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in tflite[i:i + 12]) + ",")
    lines += ["};", f"const unsigned int {var}_len = {len(tflite)};", ""]
    path.write_text("\n".join(lines))

    hdr = path.with_suffix(".h")
    hdr.write_text(
        "// AUTO-GENERATED by train_cnn.py -- do not edit\n"
        "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n"
        f"extern const unsigned char {var}[];\n"
        f"extern const unsigned int {var}_len;\n"
        "#endif\n")


def tflite_accuracy(tflite: bytes, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate the QUANTISED model, because that is the artefact that ships.
    A float model that scores 92% and an int8 model that scores 86% are a
    six-point surprise you want to find here, not on stage."""
    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = di["quantization"]
    s_out, z_out = do["quantization"]

    probs = np.zeros((len(X), C.NUM_CLASSES), dtype=np.float32)
    for i, x in enumerate(X):
        q = np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8)
        it.set_tensor(di["index"], q[None, ...])
        it.invoke()
        probs[i] = (it.get_tensor(do["index"])[0].astype(np.float32) - z_out) * s_out
    pred = cascade_predict(probs)
    return float((pred == y).mean()), probs


# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.REAL_DATA_DIR)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--aug", type=int, default=8)
    # Anchored to the repo root (see wand_config._REPO_ROOT) rather than a bare
    # relative string -- a bare "firmware/wand_infer" silently writes to a
    # phantom training/firmware/wand_infer/ when this script is launched with
    # `cd training && python train_cnn.py`, leaving the real model_data.cc at
    # the repo root untouched with no error at all.
    ap.add_argument("--out", default=str(Path(C.__file__).resolve().parent.parent /
                                          "firmware" / "wand_infer"))
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    caps = load_dataset(args.data, allow_synthetic=args.allow_synthetic)
    synthetic = any(c.synthetic for c in caps)
    stamp = "SYNTHETIC -- NOT A RESULT | " if synthetic else ""

    X, y, g = to_arrays(caps)
    n_casters = len(set(g))
    print(f"{stamp}{len(X)} captures, {n_casters} casters, "
          f"class counts {np.bincount(y, minlength=C.NUM_CLASSES).tolist()}")

    # ---- leave-one-caster-out -------------------------------------------------
    # LOCO needs >= 2 distinct casters to form even one train/test split -- with
    # only one, `train` is empty for every fold and augment_dataset() crashes on
    # an empty array deep in a numpy stack() call, which looks like a bug in
    # augment.py but is really just this precondition going unchecked. Fail
    # loudly and specifically here instead, and let the run continue far enough
    # to produce a real, flashable model_data.cc -- just an HONESTLY-LABELLED one
    # that has not been validated on an unseen hand, same quarantine philosophy
    # as the synthetic-data stamp above.
    loco_validated = n_casters >= 2
    accs, cm_total = [], np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=int)
    if not loco_validated:
        print(f"\n{stamp}SKIPPING leave-one-caster-out: only {n_casters} caster(s) "
              f"present, need >= 2 to form a single train/test split.")
        print(f"{stamp}The model below will be trained on all of this caster's data "
              f"and will very likely OVERFIT to their specific hand -- it is useful "
              f"for testing the firmware pipeline end to end, but its accuracy on "
              f"anyone else's cast is unknown until more casters are collected.")
    else:
        for caster, tr, te in loco_folds(g):
            m = train_fold(X[tr], y[tr], X[te], y[te], epochs=args.epochs, aug_factor=args.aug)
            probs = m.predict(X[te], verbose=0)
            pred = cascade_predict(probs)
            acc = float((pred == y[te]).mean())
            accs.append(acc)
            cm_total += confusion(y[te], pred)
            print(f"{stamp}LOCO fold hold-out={caster:<10} n={te.sum():4d}  acc={acc:6.3f}")
            keras.backend.clear_session()

        print(f"\n{stamp}LOCO mean {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        if synthetic and np.std(accs) < 1e-6:
            print("  (zero variance across folds is the signature of generated data -- "
                  "expected here, and disqualifying on real data)")
        print(f"{stamp}confusion matrix (rows=true, cols=predicted):")
        print(cm_total)

    # ---- final model on everything, then quantise -----------------------------
    final = train_fold(X, y, X, y, epochs=args.epochs, aug_factor=args.aug)
    final.summary()

    tfl = quantise(final, X)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wand_model.tflite").write_bytes(tfl)
    export_c_array(tfl, out / "model_data.cc")

    acc_q, _ = tflite_accuracy(tfl, X, y)
    n_params = int(sum(np.prod(w.shape) for w in final.get_weights()))
    print(f"\n{stamp}params={n_params}  tflite={len(tfl)} bytes "
          f"({len(tfl)/1024:.1f} KiB)  int8 train-set acc={acc_q:.3f}")

    (out / "model_meta.json").write_text(json.dumps({
        "synthetic": synthetic,
        "params": n_params, "tflite_bytes": len(tfl),
        "n_casters": n_casters,
        "loco_validated": loco_validated,
        "loco_mean": float(np.mean(accs)) if accs else None,
        "loco_std": float(np.std(accs)) if accs else None,
        "classes": C.GESTURE_NAMES,
    }, indent=2))
    if not loco_validated:
        print(f"\n{stamp}model_meta.json written with loco_validated=false -- "
              f"do not report this as an accuracy number until >= 2 casters exist.")


if __name__ == "__main__":
    main()
