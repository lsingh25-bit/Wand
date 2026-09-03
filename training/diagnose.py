"""training/diagnose.py

Answers the one question the LOCO number alone cannot: when the cascade rejects a
gesture, was the network wrong, or was it right but under-confident?

Those two failure modes need opposite fixes.  If the network is wrong, you need
more data or more augmentation.  If it is right but under-confident, your
thresholds are too tight and you are throwing away recall for false-positive
protection you may not need.  Run this before touching REJ_TAU_CONF.

    python diagnose.py --data data/raw
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras

import wand_config as C
from dataset import load_dataset, loco_folds, to_arrays
from train_cnn import cascade_predict, confusion, train_fold


def sweep_thresholds(probs: np.ndarray, y: np.ndarray) -> None:
    """Report the operating curve the team actually has to choose a point on.

    misfire rate = a NOISE window that fired a spell, or a spell fired as the
    wrong spell.  Under this PS's scoring a misfire is penalised, so the point to
    pick is the knee where misfire rate reaches zero, not the point of maximum
    accuracy."""
    noise = int(C.WandGestureClass.NOISE)
    order = np.argsort(probs, axis=1)
    p1 = probs[np.arange(len(probs)), order[:, -1]]
    p2 = probs[np.arange(len(probs)), order[:, -2]]
    top1 = order[:, -1]

    print(f"{'tau':>6} {'margin':>7} {'recall':>8} {'misfire':>8} {'silent':>8}")
    for tau in (0.50, 0.60, 0.70, 0.80, 0.90):
        for margin in (0.10, 0.25, 0.40):
            fire = (top1 != noise) & (p1 >= tau) & ((p1 - p2) >= margin)
            is_spell = y != noise
            recall = (fire & is_spell & (top1 == y)).sum() / max(is_spell.sum(), 1)
            misfire = (fire & ((y == noise) | (top1 != y))).sum() / max(len(y), 1)
            silent = (~fire & is_spell).sum() / max(is_spell.sum(), 1)
            print(f"{tau:6.2f} {margin:7.2f} {recall:8.3f} {misfire:8.3f} {silent:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.REAL_DATA_DIR)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--aug", type=int, default=8)
    args = ap.parse_args()

    caps = load_dataset(args.data, allow_synthetic=args.allow_synthetic)
    stamp = "SYNTHETIC -- NOT A RESULT | " if any(c.synthetic for c in caps) else ""
    X, y, g = to_arrays(caps)

    all_probs, all_y = [], []
    for caster, tr, te in loco_folds(g):
        m = train_fold(X[tr], y[tr], X[te], y[te], epochs=args.epochs, aug_factor=args.aug)
        p = m.predict(X[te], verbose=0)
        raw_acc = float((p.argmax(1) == y[te]).mean())
        casc_acc = float((cascade_predict(p) == y[te]).mean())
        print(f"{stamp}{caster:<10} argmax={raw_acc:.3f}  cascade={casc_acc:.3f}  "
              f"cost_of_cascade={raw_acc - casc_acc:+.3f}")
        all_probs.append(p)
        all_y.append(y[te])
        keras.backend.clear_session()

    P = np.concatenate(all_probs)
    Y = np.concatenate(all_y)
    print(f"\n{stamp}pooled argmax  {float((P.argmax(1) == Y).mean()):.3f}")
    print(f"{stamp}pooled cascade {float((cascade_predict(P) == Y).mean()):.3f}\n")
    print(f"{stamp}threshold sweep:")
    sweep_thresholds(P, Y)
    print(f"\n{stamp}argmax confusion (rows=true, cols=pred), classes {C.GESTURE_NAMES}:")
    print(confusion(Y, P.argmax(1)))


if __name__ == "__main__":
    main()
