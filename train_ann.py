#!/usr/bin/env python3
"""
Train the Ludo move-selection ANN from the generated dataset (pure NumPy, no PyTorch).

    python train_ann.py                                   # uses ludo_dataset_sample.csv
    python train_ann.py --csv ludo_dataset_full_4000games.csv.gz --epochs 40

Network:  62 inputs -> 128 ReLU -> 64 ReLU -> 4 outputs (one score per token)
Loss:     cross-entropy over a *masked* softmax (illegal tokens get probability 0)
Target:   `expert_action` column (the token the rule-based expert would move)
Output:   ludo_ann.npz  (weights + feature column names, loaded by ludo_game.py)
"""
import argparse
import time

import numpy as np
import pandas as pd

META = ["game_id", "split", "turn", "player", "dice", "n_legal",
        "played_by", "expert_action", "played_action", "won"]


class MLP:
    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for i, o in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / i), (i, o)))   # He init
            self.b.append(np.zeros(o))

    def forward(self, X, keep=False):
        acts, h = [X], X
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = z if k == len(self.W) - 1 else np.maximum(z, 0)
            acts.append(h)
        return (h, acts) if keep else h


def masked_softmax(logits, mask):
    z = np.where(mask > 0, logits, -1e9)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z) * (mask > 0)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(net, X, mask, y):
    pred = masked_softmax(net.forward(X), mask).argmax(axis=1)
    return float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ludo_dataset_sample.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, nargs=2, default=[128, 64])
    ap.add_argument("--out", default="ludo_ann.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    feat_cols = [c for c in df.columns if c not in META]
    print(f"{len(df):,} rows, {len(feat_cols)} input features")

    def part(name):
        d = df[df.split == name]
        return (d[feat_cols].values.astype(np.float64),
                d[[f"legal_{i}" for i in range(4)]].values.astype(np.float64),
                d["expert_action"].values.astype(int))

    Xtr, Mtr, ytr = part("train")
    Xva, Mva, yva = part("val")
    Xte, Mte, yte = part("test")
    print(f"train {len(ytr):,} | val {len(yva):,} | test {len(yte):,}")

    # Baseline: picking a random legal token
    base = float(np.mean(1.0 / Mte.sum(axis=1)))
    print(f"Baseline (random legal move) accuracy: {base:.3f}")

    net = MLP([len(feat_cols), *args.hidden, 4], rng)
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step, best_val, best_state = 0, -1.0, None
    n = len(ytr)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(n)
        total_loss = 0.0
        for s in range(0, n, args.batch):
            idx = order[s:s + args.batch]
            X, M, y = Xtr[idx], Mtr[idx], ytr[idx]
            logits, acts = net.forward(X, keep=True)
            P = masked_softmax(logits, M)
            B = len(idx)
            total_loss += -np.log(P[np.arange(B), y] + 1e-12).sum()

            d = P.copy()
            d[np.arange(B), y] -= 1.0
            d /= B
            gW, gb = [None] * len(net.W), [None] * len(net.b)
            for k in reversed(range(len(net.W))):
                gW[k] = acts[k].T @ d
                gb[k] = d.sum(axis=0)
                if k > 0:
                    d = (d @ net.W[k].T) * (acts[k] > 0)

            step += 1
            lr = args.lr * (0.5 * (1 + np.cos(np.pi * (ep - 1) / args.epochs)))  # cosine decay
            grads = gW + gb
            for j, (p, g) in enumerate(zip(params, grads)):          # Adam
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                mh = m[j] / (1 - 0.9 ** step)
                vh = v[j] / (1 - 0.999 ** step)
                p -= lr * mh / (np.sqrt(vh) + 1e-8)

        va = accuracy(net, Xva, Mva, yva)
        if va > best_val:
            best_val = va
            best_state = ([w.copy() for w in net.W], [b.copy() for b in net.b])
        print(f"epoch {ep:2d}/{args.epochs}  loss {total_loss / n:.4f}  "
              f"val acc {va:.4f}  ({time.time() - t0:.1f}s)")

    net.W, net.b = best_state
    print(f"\nBest val accuracy: {best_val:.4f}")
    print(f"Test accuracy:     {accuracy(net, Xte, Mte, yte):.4f}  (random baseline {base:.3f})")

    save = {f"W{i}": w for i, w in enumerate(net.W)}
    save.update({f"b{i}": b for i, b in enumerate(net.b)})
    save["feature_cols"] = np.array(feat_cols)
    np.savez(args.out, **save)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
