from argparse import ArgumentParser
from pathlib import Path
import csv
import json

import numpy as np
from features import CLASSES, ROOT



def squared_distances(a, b):
    aa = np.einsum("ij,ij->i", a, a)[:, None]
    bb = np.einsum("ij,ij->i", b, b)[None, :]
    return np.maximum(aa + bb - 2 * (a @ b.T), 0)


class KernelClassifier:
    def __init__(self, x, y):
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-5)
        self.train = (x - self.center) / self.scale
        d = squared_distances(self.train, self.train)
        self.bandwidth = np.median(d[d > 0])
        kernel = np.exp(-d/self.bandwidth)
        class_count = np.bincount(y, minlength=len(CLASSES))
        penalty = 0.2 * np.sqrt(class_count[y] / class_count.mean())
        kernel[np.diag_indices_from(kernel)] += penalty
        self.coef = np.linalg.solve(kernel, np.eye(len(CLASSES))[y])

    def predict_proba(self, x):
        aligned = (x - self.center) / self.scale
        kernel = np.exp(-squared_distances(aligned, self.train)/self.bandwidth)
        scores = 8 * (kernel @ self.coef)
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


def fit(x, y):
    return KernelClassifier(x, y)


def blend(camera, lidar, weight):
    alpha = np.broadcast_to(weight, camera.shape[:1])[:, None]
    return camera * (1-alpha) + lidar * alpha


def learn_gate(camera, lidar, count, labels):
    cuts = np.quantile(count[count > 0], [1/3, 2/3])
    bins = np.digitize(count, cuts)
    weights = []
    for i in range(3):
        mask = bins == i
        scores = [
            -np.log(blend(camera[mask], lidar[mask], w)[np.arange(mask.sum()), labels[mask]]).mean()
            + 0.03 * (w-0.5)**2 for w in np.linspace(0, 1, 21)
        ]
        weights.append(np.argmin(scores) / 20)
    return cuts, np.array(weights)


def metrics(y, prob):
    pred = prob.argmax(axis=1)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    tp = matrix.diagonal()
    f1 = 2 * tp / (matrix.sum(axis=0) + matrix.sum(axis=1))
    return {
        "accuracy": float(tp.sum()/len(y)),
        "macro_f1": float(f1.mean()),
        "class_f1": f1.tolist(),
        "confusion": matrix.tolist(),
    }


def run(features, out):
    camera, lidar, count, y, fold, frames = [features[k] for k in ("camera", "lidar", "count", "y", "folds", "frames")]
    train, test = fold > 0, fold == 0
    cv_camera = np.zeros((len(y), len(CLASSES)))
    cv_lidar = np.zeros_like(cv_camera)
    for k in range(1, 5):
        held = fold == k
        use = train & ~held
        cv_camera[held] = fit(camera[use], y[use]).predict_proba(camera[held])
        cv_lidar[held] = fit(lidar[use], y[use]).predict_proba(lidar[held])
    cuts, weights = learn_gate(cv_camera[train], cv_lidar[train], count[train], y[train])
    image_model = fit(camera[train], y[train])
    lidar_model = fit(lidar[train], y[train])
    pc = image_model.predict_proba(camera[test])
    pl = lidar_model.predict_proba(lidar[test])
    alpha = weights[np.digitize(count[test], cuts)]
    neutral = lidar_model.predict_proba(np.zeros((1, lidar.shape[1])))[0]
    y_test, test_frames = y[test], frames[test]
    conditions = {}
    shuffled = np.random.default_rng(2026).permutation(np.unique(test_frames))
    actual = {}
    for ratio in (0, 50, 100):
        missing = np.isin(test_frames, shuffled[:len(shuffled)*ratio//100])
        degraded = np.where(missing[:, None], neutral, pl)
        probs = {
            "camera": pc,
            "lidar": degraded,
            "equal": blend(pc, degraded, 0.5),
            "equal_routed": blend(pc, degraded, np.where(missing, 0, 0.5)),
            "adaptive": blend(pc, degraded, np.where(missing, 0, alpha)),
        }
        conditions[str(ratio)] = {name: metrics(y_test, prob) for name, prob in probs.items()}
        actual[str(ratio)] = {"frames": len(np.unique(test_frames[missing])), "objects": int(missing.sum())}
    result = {
        "classes": CLASSES, "train_frames": len(np.unique(frames[train])),
        "test_frames": len(np.unique(test_frames)), "train_support": np.bincount(y[train]).tolist(),
        "test_support": np.bincount(y_test).tolist(), "cuts": cuts.tolist(), "weights": weights.tolist(),
        "train_density_support": np.bincount(np.digitize(count[train], cuts), minlength=3).tolist(),
        "test_density_support": np.bincount(np.digitize(count[test], cuts), minlength=3).tolist(),
        "missing": actual, "conditions": conditions,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out/"predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "truth", "lidar_points", "camera", "lidar", "equal", "adaptive", "missing_half"])
        clean = [p.argmax(axis=1) for p in (pc, pl, blend(pc,pl,0.5), blend(pc,pl,alpha))]
        missing_half = np.isin(test_frames, shuffled[:len(shuffled)//2])
        for row in zip(test_frames, y_test, count[test], *clean, missing_half):
            frame, truth, n, a, b, c, d, missing = row
            writer.writerow([frame, CLASSES[truth], n, CLASSES[a], CLASSES[b], CLASSES[c], CLASSES[d], int(missing)])
    model = {"cuts": cuts, "weights": weights}
    for name, fitted in (("camera", image_model), ("lidar", lidar_model)):
        for field in ("center", "scale", "train", "bandwidth", "coef"):
            model[f"{name}_{field}"] = getattr(fitted, field)
    np.savez_compressed(out/"model.npz", **model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--features", type=Path, default=ROOT/"results/features.npz")
    p.add_argument("--output", type=Path, default=ROOT/"results")
    args = p.parse_args()
    with np.load(args.features) as features:
        run(features, args.output)
