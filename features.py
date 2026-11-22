from argparse import ArgumentParser
from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLASSES = ("Car", "Pedestrian", "Cyclist")
BINS = np.linspace(0, np.pi, 10)


def image_descriptor(patch):
    rgb = np.asarray(patch.resize((48, 64), Image.Resampling.BILINEAR)) / 255
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    dy, dx = np.gradient(gray)
    angle = np.mod(np.arctan2(dy, dx), np.pi)
    mag = np.hypot(dx, dy)
    features = []
    for y in range(0, 64, 16):
        for x in range(0, 48, 16):
            cell = np.s_[y:y+16, x:x+16]
            hist = np.histogram(angle[cell], BINS, weights=mag[cell])[0]
            features.extend(hist / (np.linalg.norm(hist) + 1e-6))
            features.extend(rgb[cell].mean(axis=(0, 1)))
            features.extend(rgb[cell].std(axis=(0, 1)))
    return features


def project(path, calib_path, width, height):
    lidar = np.fromfile(path, "<f4").reshape(-1, 4)
    cal = {key: np.fromstring(value, sep=" ")
           for key, value in (line.split(":", 1) for line in calib_path.read_text().strip().splitlines())}
    camera = np.c_[lidar[:, :3], np.ones(len(lidar))] @ cal["Tr_velo_to_cam"].reshape(3, 4).T
    camera = camera @ cal["R0_rect"].reshape(3, 3).T
    front = camera[:, 2] > 0.1
    camera = camera[front]
    intensity = lidar[front, 3]
    pixel = np.c_[camera, np.ones(len(camera))] @ cal["P2"].reshape(3, 4).T
    uv = pixel[:, :2] / pixel[:, 2, None]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv[inside], np.c_[camera[inside], intensity[inside]]


def lidar_descriptor(uv, points, box):
    x0, y0, x1, y1 = box
    u, v = uv.T
    inside = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
    q = points[inside]
    n = len(q)
    if not n:
        return np.zeros(32), 0
    nearest = np.quantile(q[:, 2], 0.2)
    surface = q[np.abs(q[:, 2] - nearest) < 2.0]
    lo, med, hi = np.quantile(q, [0.1, 0.5, 0.9], axis=0)
    front_span = np.ptp(surface[:, :3], axis=0) if len(surface) else np.zeros(3)
    hist = np.histogram2d(
        (v[inside] - y0)/(y1-y0), (u[inside] - x0)/(x1-x0),
        bins=4, range=((0, 1), (0, 1))
    )[0].ravel() / n
    return np.r_[np.log1p(n), lo, med, hi, front_span, hist], n


def frame_features(root, frame):
    image = Image.open(root/"image_2"/f"{frame}.png").convert("RGB")
    w, h = image.size
    uv, points = project(root/"velodyne"/f"{frame}.bin", root/"calib"/f"{frame}.txt", w, h)
    labels = (root/"label_2"/f"{frame}.txt").read_text().splitlines()
    for label in labels:
        fields = label.split()
        name = fields[0]
        if name not in CLASSES:
            continue
        x0, y0, x1, y1 = map(float, fields[4:8])
        if y1-y0 < 25:
            continue
        box = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
        patch = image.crop(tuple(map(round, box)))
        lidar, n = lidar_descriptor(uv, points, box)
        yield CLASSES.index(name), image_descriptor(patch), lidar, n


def extract(root, manifest, output):
    rows = []
    for i, row in enumerate(csv.DictReader(manifest.open()), 1):
        frame = row["frame"]
        rows.extend((frame, int(row["fold"]), *obj) for obj in frame_features(root, frame))
        if i % 80 == 0:
            print(f"Feature extraction: {i} frames", flush=True)
    frames, folds, y, camera, lidar, count = map(np.asarray, zip(*rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, frames=frames, folds=folds, y=y, camera=camera, lidar=lidar, count=count)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--data", type=Path, default=ROOT/"data")
    p.add_argument("--manifest", type=Path, default=ROOT/"split.csv")
    p.add_argument("--output", type=Path, default=ROOT/"results/features.npz")
    args = p.parse_args()
    if args.demo:
        rows = [{"class": CLASSES[y], "camera_features": len(x), "lidar_features": len(z),
                 "points_in_box": n} for y, x, z, n in frame_features(args.data, "000000")]
        print(json.dumps(rows, indent=2))
    else:
        extract(args.data, args.manifest, args.output)
