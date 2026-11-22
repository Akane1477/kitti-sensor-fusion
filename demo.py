import json
import numpy as np

from experiment import KernelClassifier, blend
from features import CLASSES, ROOT, frame_features


def load_branch(snapshot, name):
    branch = KernelClassifier.__new__(KernelClassifier)
    for field in ("center", "scale", "train", "bandwidth", "coef"):
        setattr(branch, field, snapshot[f"{name}_{field}"])
    return branch


with np.load(ROOT/"results/model.npz") as snapshot:
    camera = load_branch(snapshot, "camera")
    lidar = load_branch(snapshot, "lidar")
    cuts, weights = snapshot["cuts"], snapshot["weights"]

frame = "000000"
for y, x, z, count in frame_features(ROOT/"data", frame):
    pc = camera.predict_proba(np.asarray(x)[None, :])
    pl = lidar.predict_proba(np.asarray(z)[None, :])
    weight = weights[np.digitize(count, cuts)]
    combined = blend(pc, pl, weight)[0]
    print(json.dumps({
        "frame": frame,
        "label": CLASSES[y],
        "laser_points": count,
        "laser_weight": float(weight),
        "camera_prediction": CLASSES[pc[0].argmax()],
        "lidar_prediction": CLASSES[pl[0].argmax()],
        "fusion_prediction": CLASSES[combined.argmax()],
        "fusion_scores": dict(zip(CLASSES, combined.round(4).tolist())),
    }, indent=2))
