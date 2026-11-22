from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from struct import unpack
from urllib.error import URLError
from urllib.request import Request, urlopen

from zipfile import ZipFile, ZIP_DEFLATED
import csv
import random
import zlib

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
ARCHIVES = {
    "label_2": "data_object_label_2.zip",
    "calib": "data_object_calib.zip",
    "image_2": "data_object_image_2.zip",
    "velodyne": "data_object_velodyne.zip",
}
ROOT = Path(__file__).resolve().parent
CLASSES = {"Car", "Pedestrian", "Cyclist"}


class RemoteZip(BytesIO):
    def __init__(self, url):
        self.url = url
        with urlopen(Request(url, headers={"Range": "bytes=-65536"}), timeout=60) as r:
            self.total = int(r.headers["Content-Range"].split("/")[-1])
            self.tail = r.read()
        self.pos = 0

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.total)[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def get(self, start, size):
        data = bytearray()
        end = start + size - 1
        failures = 0
        while len(data) < size:
            offset = start + len(data)
            req = Request(self.url, headers={"Range": f"bytes={offset}-{end}"})
            try:
                with urlopen(req, timeout=60) as response:
                    try:
                        data.extend(response.read())
                    except IncompleteRead as partial:
                        if not partial.partial:
                            raise
                        data.extend(partial.partial)
            except (URLError, TimeoutError, ConnectionResetError):
                failures += 1
                if failures == 4:
                    raise
        return bytes(data)

    def read(self, size=-1):
        start = self.pos
        size = self.total - start if size < 0 else min(size, self.total - start)
        self.pos += size
        tail_start = self.total - len(self.tail)
        if not size:
            return b""
        if start >= tail_start:
            return self.tail[start-tail_start:start-tail_start+size]
        return self.get(start, size)

    def member(self, info):
        offset = info.header_offset
        b = self.get(offset, 30 + len(info.filename) + info.compress_size + 512)
        n, extra = unpack("<HH", b[26:30])
        payload = b[30+n+extra:30+n+extra+info.compress_size]
        data = zlib.decompress(payload, -15) if info.compress_type == ZIP_DEFLATED else payload
        assert zlib.crc32(data) == info.CRC
        return data


def archive(name):
    remote = RemoteZip(BASE + name)
    return ZipFile(BytesIO(remote.read()))


def eligible(line):
    s = line.split()
    return s[0] in CLASSES and float(s[7]) - float(s[5]) >= 25


def manifest(output):
    with archive("devkit_object.zip") as kit, archive(ARCHIVES["label_2"]) as labels:
        paths = kit.read("mapping/train_mapping.txt").decode().splitlines()
        order = list(map(int, kit.read("mapping/train_rand.txt").decode().strip().split(",")))
        by_id = {i: paths[j-1].split()[1] for i, j in enumerate(order)}
        drives = sorted(set(by_id.values()))
        random.Random(2026).shuffle(drives)
        folds = {d: i % 5 for i, d in enumerate(drives)}
        groups = {"train": [], "test": []}
        for i, d in by_id.items():
            lines = labels.read(f"training/label_2/{i:06d}.txt").decode().splitlines()
            if any(eligible(line) for line in lines):
                groups["test" if folds[d] == 0 else "train"].append((i, d))
        with output.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "drive", "split", "fold"])
            for split, count in (("train", 440), ("test", 200)):
                picked = random.Random(2026).sample(groups[split], min(count, len(groups[split])))
                for i, d in sorted(picked):
                    w.writerow([f"{i:06d}", d, split, folds[d]])


def fetch(root, frames):
    for key in ARCHIVES:
        (root/key).mkdir(parents=True, exist_ok=True)
    for key in ("label_2", "calib"):
        if all((root/key/f"{i}.txt").exists() for i in frames):
            continue
        with archive(ARCHIVES[key]) as z:
            for i in frames:
                member = f"{i}.txt"
                (root/key/member).write_bytes(z.read(f"training/{key}/{member}"))
    downloads = []
    for key, ext in (("image_2", "png"), ("velodyne", "bin")):
        missing = [i for i in frames if not (root/key/f"{i}.{ext}").exists()]
        if missing:
            remote = RemoteZip(BASE + ARCHIVES[key])
            with ZipFile(remote) as z:
                for i in missing:
                    filename = f"{i}.{ext}"
                    downloads.append((remote, z.getinfo(f"training/{key}/{filename}"), root/key/filename))
    def save(job):
        remote, info, dst = job
        dst.write_bytes(remote.member(info))
    random.Random(2026).shuffle(downloads)
    with ThreadPoolExecutor(max_workers=28) as pool:
        jobs = [pool.submit(save, item) for item in downloads]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()
            if i % 40 == 0 or i == len(downloads):
                print(f"{i}/{len(downloads)}", flush=True)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--manifest", type=Path, default=ROOT/"split.csv")
    p.add_argument("--data", type=Path, default=ROOT/"data")
    args = p.parse_args()
    if not args.demo and not args.manifest.exists():
        manifest(args.manifest)
    ids = ["000000"] if args.demo else [row["frame"] for row in csv.DictReader(args.manifest.open())]
    fetch(args.data, ids)
