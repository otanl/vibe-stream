"""Content photos for style data: a random sample of the Open Images V4 validation set (images listed as CC BY 2.0).

Reads the official validation image list (with licence, author and rotation per image), draws --count images without
rotation (fixed seed), downloads them from the public open-images-dataset bucket and saves each as <out>/<ImageID>.jpg,
resized so the short side is 768 px. <out>/images.jsonl keeps the attribution (author, licence, original URL). Already
present files are skipped.
Usage: python scripts/fetch_content_images.py --out data/content_openimages --count 2000
"""

import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

from PIL import Image

UA = "vibe-stream-research/0.1 (content photos for an image-editing study; https://github.com/otanl/vibe-stream)"
LIST = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
FILE = "https://s3.amazonaws.com/open-images-dataset/validation/{}.jpg"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(io.StringIO(get(LIST).decode("utf-8")))
            if r.get("Rotation", "0") in ("", "0", "0.0") and "creativecommons.org/licenses/by/2.0" in r.get("License", "")]
    random.Random(a.seed).shuffle(rows)
    print(f"[content] {len(rows)} unrotated CC BY 2.0 images listed", flush=True)
    meta_path = out / "images.jsonl"
    done = {json.loads(line)["id"] for line in meta_path.read_text(encoding="utf-8").splitlines()} if meta_path.exists() else set()
    with meta_path.open("a", encoding="utf-8") as meta:
        for r in rows:
            if len(done) >= a.count:
                break
            if r["ImageID"] in done:
                continue
            try:
                img = Image.open(io.BytesIO(get(FILE.format(r["ImageID"])))).convert("RGB")
            except Exception as e:  # noqa: BLE001
                print(f"[content] {r['ImageID']}: {e}", flush=True)
                continue
            if min(img.size) < 512:
                continue
            s = 768 / min(img.size)
            img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC).save(out / f"{r['ImageID']}.jpg", quality=95)
            done.add(r["ImageID"])
            meta.write(json.dumps(dict(id=r["ImageID"], author=r.get("Author"), license=r.get("License"),
                                       url=r.get("OriginalURL"), title=r.get("Title")), ensure_ascii=False) + "\n")
            meta.flush()
            if len(done) % 200 == 0:
                print(f"[content] {len(done)} images", flush=True)
    print(f"[content] done: {len(done)} images", flush=True)


if __name__ == "__main__":
    main()
