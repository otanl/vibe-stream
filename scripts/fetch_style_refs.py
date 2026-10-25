"""Style reference images: public-domain artworks from the Art Institute of Chicago API (images CC0).

Searches the collection for each keyword below (paintings, prints, drawings and watercolours only, public domain only),
keeps up to --per_query new works per keyword, and saves an 843 px wide IIIF rendition of each as <out>/<id>.jpg with
one JSON line per work in <out>/refs.jsonl (id, title, artist, date, type, style, query, source URL). Already present
files are skipped, so the script can be re-run to extend the set.
Usage: python scripts/fetch_style_refs.py --out data/style_refs_aic
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "vibe-stream-research/0.1 (style references for an image-editing study; https://github.com/otanl/vibe-stream)"
API = "https://api.artic.edu/api/v1/artworks/search"
IIIF = "https://www.artic.edu/iiif/2/{}/full/843,/0/default.jpg"
TYPES = {"Painting", "Print", "Drawing and Watercolor"}
QUERIES = [
    "impressionism", "post-impressionism", "pointillism", "fauvism", "expressionism", "symbolism", "art nouveau",
    "romanticism", "realism", "baroque", "rococo", "renaissance", "medieval", "ukiyo-e", "woodblock print",
    "japanese screen", "chinese ink painting", "indian miniature", "persian", "watercolor landscape", "gouache",
    "pastel", "oil sketch", "etching", "engraving", "lithograph", "poster", "ink drawing", "charcoal drawing",
    "still life", "portrait", "seascape", "night scene", "winter landscape", "garden", "city street", "flowers",
    "abstract", "decorative pattern", "folk art",
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "AIC-User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per_query", type=int, default=12)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "refs.jsonl"
    seen = {json.loads(line)["id"] for line in meta_path.read_text(encoding="utf-8").splitlines()} if meta_path.exists() else set()
    fields = "id,title,artist_display,date_display,image_id,is_public_domain,artwork_type_title,style_title"
    with meta_path.open("a", encoding="utf-8") as meta:
        for q in QUERIES:
            kept = 0
            for page in (1, 2, 3):
                params = {"q": q, "query[term][is_public_domain]": "true", "fields": fields, "limit": 100, "page": page}
                try:
                    hits = json.loads(get(API + "?" + urllib.parse.urlencode(params)))["data"]
                except Exception as e:  # noqa: BLE001
                    print(f"[refs] {q} page {page}: {e}", flush=True)
                    break
                time.sleep(1.0)
                for h in hits:
                    if kept >= a.per_query:
                        break
                    if (h["id"] in seen or not h.get("image_id") or not h.get("is_public_domain")
                            or h.get("artwork_type_title") not in TYPES):
                        continue
                    try:
                        img = get(IIIF.format(h["image_id"]))
                    except Exception as e:  # noqa: BLE001
                        print(f"[refs] {h['id']}: {e}", flush=True)
                        continue
                    (out / f"{h['id']}.jpg").write_bytes(img)
                    seen.add(h["id"])
                    kept += 1
                    meta.write(json.dumps(dict(id=h["id"], title=h["title"], artist=h.get("artist_display"),
                                               date=h.get("date_display"), type=h["artwork_type_title"],
                                               style=h.get("style_title"), query=q,
                                               url=IIIF.format(h["image_id"])), ensure_ascii=False) + "\n")
                    meta.flush()
                    time.sleep(0.5)
                if kept >= a.per_query or len(hits) < 100:
                    break
            print(f"[refs] {q}: {kept} new, {len(seen)} total", flush=True)


if __name__ == "__main__":
    main()
