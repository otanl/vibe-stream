"""Pick one training target per pair from scripts/make_style_data.py's candidates, or none.

A candidate qualifies when it kept the photo's structure (edge agreement >= --min_edge) and content (CLIP similarity to
the photo >= --min_input; a reference copied into the picture lowers both), changed it (<= --max_input, and LPIPS to
the photo >= --min_change) and moved towards the reference (ref_gain >= --min_gain). CLIP similarity alone lets
near-copies of the photo through when a colour shift raises ref_gain; LPIPS (AlexNet, 256 px) is computed on the GPU
on the first run and cached in <data>/change.json. Among the qualifying candidates the one with the highest
ref_gain + --w_edge * edge wins.
Writes <data>/selected.jsonl (pair, content, ref, caption, target, way, scores), prints the yield and score
distributions, and saves two contact sheets (photo | reference | target) of selected and of rejected pairs.
Usage: python scripts/select_style_targets.py --data data/style_ft_768
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

S_CHANGE = 256


def tile(img, text, s=224):
    t = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (6 + 7 * len(text), 16), (0, 0, 0), -1)
    cv2.putText(t, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def square(path):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    s = min(h, w)
    return im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def change_scores(d, rows):
    """LPIPS between each photo's centre square and its candidates at S_CHANGE px, cached in <data>/change.json."""
    cache = d / "change.json"
    out = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [r for r in rows if r["pair"] not in out]
    if not todo:
        return out
    from concurrent.futures import ThreadPoolExecutor

    import lpips
    import torch

    def small(im):
        return cv2.resize(im, (S_CHANGE, S_CHANGE), interpolation=cv2.INTER_AREA)[:, :, ::-1]

    def load(r):
        ks = sorted(r["scores"])
        return r["pair"], ks, small(square(r["content"])), [small(cv2.imread(str(d / "pairs" / r["pair"] / f"{k}.jpg")))
                                                          for k in ks]

    def tensor(ims):
        return torch.from_numpy(np.ascontiguousarray(np.stack(ims))).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1

    net = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for pair, ks, photo, cands in ex.map(load, todo):
            v = net(tensor([photo] * len(cands)), tensor(cands)).flatten().tolist()
            out[pair] = {k: round(x, 4) for k, x in zip(ks, v)}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--min_edge", type=float, default=0.25)
    p.add_argument("--min_input", type=float, default=0.62)
    p.add_argument("--max_input", type=float, default=0.95)
    p.add_argument("--min_gain", type=float, default=0.05)
    p.add_argument("--min_change", type=float, default=0.25, help="LPIPS to the photo; 0 skips it")
    p.add_argument("--w_edge", type=float, default=0.3)
    p.add_argument("--sheet", type=int, default=24, help="rows per contact sheet")
    a = p.parse_args()
    d = Path(a.data)
    rows = [json.loads(line) for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if a.min_change > 0:
        change = change_scores(d, rows)
        for r in rows:
            for k, s in r["scores"].items():
                s["change"] = change[r["pair"]][k]
    sel, rej = [], []
    for r in rows:
        ok = {k: s for k, s in r["scores"].items() if s["edge"] >= a.min_edge and a.min_input <= s["input_sim"] <= a.max_input
              and s["ref_gain"] >= a.min_gain and s.get("change", 1.0) >= a.min_change}
        if not ok:
            rej.append(r)
            continue
        k = max(ok, key=lambda k: ok[k]["ref_gain"] + a.w_edge * ok[k]["edge"])
        sel.append(dict(pair=r["pair"], content=r["content"], ref=r["ref"], caption=r["caption"],
                        target=str(d / "pairs" / r["pair"] / f"{k}.jpg"), way=k[0], scores=ok[k]))
    with (d / "selected.jsonl").open("w", encoding="utf-8") as f:
        for s in sel:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    allc = [s for r in rows for s in r["scores"].values()]
    print(f"[select] {len(sel)} of {len(rows)} pairs have a target ({100 * len(sel) / max(len(rows), 1):.0f} %); "
          f"way A {sum(s['way'] == 'A' for s in sel)}, B {sum(s['way'] == 'B' for s in sel)}")
    keys = ["ref_gain", "input_sim", "edge"] + (["change"] if a.min_change > 0 else [])
    for name, xs in (("all candidates", allc), ("selected", [s["scores"] for s in sel])):
        if xs:
            q = np.percentile(np.array([[x[k] for k in keys] for x in xs]), [10, 50, 90], axis=0)
            print(f"[select] {name:<15} " + "  ".join(f"{k} {q[0, i]:.3f}/{q[1, i]:.3f}/{q[2, i]:.3f}" for i, k in enumerate(keys))
                  + "  (10/50/90th percentile)")
    refs_used = len({s["ref"] for s in sel})
    print(f"[select] references with at least one target: {refs_used} of {len({r['ref'] for r in rows})}")
    kind = {}  # reference path -> artwork type, from refs.jsonl next to the references when there is one
    for ref_dir in {str(Path(r["ref"]).parent) for r in rows}:
        meta = Path(ref_dir) / "refs.jsonl"
        if meta.exists():
            for line in meta.read_text(encoding="utf-8").splitlines():
                m = json.loads(line)
                kind[str(Path(ref_dir) / f"{m['id']}.jpg")] = m.get("type") or "?"
    got = {s["pair"] for s in sel}
    for t in sorted({kind.get(str(Path(r["ref"])), "other") for r in rows}):
        prs = [r for r in rows if kind.get(str(Path(r["ref"])), "other") == t]
        print(f"[select]   {t:<14} {sum(r['pair'] in got for r in prs):5d} of {len(prs):5d} pairs")

    rng = random.Random(0)
    for name, items in (("selected", sel), ("rejected", rej)):
        pick = rng.sample(items, min(a.sheet, len(items)))
        if not pick:
            continue
        lines = []
        for it in pick:
            t = [tile(square(it["content"]), "photo"), tile(cv2.imread(it["ref"]), "ref " + Path(it["ref"]).stem)]
            if name == "selected":
                s = it["scores"]
                t.append(tile(cv2.imread(it["target"]), f"{Path(it['target']).stem} g{s['ref_gain']:.2f} "
                                                        f"in{s['input_sim']:.2f} e{s['edge']:.2f}"))
            else:
                k = max(it["scores"], key=lambda k: it["scores"][k]["ref_gain"])
                s = it["scores"][k]
                t.append(tile(cv2.imread(str(d / "pairs" / it["pair"] / f"{k}.jpg")),
                              f"{k} g{s['ref_gain']:.2f} in{s['input_sim']:.2f} e{s['edge']:.2f}"))
            lines.append(np.concatenate(t, 1))
        half = (len(lines) + 1) // 2
        cols = [np.concatenate(lines[:half], 0), np.concatenate(lines[half:] + [np.zeros_like(lines[0])] * (2 * half - len(lines)), 0)]
        cv2.imwrite(str(d / f"sheet_{name}.jpg"), np.concatenate(cols, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[select] wrote {d / 'selected.jsonl'} and sheets")


if __name__ == "__main__":
    main()
