"""VLM pilot: ask GPT-6-sol (Azure) the per-object questions on face-labelled renders.

For each object, one 2x2 image (front, right, two views from above) with the
box faces named. The model answers a fixed JSON schema, `--repeats` times.
Two kinds of questions: non-metric ones the VLM is meant to answer (handle,
materials, fragile, body cross-section, top shape) and three that the
geometry labeller already answers (stands upside down, which sides it lies on,
top can hold something). Those three show whether the VLM can be trusted on
near-metric questions.

    AZURE_TOKEN=... python scripts/vlm_face_pilot.py run   # renders + queries -> data/vlm_pilot/
    python scripts/vlm_face_pilot.py score                 # -> docs/objaverse/vlm_pilot_gpt6sol.{md,csv}
"""
import argparse
import base64
import collections
import concurrent.futures as cf
import csv
import io
import json
import os
import sys
import time
import urllib.request
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://personale.services.ai.azure.com").rstrip("/")
MODEL = os.environ.get("AZURE_DEPLOYMENT", "gpt-6-sol")
OUT = "data/vlm_pilot"
SET = "docs/objaverse/salt_shaker_set.csv"
GEOM = "docs/objaverse/salt_shaker_face_labels.csv"

# 20 objects chosen for variety: flat / domed caps, round / faceted / square bodies,
# mills, cranks, jars, glass, and the four smooth-shaded review cases.
PILOT = [
    "c442c91673a24b3798065d76fda3b4e0", "5ba5959ecc044c5fa44dded4c75ea25a", "9336ba99aeea42eaa4837567ab70af03",
    "d00ef0263b6b47378c6904801ce84adf", "b1e7f11f513649058b4de8567f581d43", "05308bf4da874a63b040399dca39489b",
    "77a7cef9019c4a50b982e315b38a90df", "d8e2236b427c4539a496f152bbc65d4d", "e0b380d2819249ecbb75c2552bedd0a1",
    "1e74c88055b748cbbef0c92a1d02a4c2", "12a6366c033245eb97551a615317c12e", "c70225db3dd34e8ba2f53b13bf93411a",
    "cf8640d4e4624bb4940850631ee6bb1c", "23a809f734f24ff6b817b77bd9bf0541", "a121f6c17bdd4818970cd642e74c2fd2",
    "e07a892b7ea3497ebe98f5d9ea30abc7", "88067fa666d746928bc59af924eb62fe", "b8a406f85d84423e925b6ebf804bf63b",
    "9e4437be65e94d2e81a7161a48117c62", "9ef6c018a8ce4d41824ec40b9870e412_0",
]
VIEWS = {"front": (-90, 0), "right side": (0, 0),
         "from above, front-right": (-45, 35), "from above, back-left": (135, 35)}
FACES = ["on", "under", "front", "back", "left", "right"]
SIDES = ["front", "back", "left", "right"]
MATERIALS = ["glass", "ceramic", "plastic", "metal", "wood", "stone", "other"]

PROMPT = """You see one kitchen object (a salt/pepper shaker, pepper mill or spice jar) \
rendered from four viewpoints on a 1 cm grid. Red tags name the faces of its \
bounding box: "on" (top), "under" (bottom), front, back, left, right. The object \
is shown standing upright. Answer about this specific object, as a robot would \
need to know it.

- has_handle: is there a part sticking out that a gripper could hold like a \
handle (a handle, a crank arm)? handle_face: the box face it sticks out of, or none.
- body_material, cap_material: main material of the body and of the cap/lid/top \
part (none if there is no separate cap).
- fragile: would it likely break if dropped on a hard floor?
- body_cross_section: shape of the body seen from above.
- top_shape: shape of the top surface.
- stands_upside_down: would it stay put if placed with the "on" face down on a table?
- rests_on_sides: which of front/back/left/right it would stay put on if laid on \
that face on a table (without rolling or tipping over). Empty list if none.
- top_can_hold: could a small flat item, such as a coaster or a lid, be placed \
and stay on its "on" face while it stands upright?"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["has_handle", "handle_face", "body_material", "cap_material", "fragile",
                 "body_cross_section", "top_shape", "stands_upside_down", "rests_on_sides", "top_can_hold"],
    "properties": {
        "has_handle": {"type": "boolean"},
        "handle_face": {"type": "string", "enum": FACES + ["none"]},
        "body_material": {"type": "string", "enum": MATERIALS},
        "cap_material": {"type": "string", "enum": MATERIALS + ["none"]},
        "fragile": {"type": "boolean"},
        "body_cross_section": {"type": "string", "enum": ["round", "faceted_polygon", "square_or_rectangle", "irregular"]},
        "top_shape": {"type": "string", "enum": ["flat", "domed", "pointed", "knob_or_finial", "other"]},
        "stands_upside_down": {"type": "boolean"},
        "rests_on_sides": {"type": "array", "items": {"type": "string", "enum": SIDES}},
        "top_can_hold": {"type": "boolean"},
    },
}


def object_rows():
    rows = {}
    for r in csv.DictReader(open(SET)):
        rows[r["object_uid"]] = r
    return rows


def render(oid, rows):
    from PIL import Image
    from affdet.geometry.prepare import prepare
    from affdet.geometry.render_check import face_labelled_views
    uid, _, half = oid.partition("_")
    r = rows[uid]
    scene, _ = prepare(f"data/objaverse/glbs/{uid}.glb", float(r["target_height_cm"]) / 100,
                       r.get("up_axis") or "+y", int(half) if half else None)
    ims = face_labelled_views(scene, VIEWS, 384)
    grid = Image.new("RGB", (768, 768), "white")
    for k, im in enumerate(ims.values()):
        grid.paste(im, ((k % 2) * 384, (k // 2) * 384))
    path = f"{OUT}/images/{oid}.jpg"
    grid.save(path, quality=90)
    return path


def ask(image_path):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    body = {
        "model": MODEL,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "object_attributes", "schema": SCHEMA, "strict": True}},
        "max_output_tokens": 8000,
    }
    req = urllib.request.Request(f"{ENDPOINT}/openai/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "api-key": os.environ["AZURE_TOKEN"]})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                d = json.load(resp)
            text = next(c["text"] for o in d["output"] if o.get("type") == "message" for c in o["content"]
                        if c.get("type") == "output_text")
            return json.loads(text), d.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 ** (attempt + 2))
                continue
            raise RuntimeError(e.read().decode()[:500])


def run(repeats, workers):
    os.makedirs(f"{OUT}/images", exist_ok=True)
    rows = object_rows()
    done = set()
    raw = f"{OUT}/raw_{MODEL}.jsonl"
    if os.path.exists(raw):
        done = {(d["object_id"], d["repeat"]) for d in map(json.loads, open(raw))}
    images = {oid: (f"{OUT}/images/{oid}.jpg" if os.path.exists(f"{OUT}/images/{oid}.jpg") else render(oid, rows))
              for oid in PILOT}
    jobs = [(oid, k) for oid in PILOT for k in range(repeats) if (oid, k) not in done]

    def one(job):
        oid, k = job
        ans, usage = ask(images[oid])
        return {"object_id": oid, "repeat": k, "model": MODEL, "answer": ans, "usage": usage}

    with open(raw, "a") as fh, cf.ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(rec["object_id"][:10], rec["repeat"], rec["usage"].get("total_tokens"), json.dumps(rec["answer"])[:150])


def score():
    raw = f"{OUT}/raw_{MODEL}.jsonl"
    recs = collections.defaultdict(list)
    tokens = collections.Counter()
    for d in map(json.loads, open(raw)):
        recs[d["object_id"]].append(d["answer"])
        for k in ("input_tokens", "output_tokens"):
            tokens[k] += d["usage"].get(k, 0)
    geom = {r["object_id"]: r for r in csv.DictReader(open(GEOM))}
    keys = list(SCHEMA["properties"])
    consistent = collections.Counter()
    agree = collections.Counter()
    n_side = 0
    out_rows = []
    for oid, answers in recs.items():
        g = geom[oid]
        row = {"object_id": oid, "tier": g["tier"], "repeats": len(answers)}
        for k in keys:
            vals = [json.dumps(sorted(a[k]) if isinstance(a[k], list) else a[k]) for a in answers]
            maj, cnt = collections.Counter(vals).most_common(1)[0]
            consistent[k] += cnt == len(vals)
            row[k] = json.loads(maj)
            row[f"{k}_votes"] = f"{cnt}/{len(vals)}"
        row["geom_supportable_on"] = int(g["supportable_on"])
        row["geom_sides"] = [s for s in SIDES if g[f"supportable_{s}"] == "1"]
        row["geom_supportive_on"] = int(g["supportive_on"])
        row["geom_review"] = g["review"]
        agree["stands_upside_down"] += row["stands_upside_down"] == bool(row["geom_supportable_on"])
        agree["top_can_hold"] += row["top_can_hold"] == bool(row["geom_supportive_on"])
        for s in SIDES:
            agree["rests_on_side (per face)"] += (s in row["rests_on_sides"]) == (s in row["geom_sides"])
            n_side += 1
        agree["lies on any side"] += bool(row["rests_on_sides"]) == bool(row["geom_sides"])
        out_rows.append(row)
    n = len(out_rows)
    with open("docs/objaverse/vlm_pilot_gpt6sol.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (" ".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    lines = [f"# VLM pilot: {MODEL} on {n} salt-shaker objects × {len(answers)} repeats", "",
             f"Tokens: {tokens['input_tokens']:,} in / {tokens['output_tokens']:,} out "
             f"(≈ ${tokens['input_tokens'] * 2e-6 + tokens['output_tokens'] * 10e-6:.2f} at $2 / $10 per M).", "",
             "## Consistency (all repeats give the same answer)", "", "| Question | Objects |", "|---|---|"]
    lines += [f"| {k} | {consistent[k]}/{n} |" for k in keys]
    lines += ["", "## Agreement of the majority answer with the geometry labeller", "", "| Question | Agree |", "|---|---|"]
    for k, v in agree.items():
        tot = n_side if "per face" in k else n
        lines.append(f"| {k} | {v}/{tot} ({100 * v / tot:.0f} %) |")
    lines += ["", "Per-object answers: `vlm_pilot_gpt6sol.csv`."]
    open("docs/objaverse/vlm_pilot_gpt6sol.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only-first", type=int, help="run just the first N objects (smoke test)")
    a = ap.parse_args()
    if a.only_first:
        PILOT[:] = PILOT[: a.only_first]
    run(a.repeats, a.workers) if a.cmd == "run" else score()
