#!/usr/bin/env python3
"""Split the OmniObject3D review sheets into keep/drop montages per category.

Reads data/omni/sheets/<cat>.png (the only local copy of the renders), recovers
the burned-in tile_NNN labels by template-matching Pillow-rendered candidates
(same DejaVu Sans 14px the sheet generator used; no OCR dependency), then
re-montages tiles into:
    data/omni/cull_review/<cat>__keep.png   (everything not dropped; verify-flagged
                                             tiles get an orange border + '?')
    data/omni/cull_review/<cat>__drop.png   (suggested drops, red border; only
                                             written when the category has drops)
Drop list comes from scripts/omni/drop_suggested.txt; verify flags are inlined
below from tools/docs/omni_cull_review.md. Also writes tile_manifest.json with
the OCR'd cell->tile mapping for later mesh-extraction steps.

Sheet geometry must match omni_make_sheet.sh: 6 cols, 200x150 tiles, 6px pad,
25px label band => 212x187 cells.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
SHEETS = REPO / "data/omni/sheets"
OUT = REPO / "data/omni/cull_review"
DROP_FILE = Path(__file__).parent / "drop_suggested.txt"

COLS = 6
TILE_W, TILE_H = 200, 150
PAD = 6
LABEL_H = 25
CELL_W = TILE_W + 2 * PAD     # 212
CELL_H = TILE_H + LABEL_H + 2 * PAD  # 187

# Verify flags from tools/docs/omni_cull_review.md (suspicious at sheet
# resolution -- user should eyeball). china's 023/024/026 are the "optionally
# keep these vessels" tiles inside an all-drop category.
VERIFY = {
    "knife": ["001"],
    "fork": ["001", "002"],
    "spanner": ["003"],
    "teapot": ["027", "030", "031", "032"],
    "red_wine_glass": ["001", "002", "014"],
    "vase": ["007", "013"],
    "china": ["023", "024", "026"],
    "banana": ["036", "039"],
    "carrot": ["025"],
    "cauliflower": ["002"],
    "cheese": ["026", "038", "042", "043", "044", "045", "046", "047", "048", "049"],
    "cherry": ["010", "013"],
    "cucumber": ["030"],
    "durian": ["008", "014", "017", "019"],
    "hamburger": ["027"],
    "lemon": ["033"],
    "litchi": ["025", "026", "029", "031", "032"],
    "orange": ["025", "026", "029", "031"],
    "sausage": ["018", "024"],
    "watermelon": ["037", "042", "044"],
}

ORANGE = (235, 120, 0)
RED = (210, 40, 40)


def load_font(size):
    for p in [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def parse_drops():
    drops = {}
    for line in DROP_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cat, nums = line.split(":", 1)
        drops[cat.strip()] = nums.split()
    return drops


# Label box big enough for "tile_NNN" at 14px incl. the '_' descender.
LBL_W, LBL_H_BOX = 70, 22
MSE_FLAG = 0.010  # matches above this are suspicious -> flagged for eyeballing


def build_candidates(font, max_num=400):
    """Render 'tile_NNN' exactly like omni_make_sheet.sh did; return (nums, stack)."""
    stack = np.empty((max_num, LBL_H_BOX, LBL_W), dtype=np.float32)
    for n in range(max_num):
        im = Image.new("L", (LBL_W, LBL_H_BOX), 255)
        ImageDraw.Draw(im).text((0, 0), f"tile_{n:03d}", fill=0, font=font)
        stack[n] = np.asarray(im, dtype=np.float32) / 255.0
    return stack


def cell_has_label(strip):
    """A rendered tile always gets a black label under it."""
    return min(strip.convert("L").getextrema()) < 100


def match_label(gray, x0, y0, cand):
    """MSE-match the label at sheet position (x0,y0) over ±1px shifts."""
    best = (None, 1e9)
    for dy in (0, -1, 1):
        for dx in (0, -1, 1):
            crop = gray[y0 + dy:y0 + dy + LBL_H_BOX, x0 + dx:x0 + dx + LBL_W]
            if crop.shape != (LBL_H_BOX, LBL_W):
                continue
            mse = ((cand - crop[None]) ** 2).mean(axis=(1, 2))
            n = int(mse.argmin())
            if mse[n] < best[1]:
                best = (n, float(mse[n]))
        if best[1] < 1e-4:  # exact-enough hit, skip remaining shifts
            break
    return best


def read_sheet(path, cand):
    """Return ordered list of (tile_number, PIL tile image)."""
    sheet = Image.open(path).convert("RGB")
    gray = np.asarray(sheet.convert("L"), dtype=np.float32) / 255.0
    rows = sheet.height // CELL_H
    tiles = []
    problems = []
    for row in range(rows):
        for col in range(COLS):
            x0 = col * CELL_W + PAD
            ly = row * CELL_H + PAD + TILE_H + 2
            strip = sheet.crop((x0, ly, x0 + LBL_W, min(ly + LBL_H_BOX, sheet.height)))
            if not cell_has_label(strip):
                continue
            num, mse = match_label(gray, x0, ly, cand)
            if num is None or mse > MSE_FLAG:
                problems.append(f"row {row} col {col}: weak label match "
                                f"(best=tile_{num:03d}, mse={mse:.4f})")
            ty = row * CELL_H + PAD
            tiles.append((f"{num:03d}", sheet.crop((x0, ty, x0 + TILE_W, ty + TILE_H))))
    # checksum: numbers must be strictly increasing
    seq = [int(n) for n, _ in tiles]
    if seq != sorted(set(seq)):
        problems.append(f"non-monotonic label sequence: {[n for n, _ in tiles]}")
    return tiles, problems


def montage(entries, title, out_path, font, title_font):
    """entries: list of (num, tile_img, status) with status in keep/verify/drop."""
    n = len(entries)
    cols = min(COLS, max(1, n))
    rows = (n + cols - 1) // cols
    header = 34
    W = cols * CELL_W
    H = header + rows * CELL_H
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 7), title, fill="black", font=title_font)
    for i, (num, tile, status) in enumerate(entries):
        r, c = divmod(i, cols)
        x = c * CELL_W + PAD
        y = header + r * CELL_H + PAD
        canvas.paste(tile, (x, y))
        if status == "verify":
            d.rectangle([x - 2, y - 2, x + TILE_W + 1, y + TILE_H + 1], outline=ORANGE, width=3)
        elif status == "drop":
            d.rectangle([x - 2, y - 2, x + TILE_W + 1, y + TILE_H + 1], outline=RED, width=3)
        label = f"tile_{num}" + (" ?" if status == "verify" else "")
        color = {"keep": "black", "verify": ORANGE, "drop": RED}[status]
        d.text((x, y + TILE_H + 2), label, fill=color, font=font)
    canvas.save(out_path)


def main():
    only = sys.argv[1:] or None
    OUT.mkdir(parents=True, exist_ok=True)
    drops = parse_drops()
    font = load_font(14)
    title_font = load_font(18)
    cand = build_candidates(font)
    manifest = {}
    for sheet_path in sorted(SHEETS.glob("*.png")):
        cat = sheet_path.stem
        if only and cat not in only:
            continue
        tiles, problems = read_sheet(sheet_path, cand)
        drop_set = set(drops.get(cat, []))
        verify_set = set(VERIFY.get(cat, []))
        missing = (drop_set | verify_set) - {n for n, _ in tiles}
        if missing:
            problems.append(f"flagged tiles not found on sheet: {sorted(missing)}")
        keep_entries, drop_entries = [], []
        for num, img in tiles:
            if num in drop_set:
                drop_entries.append((num, img, "verify" if num in verify_set else "drop"))
            else:
                keep_entries.append((num, img, "verify" if num in verify_set else "keep"))
        montage(keep_entries, f"{cat} - KEEP ({len(keep_entries)})",
                OUT / f"{cat}__keep.png", font, title_font)
        if drop_entries:
            montage(drop_entries, f"{cat} - DROP ({len(drop_entries)})",
                    OUT / f"{cat}__drop.png", font, title_font)
        manifest[cat] = {
            "tiles": [n for n, _ in tiles],
            "keep": [n for n, _, _ in keep_entries],
            "drop": [n for n, _, _ in drop_entries],
            "verify": sorted(verify_set & {n for n, _ in tiles}),
        }
        status = "OK" if not problems else "CHECK: " + "; ".join(problems)
        print(f"{cat:16s} tiles={len(tiles):3d} keep={len(keep_entries):3d} "
              f"drop={len(drop_entries):2d} verify={len(verify_set & {n for n,_ in tiles}):2d}  {status}")
    (OUT / "tile_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {OUT}/<cat>__keep.png, __drop.png, tile_manifest.json")


if __name__ == "__main__":
    main()
