from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen, Request


YCB_OBJECTS = {
    "024_bowl": "bowl",
    "025_mug": "mug",
    "019_pitcher_base": "pitcher_base",
    "020_pitcher_lid": "pitcher_lid",
    "004_sugar_box": "sugar_box",
    "006_mustard_bottle": "mustard_bottle",
    "032_knife": "knife",
    "031_spoon": "spoon",
    "033_spatula": "spatula",
    "029_plate": "plate",
    "027_skillet": "skillet",
    "028_skillet_lid": "skillet_lid",
    "011_banana": "banana",
    "013_apple": "apple",
    "014_lemon": "lemon",
    "015_peach": "peach",
    "016_pear": "pear",
    "017_orange": "orange",
    "018_plum": "plum",
    "012_strawberry": "strawberry",
}

SUPPLEMENT_OBJECTS = [
    "cutting_board",
    "salt_shaker",
    "pot",
    "pot_lid",
    "tray",
]


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    base = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(base)):
            raise RuntimeError("Unsafe tar extraction path detected.")
    tar.extractall(path)


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "affdet-downloader"})
    try:
        with urlopen(req, timeout=30) as resp, dest.open("wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise


def _find_obj(root: Path) -> Path:
    candidates = [p for p in root.rglob("*.obj") if "google_16k" in str(p) and "textured" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError("No google_16k textured .obj found in extracted archive.")
    return candidates[0]


def _copy_mesh_bundle(src_obj: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir = src_obj.parent
    for file in src_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, dest_dir / file.name)
    dest_obj = dest_dir / "model.obj"
    copied_obj = dest_dir / src_obj.name
    copied_obj.replace(dest_obj)

    lines = dest_obj.read_text().splitlines()
    mtllib = None
    updated = []
    for line in lines:
        if line.lower().startswith("mtllib "):
            mtllib = line.split(None, 1)[1].strip()
            updated.append("mtllib model.mtl")
        else:
            updated.append(line)
    if mtllib:
        src_mtl = dest_dir / mtllib
        if src_mtl.exists():
            src_mtl.replace(dest_dir / "model.mtl")
        dest_obj.write_text("\n".join(updated) + "\n")


GOOGLE_BASE = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YCB meshes.")
    parser.add_argument("--out", type=Path, default=Path("data/meshes"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--objects", nargs="*", default=None,
                        help="Optional subset of YCB object keys (e.g. 024_bowl).")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    chosen = YCB_OBJECTS if args.objects is None else {
        key: YCB_OBJECTS[key] for key in args.objects
    }

    for ycb_id, class_name in chosen.items():
        dest_dir = args.out / class_name
        if dest_dir.exists() and not args.overwrite:
            print(f"[skip] {class_name} already exists at {dest_dir}")
            continue
        if dest_dir.exists():
            shutil.rmtree(dest_dir)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                archive = tmp_path / f"{ycb_id}.tgz"

                google_url = f"{GOOGLE_BASE}/{ycb_id}_google_16k.tgz"
                print(f"[download] {ycb_id} -> {google_url}")
                _download(google_url, archive)

                with tarfile.open(archive, "r:gz") as tar:
                    _safe_extract(tar, tmp_path)
                obj_path = _find_obj(tmp_path)
                _copy_mesh_bundle(obj_path, dest_dir)
                print(f"[ok] {class_name} -> {dest_dir}")
        except Exception as e:
            print(f"[error] {ycb_id}: {e} — skipping")

    for supplement in SUPPLEMENT_OBJECTS:
        expected = args.out / supplement / "model.obj"
        if not expected.exists():
            print(f"[todo] supplement mesh missing for {supplement}: {expected}")


if __name__ == "__main__":
    main()
