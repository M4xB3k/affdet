"""Draw predicted bounding boxes + class names on scene images.

Loads trained detector weights, runs inference on a few images, and writes
annotated copies (boxes labelled with class name + confidence) — the visual
proof that the detector localises + names objects. Uses Ultralytics' built-in
plotter.

    python -m affdet.detect.annotate \
        --weights /scratch/$USER/affdet/detect_runs/yolo11s_run1/weights/best.pt \
        --images  /scratch/$USER/affdet/detect_dataset/images/val \
        --out     /scratch/$USER/affdet/annotated --n 6 --conf 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="trained best.pt")
    p.add_argument("--images", required=True, help="dir of images to annotate")
    p.add_argument("--out", required=True, help="output dir for annotated images")
    p.add_argument("--n", type=int, default=6, help="how many images to annotate")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = p.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(
        q for q in Path(args.images).iterdir()
        if q.suffix.lower() in (".png", ".jpg", ".jpeg")
    )[: args.n]
    if not imgs:
        raise SystemExit(f"no images under {args.images}")

    model = YOLO(args.weights)
    for img in imgs:
        # save=False; we grab the plotted array and write it ourselves so the
        # filename matches the source scene.
        res = model.predict(str(img), conf=args.conf, verbose=False)[0]
        plotted = res.plot()  # BGR ndarray with boxes + "class conf" labels
        import cv2
        dst = out / f"{img.stem}_pred.png"
        cv2.imwrite(str(dst), plotted)
        names = [res.names[int(c)] for c in res.boxes.cls.tolist()] if res.boxes is not None else []
        print(f"[annotate] {img.name}: {len(names)} boxes -> {dst.name}  ({', '.join(names)})")

    print(f"[annotate] wrote {len(imgs)} annotated images to {out}")


if __name__ == "__main__":
    main()
