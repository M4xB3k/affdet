"""Run the whole-photo VLM baseline over every gt2_*.json'd real photo and
score it, end to end: one VLM call per photo (raw image, no detector help),
timed, scored against gt2 via affdet.eval.baseline_vlm_score.

Mirrors what our own pipeline does on the same 19 photos (see
timed_batch_infer.py) so both sides of the "VLM quality at a fraction of the
cost" comparison are measured the same way: per-photo wall-clock time plus
P/R/F1 against the same ground truth.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from affdet.eval.baseline_vlm_prompts import (
    build_baseline_system_prompt, build_baseline_user_prompt,
    parse_baseline_response,
)
from affdet.eval.baseline_vlm_score import score_baseline_response


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _make_client():
    import os
    from openai import OpenAI

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_endpoint and azure_key:
        base_url = azure_endpoint.rstrip("/") + "/openai/v1/"
        return OpenAI(base_url=base_url, api_key=azure_key)
    return OpenAI()


def _call_vlm(client, model: str, photo_path: Path) -> tuple[dict, float]:
    b64 = _encode_image(photo_path)
    start = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_baseline_system_prompt()},
            {"role": "user", "content": [
                {"type": "text", "text": build_baseline_user_prompt()},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        # GPT-5-series reasoning models reject max_tokens/temperature (fixed
        # at 1). "minimal" reasoning was found to misclassify knife types in
        # the training-label pipeline (affdet/label/vlm.py) -- this call is
        # far heavier (find every object AND judge 6 predicates x every
        # ordered pair in one shot), so use "low" at minimum and a large
        # completion budget; a real photo scene (~5 objects = ~20 ordered
        # pairs x 6 predicates = ~120 judgments) needs much more room than a
        # single-pair label call.
        max_completion_tokens=16000,
        reasoning_effort="low",
    )
    elapsed = time.monotonic() - start
    parsed = parse_baseline_response(resp.choices[0].message.content)
    return parsed, elapsed


def run_baseline_eval(gt_dir: Path, photos_dir: Path, client=None,
                       model: str = "gpt-4o") -> dict:
    """gt_dir contains gt2_<STEM>.json files; photos_dir contains
    <STEM>.JPG/.jpg matching each gt2 file's "image" field's stem."""
    gt_dir = Path(gt_dir); photos_dir = Path(photos_dir)
    client = client if client is not None else _make_client()

    per_photo: dict[str, dict] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for gt_path in sorted(gt_dir.glob("gt2_*.json")):
        stem = gt_path.stem.removeprefix("gt2_")
        gt = json.loads(gt_path.read_text())

        photo_path = photos_dir / gt["image"]
        if not photo_path.exists():
            matches = list(photos_dir.glob(f"{stem}.*"))
            if not matches:
                continue
            photo_path = matches[0]

        response, elapsed = _call_vlm(client, model, photo_path)
        scored = score_baseline_response(response, gt)
        scored["elapsed_seconds"] = elapsed
        per_photo[stem] = scored

        for k in ("tp", "fp", "fn", "tn"):
            totals[k] += scored[k]

    return {"per_photo": per_photo, "totals": totals}
