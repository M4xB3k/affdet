"""E7 scene-level VLM arm: one call per scene, numbered boxes, full table back.

Distinct from `vlm.py`, which labels one pair per call from a composed
two-object render. Here the VLM sees the real multi-object scene the student
sees, and answers every candidate judgement for it at once — the same output
shape the student produces, at one call per scene rather than one per pair.

Object identity travels through the DRAWN BOX NUMBER, not through the class
name. 14 % of the eval scenes hold two objects of one class, and a text
reference ("the knife") cannot address them; "[3]" can, and maps back to
`knife_028` through a table the caller holds. The VLM is never asked for pixel
coordinates — it reads an id painted onto the image, exactly as the student
reads an obj_id attached to the same box.

Arms are defined in `scene_prompts.py`. Arm "e" draws no boxes at all and is
therefore the one arm where identity is ambiguous; it runs on the
duplicate-class-free subset only (see data/eval/E7_END_TO_END_SETUP.md).
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from affdet.label.scene_prompts import (
    ARMS, build_scene_system_prompt, build_scene_user_prompt,
)

#: High-contrast, colour-blind-safe cycle. Box colour carries no meaning --
#: it only separates adjacent boxes -- but reusing one colour for every box
#: makes overlapping objects hard to tell apart.
_COLOURS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (170, 110, 40), (128, 0, 0),
]


def draw_numbered_boxes(image_path: Path | str, bboxes: dict[str, list[float]],
                        order: list[str], out_path: Path | str,
                        label_classes: bool = True) -> dict[str, int]:
    """Draw one numbered box per object; return obj_id -> box number.

    `order` fixes the numbering so it is deterministic across arms and reruns —
    a scene must get the same numbers whether it is rendered for gplus or for
    g, or the two arms' answers cannot be compared row by row.
    """
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=17)
    mapping: dict[str, int] = {}

    for i, obj_id in enumerate(order):
        if obj_id not in bboxes:
            continue
        n = i + 1
        mapping[obj_id] = n
        x0, y0, x1, y1 = (float(v) for v in bboxes[obj_id])
        colour = _COLOURS[i % len(_COLOURS)]
        d.rectangle([x0, y0, x1, y1], outline=colour, width=3)

        text = f"[{n}] {obj_id.rsplit('_', 1)[0]}" if label_classes else f"[{n}]"
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        # Label above the box, or inside it when the box touches the top edge.
        ly = y0 - th - 6 if y0 - th - 6 >= 0 else y0 + 2
        d.rectangle([x0, ly, x0 + tw + 8, ly + th + 6], fill=colour)
        d.text((x0 + 4, ly + 2), text, font=font, fill=(255, 255, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return mapping


def _encode(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def parse_response(content: str, questions: list[dict],
                   arm: str) -> dict[tuple, float]:
    """Map the model's JSON table back onto the requested questions.

    Returns {(actor_obj_id, recipient_obj_id, predicate): score}. Rows that
    reference a box or pair that was never asked about are DROPPED rather than
    guessed at: a hallucinated row is not evidence, and silently coercing it
    would put a fabricated score into the results.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    data = json.loads(text)
    rows = data.get("judgements", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"expected a list of judgements, got {type(rows)}")

    if arm == "e":
        want = {(q["actor_label"], q["recipient_label"], q["predicate"]): q
                for q in questions}
    else:
        want = {(q["actor_box"], q["recipient_box"], q["predicate"]): q
                for q in questions}

    out: dict[tuple, float] = {}
    for row in rows:
        if not isinstance(row, dict) or "score" not in row:
            continue
        try:
            if arm == "e":
                key = (row["actor"], row["recipient"], row["predicate"])
            else:
                key = (int(row["actor_box"]), int(row["recipient_box"]),
                       row["predicate"])
        except (KeyError, TypeError, ValueError):
            continue
        q = want.get(key)
        if q is None:
            continue
        score = float(row["score"])
        if not 0.0 <= score <= 10.0:
            continue
        out[(q["actor"], q["recipient"], q["predicate"])] = score
    return out


@dataclass
class SceneLabeler:
    """One VLM call per scene. See module docstring."""

    model: str = "gpt-5-mini"
    arm: str = "gplus"
    max_retries: int = 4
    client: Any = None
    reasoning_effort: str = "low"
    #: Opt-in: restore the pairwise labeller's committed-judgement fields
    #: (canCut "type", canContain "enclosure", canSupport "size_note"), which
    #: the scene adaptation dropped. Off by default so the committed E7
    #: numbers stay reproducible.
    fields: bool = False
    total_prompt_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {ARMS}")

    @staticmethod
    def make_client():
        """Azure only. Falling back to a plain OpenAI() client would send
        production-shaped calls to a different provider and return numbers that
        look comparable to the corpus labels but are not."""
        import os
        from openai import OpenAI

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not key:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set; "
                "refusing to fall back to a non-Azure endpoint")
        return OpenAI(base_url=endpoint.rstrip("/") + "/openai/v1/",
                      api_key=key)

    def label_scene(self, client, image_path: Path, questions: list[dict],
                    vocabulary: Iterable[str],
                    predicates: list[str]) -> dict[tuple, float]:
        system = build_scene_system_prompt(predicates, self.arm)
        user = build_scene_user_prompt(questions, list(vocabulary), self.arm,
                                       fields=self.fields)
        content = [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{_encode(image_path)}",
                "detail": "high"}},
        ]
        last: Exception | None = None
        for _ in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": content}],
                    # Same knobs as the pair labeller: GPT-5-series reject
                    # temperature/max_tokens, and "minimal" reasoning was
                    # retired after producing confident wrong scores on four
                    # of five predicates (see vlm.py).
                    max_completion_tokens=4000,
                    reasoning_effort=self.reasoning_effort,
                )
                if resp.usage is not None:
                    self.total_prompt_tokens += resp.usage.prompt_tokens
                    self.total_completion_tokens += resp.usage.completion_tokens
                text = resp.choices[0].message.content
                if not text:
                    raise ValueError(
                        "empty response "
                        f"(finish_reason={resp.choices[0].finish_reason!r})")
                return parse_response(text, questions, self.arm)
            except Exception as e:  # noqa: BLE001 - retried, then surfaced
                last = e
        raise RuntimeError(f"scene call failed after {self.max_retries} "
                           f"attempts: {last}")
