"""Scene-first manual labelling tool — human ground truth for E3 round 4.

Unlike manual_tool.py (one composed actor+recipient render per item, needs a
Blender render job per unique pair), this shows ONE FULL SCENE image at a
time and lists every structurally-plausible question for that scene in a
sidebar. Score with a keypress, it saves and auto-advances to the next
question in the SAME scene; once a scene's questions are exhausted, moves to
the next scene. Zero render dependency: scene_NNNNN.png already exists for
every scene in the corpus.

Reuses manual_tool.py's record schema, blindness enforcement is structural
here too (the item list carries no VLM score / model prediction field —
see manual_tool.BLIND_FIELD_PATTERNS, same invariant applies).

Protocol: tools/docs/labeling_protocol.md.

Run:
    python -m affdet.label.scene_tool --scenes <scene_items.json> \
        --scenes-dir <omni_scenes_all_merged> --out <session.jsonl>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from affdet.label.manual_tool import (
    SCHEMA_VERSION, TOOL_VERSION, _blind_hits, _fetch_remote_image,
    append_record, definition_for, load_session, make_record,
)
from affdet.label.prompts import AFFORDANCE_QUESTION

_HTML = Path(__file__).parent / "scene_tool.html"


class SceneListError(ValueError):
    pass


def load_scene_items(path: Path | str) -> dict:
    """Load and validate a scene-items file, enforcing labeller blindness
    (same invariant as manual_tool.load_pair_list — no VLM score / model
    prediction field may reach the labeller)."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or "scenes" not in doc:
        raise SceneListError(f"{path} must be an object with a 'scenes' array")
    if not doc.get("pair_list_version"):
        raise SceneListError(f"{path} is missing 'pair_list_version'")

    hits = _blind_hits(doc)
    if hits:
        raise SceneListError(
            f"{path} carries answer-bearing field(s): {', '.join(sorted(hits))}. "
            "The labeller must be blind to the VLM label and the model "
            "prediction (tools/docs/labeling_protocol.md §2).")

    # A pair_id CAN legitimately recur across scenes -- the same (actor,
    # recipient) instance pair happening to co-occur in two of the 500
    # sampled scenes. That's not an error (unlike manual_tool.py's flat pair
    # list, where each pair_id is one standalone item): the answer is the
    # same regardless of which scene asks, so answering it once must mark
    # it done everywhere it appears (_scene_payload checks this per-item,
    # independently, via the same done_ids set). Only check that within a
    # SINGLE scene, no item repeats -- that WOULD be a sampler bug (the same
    # question asked twice in one scene's own list).
    for scene in doc["scenes"]:
        seen_in_scene: set[str] = set()
        for item in scene.get("items", []):
            pid = item["pair_id"]
            if pid in seen_in_scene:
                raise SceneListError(
                    f"duplicate pair_id {pid!r} within scene {scene['scene_id']!r}")
            seen_in_scene.add(pid)
    return doc


def _exclusions_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + ".excluded.jsonl")


def load_exclusions(out_path: Path) -> dict[str, str]:
    """scene_id -> reason, from the exclusions sidecar. Same load-fresh-
    every-request pattern as load_session, for the same reason: a browser
    refresh must see every exclusion recorded since server boot."""
    path = _exclusions_path(out_path)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[rec["scene_id"]] = rec.get("reason", "")
    return out


def append_exclusion(out_path: Path, scene_id: str, reason: str,
                      labeller: str, session_id: str) -> None:
    """A scene the labeller judged unusable (e.g. severe occlusion hiding an
    object well enough that no honest judgment is possible) -- excluded from
    E3 round-4's scene-level ground truth entirely, with a reason kept for
    audit. Append-only, same durability contract as append_record."""
    path = _exclusions_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "scene_id": scene_id,
        "reason": reason,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "labeller": labeller,
        "session_id": session_id,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def _build_scene(scene: dict, done_ids: set[str]) -> tuple[dict, int]:
    """One scene's payload shape (items annotated `done: bool`) plus its
    done-item count. Factored out of _scene_payload so /api/scene/<id> can
    return a single scene UNCONDITIONALLY -- including one that's fully
    answered and would otherwise be filtered out of the main queue -- for
    the "go back to an item whose scene already completed" case."""
    items = []
    n_done = 0
    for item in scene["items"]:
        is_done = item["pair_id"] in done_ids
        if is_done:
            n_done += 1
        items.append({
            "pair_id": item["pair_id"],
            "actor": item["actor"],
            "recipient": item["recipient"],
            "predicate": item["predicate"],
            "question": AFFORDANCE_QUESTION[item["predicate"]],
            "definition": definition_for(item["predicate"]),
            "done": is_done,
        })
    return {
        "scene_id": scene["scene_id"],
        "image_path": scene["image_path"],
        "image_size": scene.get("image_size"),
        "bboxes": scene.get("bboxes", {}),
        "items": items,
    }, n_done


def _scene_payload(doc: dict, records: list[dict], excluded: dict[str, str],
                    graded: bool) -> dict:
    """Every scene with its items annotated `done: bool` -- the frontend
    needs to know, per item, whether it's already answered (so it can skip
    straight to the first unanswered item and show a checkmark for the
    rest), not just a flat remaining-items queue like manual_tool.py's
    pair-first flow. A scene disappears from the payload once every one of
    its items is done, OR once the labeller has explicitly excluded it."""
    done_ids = {r["pair_id"] for r in records}
    scenes_out = []
    total_items = 0
    total_done = 0
    for scene in doc["scenes"]:
        if scene["scene_id"] in excluded:
            continue
        total_items += len(scene["items"])
        built, n_done = _build_scene(scene, done_ids)
        total_done += n_done
        if n_done == len(built["items"]):
            continue  # fully-answered scenes drop out of the active queue
        scenes_out.append(built)
    return {
        "scenes": scenes_out,
        "done": total_done,
        "total": total_items,
        "total_scenes": len(doc["scenes"]) - len(excluded),
        "n_excluded": len(excluded),
        "graded": graded,
    }


def build_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, _HTML.read_text(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                records = load_session(state["out"])
                excluded = load_exclusions(state["out"])
                payload = _scene_payload(state["doc"], records, excluded,
                                          state["graded"])
                return self._send(200, json.dumps(payload))
            if self.path.startswith("/api/scene/"):
                # Returns ONE scene unconditionally (even fully-answered),
                # for goBack() when the item's scene already dropped out of
                # the main /api/state queue -- a normal filtered payload
                # can't be used to jump back into a completed scene.
                scene_id = self.path[len("/api/scene/"):].split("?")[0]
                scene = state["by_scene_id"].get(scene_id)
                if scene is None:
                    return self._send(404, json.dumps({"error": "unknown scene_id"}))
                records = load_session(state["out"])
                done_ids = {r["pair_id"] for r in records}
                built, _ = _build_scene(scene, done_ids)
                return self._send(200, json.dumps(built))
            if self.path.startswith("/img/"):
                rel = self.path[len("/img/"):].split("?")[0]
                if state.get("remote_host"):
                    remote_path = f"{state['remote_scenes_root'].rstrip('/')}/{rel}"
                    data = _fetch_remote_image(state["remote_host"], remote_path)
                    if data is None:
                        return self._send(404, b"missing (remote)", "text/plain")
                    return self._send(200, data, "image/png")
                img = (state["scenes_dir"] / rel).resolve()
                if not str(img).startswith(str(state["scenes_dir"].resolve())):
                    return self._send(403, b"forbidden", "text/plain")
                if not img.exists():
                    return self._send(404, b"missing", "text/plain")
                return self._send(200, img.read_bytes(), "image/png")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/judge":
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                item = state["by_id"].get(body.get("pair_id"))
                if item is None:
                    return self._send(400, json.dumps({"error": "unknown pair_id"}))
                # make_record's schema has one scene_id slot -- stamp the scene
                # the labeller was actually LOOKING AT (body["scene_id"], sent
                # by the frontend). The full pair_id -> [every scene it appears
                # in] map lives in the meta sidecar (item_scene), which is what
                # the offline join-back-onto-scenes step must use, not this
                # field -- a pair_id recurring across scenes means the answer
                # applies to all of them, not just the one named here.
                rec = make_record(
                    {**item, "scene_id": body.get("scene_id")},
                    body.get("verdict"),
                    confidence=body.get("confidence"),
                    confidence_explicit=bool(body.get("confidence_explicit")),
                    note=body.get("note", ""),
                    elapsed_ms=body.get("elapsed_ms", 0),
                    pair_list_version=state["pair_list_version"],
                    session_id=state["session_id"],
                    human_score=body.get("human_score"),
                    labeller=state["labeller"],
                    revision_of=body.get("revision_of"),
                )
                append_record(state["out"], rec)
                return self._send(200, json.dumps({"ok": True}))
            if self.path == "/api/exclude_scene":
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                scene_id = body.get("scene_id")
                if not scene_id:
                    return self._send(400, json.dumps({"error": "missing scene_id"}))
                append_exclusion(state["out"], scene_id, body.get("reason", ""),
                                  state["labeller"], state["session_id"])
                return self._send(200, json.dumps({"ok": True}))
            return self._send(404, b"not found", "text/plain")

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", required=True, help="scene_items JSON")
    ap.add_argument("--out", required=True, help="output JSONL (appended)")
    ap.add_argument("--scenes-dir",
                    help="dir of scene_NNNNN/scene_NNNNN.png dirs (local disk). "
                         "Required unless --remote-host is set.")
    ap.add_argument("--remote-host", default=None,
                    help="ssh host to fetch scene images from live instead "
                         "of requiring a local copy (see manual_tool.py's "
                         "--remote-host for the same mechanism).")
    ap.add_argument("--remote-scenes-dir",
                    help="remote scenes dir on --remote-host, e.g. "
                         "/scratch/user/omni_scenes_all_merged.")
    ap.add_argument("--labeller", default="max")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--graded", action="store_true", default=True)
    args = ap.parse_args(argv)

    if args.remote_host and not args.remote_scenes_dir:
        ap.error("--remote-scenes-dir is required when --remote-host is set")
    if not args.remote_host and not args.scenes_dir:
        ap.error("--scenes-dir is required unless --remote-host is set")

    doc = load_scene_items(args.scenes)
    version = doc["pair_list_version"]
    by_id: dict[str, dict] = {}
    by_scene_id: dict[str, dict] = {sc["scene_id"]: sc for sc in doc["scenes"]}
    #: pair_id -> every scene it appears in (a pair_id can legitimately
    #: recur across scenes, see load_scene_items) -- the authoritative join
    #: key for the offline "distribute labels back onto scenes" step.
    #: Written into the meta sidecar below, NOT relied on via the record's
    #: single scene_id field.
    item_scenes: dict[str, list[str]] = {}
    for scene in doc["scenes"]:
        for item in scene["items"]:
            by_id[item["pair_id"]] = item
            item_scenes.setdefault(item["pair_id"], []).append(scene["scene_id"])

    records = load_session(args.out)
    session_id = args.session_id or datetime.now().strftime("%Y-%m-%d-%H%M")
    state = {
        "doc": doc,
        "by_id": by_id,
        "by_scene_id": by_scene_id,
        "graded": args.graded,
        "out": Path(args.out),
        "scenes_dir": Path(args.scenes_dir) if args.scenes_dir else None,
        "remote_host": args.remote_host,
        "remote_scenes_root": args.remote_scenes_dir,
        "pair_list_version": version,
        "session_id": session_id,
        "labeller": args.labeller,
    }

    meta = Path(args.out).with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "session_id": session_id,
        "labeller": args.labeller,
        "scene_items_list": str(args.scenes),
        "pair_list_version": version,
        "scene_items_sha1": hashlib.sha1(Path(args.scenes).read_bytes()).hexdigest(),
        "pair_id_to_scene_ids": item_scenes,
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "protocol": "tools/docs/labeling_protocol.md",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "n_total_items": len(by_id),
        "n_scenes": len(doc["scenes"]),
        "n_already_done": len(records),
    }, indent=2))

    total = len(by_id)
    done = len({r["pair_id"] for r in records} & set(by_id))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{total - done} of {total} items remaining -> {url}")
    print(f"writing to {args.out} (autosaved after every judgement)")
    if not args.no_browser:
        webbrowser.open(url)
    srv = HTTPServer(("127.0.0.1", args.port), build_handler(state))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped — progress saved in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
