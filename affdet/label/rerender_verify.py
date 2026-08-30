"""E3 re-render verification tool.

The bbox/deterministic-RNG fix to affdet/render/omni_pair_render.py (see
docs/superpowers/plans/2026-08-11-e3-pair-render-bboxes.md) required
re-rendering the 146 pairs already judged by a human in
data/eval/e3_full_147pair_comparison.json. A pixel-diff check of those
re-renders against the originals showed real differences on several pairs
(pose changes, and in at least one case a different mesh variant) -- too
coarse a signal to trust automatically. This tool shows the human, side by
side, the original render they judged and the new re-render, along with
their own prior score and the VLM's score, and asks a single question per
pair: does the new image still show the scene you judged?

Unlike manual_tool.py, this is NOT a blind labelling session -- there is no
fresh judgment being collected, so showing the prior scores does not
contaminate anything. Verdicts are match / mismatch / unsure, not a fresh
0-9 score.

Run:
    python -m affdet.label.rerender_verify \\
        --pairs data/labels/pair_pilot/pairs_e3_judged_subset.json \\
        --comparison data/eval/e3_full_147pair_comparison.json \\
        --old-renders data/labels/pair_pilot \\
        --new-renders data/labels/pair_pilot/pair_renders_bbox_v2 \\
        --out data/eval/e3_rerender_verify.jsonl
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable

VERIFY_VERDICTS: tuple[str, ...] = ("match", "mismatch", "unsure")


#: Historical pair-render batch directories to search, in the order they were
#: produced. A pair's original render can live in any one of these -- see
#: docs/superpowers/plans/2026-08-11-e3-pair-render-bboxes.md for why the
#: corpus is split across multiple batches.
DEFAULT_OLD_RENDER_SUBDIRS: tuple[str, ...] = (
    "renders_e3_round2", "renders_e3_combined",
    "renders_e3_round3", "renders_e3_expansion",
)


def find_old_render_dir(
    search_dirs: list[Path], actor: str, recipient: str,
) -> Path | None:
    """First `search_dirs` entry containing a rendered `<actor>__<recipient>/`
    pair dir with at least one angle image, or None if the pair is not found
    in any of them."""
    name = f"{actor}__{recipient}"
    for base in search_dirs:
        candidate = base / name
        if candidate.is_dir() and any(candidate.glob("angle_*.png")):
            return candidate
    return None


def group_judgments_by_pair(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Comparison-JSON rows (one per actor/recipient/predicate) grouped by
    (actor, recipient) -- a pair judged under multiple predicates gets one
    entry per predicate, in the order they appear in `rows`."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["actor"], row["recipient"])].append(row)
    return dict(grouped)


def build_items(
    pairs: list[dict[str, str]],
    judgment_rows: list[dict[str, Any]],
    *,
    old_search_dirs: list[Path],
    new_render_dir: Path,
) -> list[dict[str, Any]]:
    """One review item per pair: where its old and new renders live (None if
    missing) and every judgment (predicate + human/VLM score) recorded
    against it. `pair_id` is the `<actor>__<recipient>` dir name -- stable
    and human-readable, unlike manual_tool.py's content-hash pair_id, since
    there is no VLM-label leakage risk to guard against here."""
    judgments_by_pair = group_judgments_by_pair(judgment_rows)
    items = []
    for pair in pairs:
        actor, recipient = pair["actor"], pair["recipient"]
        old_dir = find_old_render_dir(old_search_dirs, actor, recipient)
        new_dir = find_old_render_dir([new_render_dir], actor, recipient)
        items.append({
            "pair_id": f"{actor}__{recipient}",
            "actor": actor,
            "recipient": recipient,
            "old_dir": old_dir,
            "new_dir": new_dir,
            "judgments": judgments_by_pair.get((actor, recipient), []),
        })
    return items


def make_verify_record(pair_id: str, verdict: str, *, note: str) -> dict[str, Any]:
    """One verification verdict. `match`/`mismatch`/`unsure` -- not a fresh
    0-9 judgment, since no new labelling is happening here (see module
    docstring for why this tool is allowed to show the human's own prior
    score, unlike manual_tool.py)."""
    if verdict not in VERIFY_VERDICTS:
        raise ValueError(f"verdict must be one of {VERIFY_VERDICTS}, got {verdict!r}")
    return {
        "pair_id": pair_id,
        "verdict": verdict,
        "note": note,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def append_verify_record(path: Path | str, record: dict[str, Any]) -> None:
    """Append one verdict and flush to disk immediately -- same durability
    contract as manual_tool.append_record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def load_verify_session(path: Path | str) -> list[dict[str, Any]]:
    """Read a session back, tolerating a truncated final line from a crash.

    A pair_id can appear more than once (the UI's back button lets a verdict
    be resubmitted) -- last write wins, so the returned list has at most one
    record per pair_id, holding its most recent verdict."""
    path = Path(path)
    if not path.exists():
        return []
    by_pair: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_pair[rec["pair_id"]] = rec
    return list(by_pair.values())


def remaining_verify_items(
    items: Iterable[dict[str, Any]], records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Items not yet given a verdict."""
    done = {r["pair_id"] for r in records}
    return [i for i in items if i["pair_id"] not in done]


def build_verify_payload(
    items: list[dict[str, Any]], *, done: int, total: int,
) -> dict[str, Any]:
    """What the page needs. `old_dir`/`new_dir` (Path | None, filesystem-only)
    become `old_available`/`new_available` booleans -- the page never sees a
    local filesystem path, only whether an image exists to fetch from /img/."""
    return {
        "items": [
            {
                "pair_id": i["pair_id"],
                "actor": i["actor"],
                "recipient": i["recipient"],
                "old_available": i["old_dir"] is not None,
                "new_available": i["new_dir"] is not None,
                "judgments": i["judgments"],
            }
            for i in items
        ],
        "done": done,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

_HTML = (Path(__file__).parent / "rerender_verify.html")


def build_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet during a session
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
                return self._send(200, json.dumps(state["payload"]))
            if self.path.startswith("/img/"):
                # /img/<old|new>/<pair_id>/<filename>
                parts = self.path[len("/img/"):].split("?")[0].split("/", 2)
                if len(parts) != 3:
                    return self._send(404, b"not found", "text/plain")
                which, pair_id, filename = parts
                item = state["by_id"].get(pair_id)
                if item is None:
                    return self._send(404, b"unknown pair", "text/plain")
                base_dir = item["old_dir"] if which == "old" else item["new_dir"] if which == "new" else None
                if base_dir is None:
                    return self._send(404, b"no render for this pair", "text/plain")
                img = (base_dir / filename).resolve()
                # Path containment: only files under this pair's own render
                # dir may be served, regardless of what filename is requested.
                if not str(img).startswith(str(base_dir.resolve())):
                    return self._send(403, b"forbidden", "text/plain")
                if not img.exists():
                    return self._send(404, b"missing", "text/plain")
                return self._send(200, img.read_bytes(), "image/png")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/verify":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            pair_id = body.get("pair_id")
            if pair_id not in state["by_id"]:
                return self._send(400, json.dumps({"error": "unknown pair_id"}))
            try:
                rec = make_verify_record(pair_id, body.get("verdict"), note=body.get("note", ""))
            except ValueError as e:
                return self._send(400, json.dumps({"error": str(e)}))
            append_verify_record(state["out"], rec)
            state["done"] += 1
            return self._send(200, json.dumps({"ok": True, "done": state["done"]}))

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, help="pair list JSON (list of {actor, recipient})")
    ap.add_argument("--comparison", required=True,
                    help="e3_full_147pair_comparison.json (human/vlm scores)")
    ap.add_argument("--old-renders", required=True,
                    help="dir containing the historical renders_e3_* batch dirs")
    ap.add_argument("--new-renders", required=True,
                    help="dir of freshly re-rendered <actor>__<recipient>/ dirs")
    ap.add_argument("--out", required=True, help="output JSONL (appended)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    pairs = json.loads(Path(args.pairs).read_text())
    judgment_rows = json.loads(Path(args.comparison).read_text())
    old_renders = Path(args.old_renders)
    old_search_dirs = [old_renders / d for d in DEFAULT_OLD_RENDER_SUBDIRS]
    new_render_dir = Path(args.new_renders)

    items = build_items(
        pairs, judgment_rows,
        old_search_dirs=old_search_dirs, new_render_dir=new_render_dir,
    )
    records = load_verify_session(args.out)
    todo = remaining_verify_items(items, records)

    if not todo:
        print(f"All {len(items)} items already verified in {args.out}. Nothing to do.")
        return 0

    state = {
        "payload": build_verify_payload(todo, done=len(records), total=len(items)),
        "by_id": {i["pair_id"]: i for i in items},
        "out": Path(args.out),
        "done": len(records),
    }

    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(todo)} of {len(items)} items remaining -> {url}")
    print(f"writing to {args.out} (autosaved after every verdict)")
    if not args.no_browser:
        webbrowser.open(url)
    srv = HTTPServer(("127.0.0.1", args.port), build_handler(state))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped — {state['done']}/{len(items)} verified, all saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
