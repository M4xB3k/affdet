"""OmniLabeler — VLM-based affordance scorer for paired-scene renders.

Reads a pairs.json + the manifests written by affdet/render/omni_pair_render.py
(actor + recipient composed together in ONE real scene, multiple camera
angles, true relative scale) and scores every structurally-possible predicate
for each pair in a SINGLE VLM call (gated by intersection_matrix.is_candidate_pair,
so the model is never asked a meaningless question like "can this knife
contain the apple"). canPourInto is excluded from VLM calls entirely — it is
rule-based (see vlm_resolver.py), not a visual judgment. canPierce WAS also
rule-based (hardcoded true for any knife/fork) until 2026-08-03, when a
blunt-tip novelty knife was found scoring canCut correctly low but would
have piercing hardcoded true anyway — the rule missed that piercing depends
on tip sharpness, not blade type, which only a visual check catches; see
VLM_PREDICATES below.

Output format (one JSON object per line):
  {"actor": "dish_011", "recipient": "durian_030", "predicate": "canSupport",
   "score": 4, "rationale": "..."}

Resumable: already-labeled (actor, recipient, predicate) triples are skipped
on restart. Chunked: runs in chunks of --chunk-size with a cooldown between
chunks to stay well under the 10k RPD limit for tier-1 OpenAI accounts.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from affdet.intersection_matrix import TIER1_PREDICATES, is_candidate_pair
from affdet.label.prompts import build_system_prompt, build_user_prompt
from affdet.label.rate_limiter import TokenRateLimiter

# Rough estimate of prompt tokens for a single labeling call, used only to
# reserve budget from the rate limiter *before* the real usage is known
# (report_actual() corrects the bucket afterward). Base64-encoded grid image
# dominates the prompt; ~2.7k tokens matches observed real usage well.
_ESTIMATED_TOKENS_PER_CALL = 2700

CHUNK_SIZE_DEFAULT = 500
CHUNK_COOLDOWN_DEFAULT = 60  # seconds between chunks

# canPourInto is rule-based, not VLM-judged: its per-instance truth comes
# from pourable_from/pourable_into in resolved_instances.py (a fixed
# from/into rule table keyed on effective class + instance number — e.g.
# bottle is ruled out entirely as "all sealed"). No amount of looking at an
# image tells you whether a specific scan's lid is glued shut; that's a
# curation fact, not a visual judgment.
_RULE_BASED_PREDICATES = {"canPourInto"}
VLM_PREDICATES: list[str] = [p for p in TIER1_PREDICATES if p not in _RULE_BASED_PREDICATES]


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def compose_cancut_grid(angle0: Path, angle1: Path, actor_tile: Path,
                         recipient_tile: Path, out_path: Path) -> None:
    """canCut-only image: top row = 2 paired-scene angles (real relative
    scale, so the model can still judge size fit), bottom row = solo tiles
    of the actor and recipient alone (unambiguous view of tool TYPE -- blade
    mechanism, handle, hinge -- that gets lost when quartered into a small
    paired-angle tile). Solo tiles carry no size reference on their own,
    which is exactly why they were dropped for size-sensitive predicates;
    scoping this to canCut only, alongside 2 real paired angles, keeps a
    size reference in the same image without losing type-classification
    detail."""
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in
            (angle0, angle1, actor_tile, recipient_tile)]
    cell_w, cell_h = imgs[0].size
    imgs = [imgs[0], imgs[1],
            imgs[2].resize((cell_w, cell_h)), imgs[3].resize((cell_w, cell_h))]

    grid = Image.new("RGB", (cell_w * 2, cell_h * 2))
    grid.paste(imgs[0], (0, 0))
    grid.paste(imgs[1], (cell_w, 0))
    grid.paste(imgs[2], (0, cell_h))
    grid.paste(imgs[3], (cell_w, cell_h))
    grid.save(out_path)


def _split_instance(inst: str) -> str:
    """'durian_030' -> 'durian'. Raises if inst has no trailing _NNN id."""
    tail = inst.rsplit("_", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return tail[0]
    raise ValueError(f"Cannot resolve category for instance id: {inst!r}")


def _load_done(out_path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not out_path.exists():
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["actor"], r["recipient"], r["predicate"]))
            except Exception:
                pass
    return done


def build_pairs(pairs_path: Path, renders_dir: Path) -> list[dict]:
    """Enumerate labeling jobs from a pairs.json + rendered manifests.

    Each pairs.json entry ({"actor": ..., "recipient": ...}) is joined against
    its rendered pair directory (<renders_dir>/<actor>__<recipient>/), and the
    candidate predicates are computed once via is_candidate_pair — a pair with
    zero candidate predicates is skipped (nothing to label regardless of the
    render). A pair with an unusable render (container_flipped) is NOT
    skipped: a flipped container genuinely cannot contain/support/be poured
    into in that pose, so it still gets a job, just flagged auto_negative so
    run() writes a rule-based negative for the container-dependent predicates
    without spending an API call or judging a broken image. Every candidate
    pair therefore gets a label -- no silent gaps in the corpus from render
    failures. Returns one dict per PAIR (not per predicate) — the whole
    labeling call for a pair happens in one API request (when not auto-neg)."""
    with open(pairs_path) as f:
        raw_pairs = json.load(f)

    jobs: list[dict] = []
    for spec in raw_pairs:
        actor_inst, recipient_inst = spec["actor"], spec["recipient"]
        pair_dir = renders_dir / f"{actor_inst}__{recipient_inst}"
        manifest_path = pair_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        auto_negative = bool(manifest.get("container_flipped"))

        actor_cat = _split_instance(actor_inst)
        recipient_cat = _split_instance(recipient_inst)
        predicates = [
            p for p in VLM_PREDICATES
            if is_candidate_pair(actor_cat, recipient_cat, p)
        ]
        if not predicates:
            continue

        grid_path = pair_dir / manifest.get("grid_file", "grid.png")
        if not auto_negative and not grid_path.exists():
            continue

        jobs.append({
            "actor_inst": actor_inst,
            "recipient_inst": recipient_inst,
            "actor_cat": actor_cat,
            "recipient_cat": recipient_cat,
            "predicates": predicates,
            "grid_path": grid_path,
            "grid_rows": manifest["grid"]["rows"],
            "grid_cols": manifest["grid"]["cols"],
            "auto_negative": auto_negative,
            "pair_dir": pair_dir,
        })

    return jobs


@dataclass
class OmniLabeler:
    model: str = "gpt-5-mini"
    max_retries: int = 5
    chunk_size: int = CHUNK_SIZE_DEFAULT
    chunk_cooldown: int = CHUNK_COOLDOWN_DEFAULT
    client: Any = None  # injected in tests; built lazily from OpenAI() otherwise
    rate_limiter: TokenRateLimiter | None = None  # only consulted when max_workers > 1
    tiles_dir: Path | None = None  # solo-object tiles root (<tiles_dir>/<class>/tile_<NNN>.png);
    # when set, canCut jobs use compose_cancut_grid() instead of the standard
    # paired grid (2 paired angles + 2 solo tiles, for clearer type
    # classification -- see compose_cancut_grid's docstring)

    total_prompt_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)
    total_reasoning_tokens: int = field(default=0, init=False)
    _usage_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def run(
        self,
        pairs_path: Path,
        renders_dir: Path,
        out_path: Path,
        offset: int = 0,
        limit: int | None = None,
        dry_run: bool = False,
        max_workers: int = 1,
    ) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        jobs = build_pairs(Path(pairs_path), Path(renders_dir))

        if dry_run:
            from collections import Counter
            n_predicates = sum(len(j["predicates"]) for j in jobs)
            print(f"Total pairs: {len(jobs)}, total predicate judgments: {n_predicates}")
            for key, count in sorted(Counter(
                f"{j['actor_cat']}.{j['recipient_cat']}" for j in jobs
            ).items()):
                print(f"  {key}: {count}")
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = _load_done(out_path)
        remaining = [
            j for j in jobs
            if any(
                (j["actor_inst"], j["recipient_inst"], p) not in done
                for p in j["predicates"]
            )
        ]
        if offset:
            remaining = remaining[offset:]
        if limit is not None:
            remaining = remaining[:limit]

        total = len(remaining)
        print(f"Already done: {len(done)} predicate judgments, "
              f"remaining: {total} pairs")
        if total == 0:
            print("Nothing to do.")
            return

        client = self.client if self.client is not None else self._make_client()

        chunks = [remaining[i:i + self.chunk_size] for i in range(0, total, self.chunk_size)]
        n_chunks = len(chunks)
        global_done = 0
        global_failed = 0

        with open(out_path, "a") as f_out:
            for chunk_idx, chunk in enumerate(chunks):
                print(f"\n── Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} pairs) ──")

                iterator = (
                    tqdm(chunk, desc=f"chunk {chunk_idx+1}/{n_chunks}", unit="pair")
                    if tqdm else chunk
                )

                chunk_failed = 0

                def _handle_result(job, pending, result) -> bool:
                    """Writes a record per predicate; returns True if failed."""
                    if result is None:
                        if tqdm is None:
                            print("FAILED")
                        return True
                    for predicate, score in result.items():
                        if predicate not in pending:
                            continue
                        record = {
                            "actor": job["actor_inst"],
                            "recipient": job["recipient_inst"],
                            "predicate": predicate,
                            "score": score,
                            "rationale": None,
                        }
                        f_out.write(json.dumps(record) + "\n")
                    f_out.flush()
                    if tqdm is None:
                        print(result)
                    return False

                if max_workers <= 1:
                    for job in iterator:
                        pending = [
                            p for p in job["predicates"]
                            if (job["actor_inst"], job["recipient_inst"], p) not in done
                        ]
                        if not pending:
                            continue
                        if tqdm is None:
                            print(
                                f"[{global_done+1}/{total}] {job['actor_inst']} x "
                                f"{job['recipient_inst']} -> {pending}",
                                end=" ... ", flush=True,
                            )

                        result = self._run_job(client, job, pending)
                        failed = _handle_result(job, pending, result)
                        global_failed += int(failed)
                        chunk_failed += int(failed)
                        global_done += 1
                else:
                    pending_jobs = []
                    for job in chunk:
                        pending = [
                            p for p in job["predicates"]
                            if (job["actor_inst"], job["recipient_inst"], p) not in done
                        ]
                        if pending:
                            pending_jobs.append((job, pending))

                    write_lock = threading.Lock()
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {
                            pool.submit(self._run_job, client, job, pending): (job, pending)
                            for job, pending in pending_jobs
                        }
                        completed_iter = as_completed(futures)
                        if tqdm:
                            completed_iter = tqdm(
                                completed_iter, total=len(futures),
                                desc=f"chunk {chunk_idx+1}/{n_chunks}", unit="pair",
                            )
                        for future in completed_iter:
                            job, pending = futures[future]
                            result = future.result()
                            with write_lock:
                                failed = _handle_result(job, pending, result)
                            global_failed += int(failed)
                            chunk_failed += int(failed)
                            global_done += 1
                            if self.rate_limiter is not None and tqdm:
                                completed_iter.set_postfix(
                                    tpm=f"{self.rate_limiter.current_tokens_per_minute():,.0f}"
                                )

                print(
                    f"Chunk {chunk_idx+1} done — "
                    f"{len(chunk) - chunk_failed} labeled, {chunk_failed} failed. "
                    f"Total so far: {global_done}/{total} ({global_failed} failed). "
                    f"Tokens so far: {self.total_prompt_tokens} prompt + "
                    f"{self.total_completion_tokens} completion "
                    f"({self.total_reasoning_tokens} reasoning)"
                )

                if chunk_idx < n_chunks - 1:
                    print(f"Cooling down {self.chunk_cooldown}s before next chunk...")
                    time.sleep(self.chunk_cooldown)

        print(
            f"\nDone. {global_done} pairs labeled, {global_failed} failed. "
            f"Output: {out_path}\n"
            f"Total tokens: {self.total_prompt_tokens} prompt + "
            f"{self.total_completion_tokens} completion "
            f"({self.total_reasoning_tokens} of which reasoning)"
        )

    def _run_job(self, client, job: dict, pending: list[str]) -> dict | None:
        if job.get("auto_negative"):
            # Container settled flipped and rendering is unusable -- genuinely
            # cannot contain/support/be poured into in that pose, so this is
            # a real rule-based negative, not a guess. Skip the API call.
            return {p: 0 for p in pending}
        return self._label_pair(client, job, pending)

    def _cancut_grid_path(self, job: dict) -> Path | None:
        """Builds (or reuses a cached) canCut-only 2x2 grid: 2 paired angles
        + 2 solo tiles. Returns None if any required source image is
        missing, so the caller can fall back to the standard grid."""
        pair_dir = job.get("pair_dir")
        if pair_dir is None or self.tiles_dir is None:
            return None
        angle0, angle1 = pair_dir / "angle_00.png", pair_dir / "angle_01.png"
        actor_num = job["actor_inst"].rsplit("_", 1)[-1]
        recip_num = job["recipient_inst"].rsplit("_", 1)[-1]
        actor_tile = Path(self.tiles_dir) / job["actor_cat"] / f"tile_{actor_num}.png"
        recip_tile = Path(self.tiles_dir) / job["recipient_cat"] / f"tile_{recip_num}.png"
        if not all(p.exists() for p in (angle0, angle1, actor_tile, recip_tile)):
            return None
        out_path = pair_dir / "cancut_grid.png"
        if not out_path.exists():
            compose_cancut_grid(angle0, angle1, actor_tile, recip_tile, out_path)
        return out_path

    def _label_pair(self, client, job: dict, predicates: list[str]) -> dict | None:
        system_prompt = build_system_prompt(predicates)
        image_path = job["grid_path"]
        used_composed_grid = False
        if "canCut" in predicates:
            # canPierce also benefits from the solo tiles (tip shape doesn't
            # need a size reference the way canCut's food-fit judgment does)
            # -- and since is_candidate_pair makes canPierce a candidate for
            # every knife pair canCut is, they're ALWAYS bundled into one
            # call for real knife pairs, so gating this on predicates ==
            # ["canCut"] alone meant the clearer solo-tile image was never
            # actually used in production.
            composed_path = self._cancut_grid_path(job)
            if composed_path is not None:
                image_path = composed_path
                used_composed_grid = True
        user_prompt = build_user_prompt(
            job["actor_cat"], job["recipient_cat"], predicates,
            job["grid_rows"], job["grid_cols"],
            solo_tile_bottom_row=used_composed_grid,
        )
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{_encode_image(image_path)}",
                "detail": "high",
            }},
        ]
        from openai import APIStatusError

        for attempt in range(self.max_retries):
            try:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire(_ESTIMATED_TOKENS_PER_CALL)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    # GPT-5-series reasoning models reject max_tokens/temperature
                    # (fixed at 1). reasoning_effort="minimal" used to be the
                    # default for predicates not specifically bumped, but by
                    # 2026-08-03 four of the five VLM-judged predicates
                    # (canCut, canContain, canSupport, canPierce) had each,
                    # one at a time, been found to need "low" instead --
                    # "minimal" produced confident wrong scores every time,
                    # only ever caught after a real labeling defect surfaced
                    # (box-cutter knives misclassified, size-mismatch
                    # containers/supports scored high, a blunt-tip novelty
                    # knife scored as a real piercer). Retired "minimal"
                    # rather than wait to rediscover the same failure for
                    # canPeel too -- "low" is used for every predicate now.
                    # "low" burns real hidden reasoning tokens the old
                    # 50-per-predicate budget didn't cover -- calls came back
                    # empty (finish_reason="length") until raised.
                    max_completion_tokens=1500,
                    reasoning_effort="low",
                )
                if resp.usage is not None:
                    self._log_usage(resp.usage)
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError(
                        f"empty response content (finish_reason="
                        f"{resp.choices[0].finish_reason!r})"
                    )
                parsed = json.loads(content.strip())
                missing = [p for p in predicates if p not in parsed]
                if missing:
                    raise ValueError(f"response missing predicate(s): {missing}")
                # canSupport {"score": N, "size_note": "..."} and canCut
                # {"type": "...", "score": N} (see prompts.py's _OUTRO) both
                # unwrap to the plain int every other predicate already is --
                # the extra field is discarded, only there to make the model
                # articulate its reasoning (measurably stabilizes both).
                return {
                    p: (v["score"] if isinstance(v, dict) else v)
                    for p, v in parsed.items() if p in predicates
                }
            except APIStatusError as e:
                # 4xx other than 429 (bad request, auth, content policy,
                # model-not-found) will not succeed on retry -- fail fast
                # instead of burning the full backoff ladder on every pair.
                if e.status_code != 429 and 400 <= e.status_code < 500:
                    self._log_retry(attempt, e, terminal=True)
                    return None
                self._sleep_backoff(attempt, e)
            except Exception as e:
                self._sleep_backoff(attempt, e)
        return None

    def _sleep_backoff(self, attempt: int, err: Exception) -> None:
        wait = min(2 ** attempt, 60)
        self._log_retry(attempt, err, wait=wait)
        time.sleep(wait)

    def _log_retry(self, attempt: int, err: Exception, wait: int = 0,
                    terminal: bool = False) -> None:
        if terminal:
            msg = f"  [failed, not retrying] {err}"
        else:
            msg = f"  [retry {attempt+1}/{self.max_retries}] {err} — waiting {wait}s"
        if tqdm_available():
            from tqdm import tqdm
            tqdm.write(msg)
        else:
            print(msg)

    def _log_usage(self, usage) -> None:
        reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        with self._usage_lock:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.total_reasoning_tokens += reasoning
        if self.rate_limiter is not None:
            self.rate_limiter.report_actual(usage.prompt_tokens + usage.completion_tokens)

    @staticmethod
    def _make_client():
        import os
        from openai import OpenAI

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if azure_endpoint and azure_key:
            # Azure's v1 surface serves classic chat.completions too, so no
            # need for the separate AzureOpenAI client / api_version dance —
            # model= must be the Azure *deployment name*, not a model id.
            base_url = azure_endpoint.rstrip("/") + "/openai/v1/"
            return OpenAI(base_url=base_url, api_key=azure_key)
        return OpenAI()


def tqdm_available() -> bool:
    try:
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False
