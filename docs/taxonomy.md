# Affordance Taxonomy

Canonical definition of the affordance taxonomy for the thesis: the affordances,
the objects, and every object × affordance intersection.

This document supersedes the looser 6-affordance set sketched in
`meeting_slides_content.md` / `presentation_2.tex`. Where they disagree, this file wins.

---

## 0. Governing Principles

These four principles constrain every definition below. Read them first.

### P1 — Binary, part-resolved predicates
Every affordance is a **boolean** predicate: it holds or it does not. The TAMP
planner (Fast Downward / PDDL) consumes boolean predicates only — there is no
native notion of "0.7 good at cutting". The baselines and the committed
evaluation metrics (precision / recall / F1) are all defined on binary outcomes.

The boolean is asserted over a **part-resolved tuple**, not a bare object pair:
the unit of prediction for Tier 1 is `(actor, actor_part, recipient)` → {0,1}
(see P4 for why the actor part is predicted by the model and the recipient is
whole-object in the baseline). "Binary" qualifies the *verdict*; it does not
mean the model's only job is an object-pair yes/no — the model also predicts the
acting part, which is the load-bearing novel claim (P4).

### P2 — Instance-level, not class-level
An affordance is a property of the **depicted object instance**, not its class.
A butter knife cannot cut bread; a serrated bread knife can — both are class
"knife". The intersection matrix in §3 therefore never asserts
`canCut(knife, bread) = ✓`. It records a *candidate* pair plus the **visual
discriminator** that actually decides it (here: blade edge type). This
instance sensitivity is the central argument for a learned visual model over
the attribute/class-based rule baseline: the rules see "knife, rigid, has_edge"
and cannot tell the two knives apart; DINOv2 features over the actual image can.

### P3 — The model predicts the acting part (the load-bearing novel claim)
For each candidate Tier-1 pair the model predicts **both** the binary relation
verdict **and the actor part** (which member acts — e.g. for `canCut`, that the
acting part is the `edge`/blade, not the `handle`). This is what makes the
part-parameterization *load-bearing* rather than decoration: the strong, novel,
visually-grounded claim is "the model finds, from RGB, which part performs the
action", not merely "cutting is possible somewhere". The actor part is **not**
supplied by the rule emitter for the 7 Tier-1 predicates — the model produces
it. (This supersedes, for Tier 1, the earlier "VLM labels the binary verdict
only, parts come from the rule emitter" scoping in finding `01_vlm_labeling.md`;
that appendix predates this decision — see §4 note.)

The **recipient region is whole-object for the baseline** (the model does not
localize where on the recipient the action lands). Per-recipient-region /
graded-suitability prediction is the explicit *ambitious extension* (the novel
reach beyond the baseline), deliberately out of scope here so the baseline stays
tight and the extension has a real frontier to claim.

The model's MLP emits a sigmoid score ∈ [0, 1] for the verdict; it is
**thresholded to a boolean before it reaches the planner** (P1). The raw score
MAY be reported in the discussion as qualitative evidence of *suitability* (e.g.
higher `canCut` confidence for a serrated than a smooth knife). It is **never**
the planner contract and is **not** an evaluated quantity in the baseline.

### P4 — Member/face model (the part-parametrization rule)
Affordances are part-parameterized using **only the vocabulary that already
exists in `pipeline/afford.py`** — no new parts or faces are introduced:

- **Members** (`ALL_MEMBERS`, afford.py:57): `body`, `handle`, `edge`
- **Faces** (`PARTS_TO_FACES`, afford.py:38–42): `on`, `under`, `front`, `back`, `left`, `right`

The parametrization is **asymmetric by physical role**, and — for the baseline —
asymmetric by *who produces it*:

| Role | Parametrized by | Produced by (baseline) | Why |
|---|---|---|---|
| **Actor** (the thing performing the action) | a **member** | **the model** (predicted from RGB) | the motion planner must know *which physical part* acts (you cut with the `edge`, not the `handle`); predicting it from the image is the novel claim (P3) — exactly why the existing `canGraspFrom` is member-parameterized |
| **Recipient** (the thing acted upon) | a **face** | **not predicted in the baseline** — whole-object | localizing *where on the recipient* the action lands is the ambitious extension (P3), out of baseline scope; the baseline treats the recipient as the whole object |
| Whole-object affordance | neither | n/a | some affordances have no meaningful contact part (see `canContain`) |

So for a Tier-1 pair the baseline model output is **`(actor, actor_part,
recipient)` → {0,1}**. The recipient `face` column in §1's table is the
*physical* contact face (kept for documentation and for the PDDL adapter), **not**
something the baseline model predicts. Where the recipient face is needed to
emit a 4-arity PDDL predicate, it is supplied by the existing rule emitter or a
fixed per-predicate default (the PDDL adapter is the lossy boundary; the
detector is not constrained by it — the rich actor-part output is the detector's
contribution, the adapter is downstream).

Arity is therefore **mixed by physics, not uniform**. PDDL accepts any arity;
the only real constraint is that the consuming domain action has the arguments
it needs. Semantic part names that appeared on earlier slides
(`BladeEdge`, `SpoutFace`, `InsideFace`, `RimFace`) are **thesis-prose aliases
for readability only** — the contract uses the canonical vocabulary above.

---

## 1. Affordance Definitions

Two tiers. Tier 1 is the thesis's novel contribution. Tier 2 is kept only so
the pipeline output stays complete and drop-in compatible with U-TAMP-2.

### Tier 1 — Inter-object affordances (learned; novel contribution)

7 directed relations over an ordered object pair `(A, B)`. Directed:
`canCut(A,B) ≠ canCut(B,A)`.

In the columns below, **Actor part (member)** is *predicted by the model* (the
novel claim, P3); **Recipient part (face)** is the *physical contact face* kept
for documentation and the PDDL adapter — it is **not** predicted by the baseline
model (recipient is whole-object; P4). Baseline model output per pair is
`(actor, actor_part, recipient)` → {0,1}.

| # | Affordance | Meaning | Actor part (member) — *model-predicted* | Recipient part (face) — *physical/adapter, not predicted* | Decided at instance level by |
|---|---|---|---|---|---|
| 1 | `canCut(A,B)` | A can sever / divide B | A.`edge` (the blade) | B.`on` (cut from above) | A's edge type (smooth/serrated/none), B's softness |
| 2 | `canContain(A,B)` | A can hold B inside it | — (whole object) | — (whole object) | A's concavity & open top, B fitting inside A |
| 3 | `canPourInto(A,B)` | A can pour its contents into B | A.`body` (no spout member exists) | B.`on` (into the opening) | A pourable + has lip, B has open top |
| 4 | `canScoop(A,B)` | A can scoop material out of B | A.`body` (the bowl/concavity of the tool) | B.`on` (scoop from the open top) | A's concavity, B open & holds loose contents |
| 5 | `canFitOnto(A,B)` | A seats onto B (lid → vessel) | A.`on` (mating face) | B.`on` (rim face) | A/B rim geometry match |
| 6 | `canSupport(A,B)` | A's face can stably bear B's face | A.`on`/`under` (face) | B.`on`/`under` (face) | A flat & rigid & large enough, B has a stable resting face |
| 7 | `canSlideOn(A,B)` | A can be translated across B's surface without tipping | A.`under` (face that contacts) | B.`on` (the surface) | A has a flat low base, B is a flat extended surface |

Notes:
- **`canCut` actor = `edge`.** `edge` is already a live member in the codebase
  (e.g. `frying_pan.grasping.types` includes `"edge"` in `real_aff_dict.json`).
  No new vocabulary.
- **`canContain` and `canFitOnto` are whole-object** (no meaningful contact
  part). They are 2-arity-style relations; the user's own
  `"canContain": {"objects": [...]}` edit in `real_aff_dict.json` already
  follows this object-level shape.
- **`canPourInto` / `canScoop` actor = `body`.** There is no `spout` or
  `bowl-of-spoon` member; `body` + the recipient `on` face is the honest
  approximation rather than inventing vocabulary.
- **`canSupport` is the renamed `isP2PSupportive`** — see §4.
- **`canSlideOn` is inter-object on purpose.** Standalone "slidable" loses the
  information of *what* it slides on; the relation keeps it. The support
  surface is a scene entity — see §2.4.

### Tier 2 — Individual affordances (learned, rich predicates; pipeline parity)

2 single-object affordances. **Learned by the thesis model**, emitted in the
**same rich predicate form U-TAMP-2's planner actually consumes** so the output
is genuinely drop-in. Not the thesis's *novel* contribution (that is Tier 1),
but full predicates, not coarse booleans.

| # | Affordance | Emitted predicate (exact arity) | Parametrization |
|---|---|---|---|
| 8 | `canGraspFrom` | `(canGraspFrom <obj> <member> <face> <approach>)` — **4-arity** (`afford.py:732, 780–791`) | actor `member` (`body`/`handle`/`edge`) + contact `face` + `approach` direction pair |
| 9 | `canLift` | `(canLift <obj> <member>)` — **2-arity** (`MEM_AFF` schema, `afford.py:54`) | actor `member` |

These keep the canonical names (not "graspable"/"liftable" booleans) — they
**are** the existing U-TAMP-2 predicates, now produced by the learned model
instead of the rule layer.

**Label source (the honest answer to "where do member+approach labels come
from"):** the model is trained to **imitate U-TAMP-2's own rule-based emitter**
run on the training scenes. The existing `_parse_any_affordances()` already
produces ground-truth `canGraspFrom` / `canLift` for any scene with object
attributes; those become the supervision targets. So Tier 2 is distilled from
the rule layer at exactly its current truth-level — the same relationship Tier 1
has to the VLM teacher. No new grasp-pose annotation effort.

> **Defense note — state this proactively.** Split the claim by what RGB can and
> cannot do. The **part/approach structure** (which member is a handle, which
> face is up, which approach clears it) is visual-semantic and learnable from
> RGB. The **metric-fit component** (does a 7 cm gripper physically close on
> this object) depends on absolute scale, which no RGB-only method recovers —
> U-TAMP-2 itself only gets it from a **VLM-estimated** `bbox_size` (primed by
> hardcoded few-shot dims in `llm_prompts_affd.py`, not sensed). The thesis
> model therefore reproduces `canGraspFrom` at the **same imperfect truth-level
> the rule layer already operates at**, inheriting — not claiming to fix — the
> metric-scale limitation. The novelty is Tier 1; Tier 2 is faithful
> reproduction for pipeline completeness.

### Dropped affordances (and why)

The filter: **an individual affordance is standalone only if it is not merely
one side of an inter-object relation.**

| Dropped | Reason |
|---|---|
| `supporting` / `isSupportive` (individual) | only ever the actor side of `canSupport` |
| `placing` / `isSupportable` (individual) | only ever the recipient side of `canSupport` |
| `sliding` (standalone individual) | promoted to inter-object `canSlideOn` (carries strictly more information) |
| `cuttable`, `containable`, `pourable` (as standalone outputs) | each is only the recipient side of the corresponding Tier 1 relation; they exist only implicitly inside the pair predicate, never emitted alone |

---

## 2. Object Definitions

22 object types. Core from the YCB (Yale–CMU–Berkeley) object set — textured
meshes available for Blender rendering, most physically in the lab. 4
supplements are not in YCB but are already present in the U-TAMP-2 object class
taxonomy in `afford.py`, so they need no extra integration.

### 2.1 Containers & pourables (6)
| YCB # | Object | Salient visual properties |
|---|---|---|
| 24 | bowl | open top, concave, low rim |
| 25 | mug | open top, concave, handle |
| 19 | pitcher base | open top, pour lip, handle, deep |
| 20 | pitcher lid | shallow cap, mates to pitcher |
| 4 | sugar box | closed box, rigid |
| 6 | mustard bottle | closed, squeezable, narrow neck |

### 2.2 Utensils & cookware (6)
| YCB # | Object | Salient visual properties |
|---|---|---|
| 32 | knife | blade + handle; **edge type is the key discriminator** |
| 31 | spoon | shallow concave bowl + handle |
| 33 | spatula | flat thin blade + handle |
| 29 | plate | flat, very low rim |
| 27 | skillet | shallow pan, handle, open top |
| 28 | skillet lid | dome cap, mates to skillet |

### 2.3 Cuttable foods (8)
| YCB # | Object | Salient visual properties |
|---|---|---|
| 11 | banana | soft, elongated |
| 13 | apple | firm, round |
| 14 | lemon | firm, small, round |
| 15 | peach | soft, round |
| 16 | pear | firm, tapered |
| 17 | orange | firm, round, thick skin |
| 18 | plum | soft, small |
| 12 | strawberry | soft, very small |

(8 distinct meshes; function as one "cuttable food" category for affordance
purposes — they differ in softness, which is the `canCut` recipient
discriminator, not in which affordances apply.)

### 2.4 Supplements — not in YCB, already in afford.py taxonomy (4)
| Object | afford.py class | Why included |
|---|---|---|
| cutting board | `boards` | `canCut` needs a support surface; also a `canSlideOn` / `canSupport` surface |
| salt shaker | `shakers_and_dispensers` | `canPourInto` actor with unique dispensing geometry |
| pot + lid | `medium_pots_and_pans` | large container: `canContain`, `canPourInto`, a third `canFitOnto` pair |
| tray | `trays` | `canSupport` / `canSlideOn` for multiple objects; transport surface |

### 2.5 The support surface as a scene entity (resolution)
`canSupport` and `canSlideOn` have a *surface* as recipient. **Resolution: the
surface is always a first-class detected object, never abstract background.**
In the kitchen scenarios the eligible surfaces are themselves objects in the set
— `cutting board`, `tray`, `plate`, `skillet`, the closed `sugar box` top. The
tabletop itself is treated as fixed background and is **not** an affordance
recipient (objects are organized *on the table* by the planner's transport
actions, not via a learned `canSupport(table, X)`). Rationale: the pairwise
model needs an image crop / feature for the recipient; a first-class object has
one, abstract background does not. This keeps every Tier 1 relation a genuine
object–object pair with two real crops.

> `canSupport` vs `canSlideOn` (stated to pre-empt the obvious question):
> `canSupport(A,B)` = A's face can **statically bear** B's weight.
> `canSlideOn(B,A)` = B can be **translated across** A's surface without
> tipping. Related but distinct actions: a tall thin bottle may be supportable
> on a board yet not slidable on it (tips); a flat coaster is both. They are
> separate predicates, not synonyms.

---

## 3. Intersection Matrix

Per P2, cells are **class-level candidate pairs annotated with the
instance-level discriminator** — never a bare ✓. Read each cell as: "this pair
is a *candidate*; the affordance actually holds iff the stated visual
condition holds on the depicted instances." Empty cell = not a candidate
(structurally impossible regardless of instance).

Legend: **A** = object can be the *actor*; **R** = object can be the
*recipient*; `· ` = not applicable.

### 3.1 `canCut(actor, recipient)` — actor.`edge` → recipient.`on`
| Object | Role | Holds iff (instance discriminator) |
|---|---|---|
| knife | A | the blade `edge` is sharp / serrated (a butter knife: **negative**; serrated bread knife: **positive**) |
| 8 fruits (banana…strawberry) | R | flesh soft enough for the depicted blade; thick-skinned (orange) needs a sharper edge than soft (banana) |
| cutting board | R? | only as a chopping *support*, not itself cut — modeled via `canSupport`, **not** `canCut` |
| all others | · | no cutting edge / not a food |

### 3.2 `canContain(actor, recipient)` — whole object, no parts
| Object | Role | Holds iff |
|---|---|---|
| bowl, mug, pitcher base, skillet, pot | A | depicted with an open top and enough internal volume |
| salt shaker | A? | only granular contents — out of scope here; treat as **not** a `canContain` actor (its dispensing is `canPourInto`) |
| small objects: strawberry, plum, lemon, salt shaker, spoon | R | small enough to fit inside the actor instance (size relation judged visually) |
| sugar box, mustard bottle | · | closed — cannot receive a placed object |
| flat objects: plate, cutting board, tray, lids | · | no containing volume |

### 3.3 `canPourInto(actor, recipient)` — actor.`body` → recipient.`on`
| Object | Role | Holds iff |
|---|---|---|
| pitcher base, mug, bowl, pot | A | open + has a lip the contents can leave over (pitcher: strong; bowl: weak/spilly) |
| salt shaker | A | perforated top, granular dispensing |
| mustard bottle | A | squeezable, narrow neck |
| bowl, mug, pot, plate, skillet | R | has an opening wide enough to receive the pour without the actor instance overshooting |
| closed / flat-no-rim objects | · | nothing to pour into |

### 3.4 `canScoop(actor, recipient)` — actor.`body` → recipient.`on`
| Object | Role | Holds iff |
|---|---|---|
| spoon | A | has a concave bowl (a flat spatula: **negative** for scoop) |
| spatula | A? | only for sliding-under solids, not scooping loose contents — treat as **negative** for `canScoop` (kept distinct from `canCut`/lift) |
| bowl, mug, pot, skillet | R | open and holds loose/scoopable contents |
| solid foods, closed objects | · | nothing to scoop / not a container |

### 3.5 `canFitOnto(actor, recipient)` — whole object, rim match
| Object | Role | Holds iff |
|---|---|---|
| pitcher lid | A | rim matches **pitcher base** specifically |
| skillet lid | A | rim matches **skillet** specifically |
| pot lid | A | rim matches **pot** specifically |
| pitcher base / skillet / pot | R | the matching lid instance seats on it |
| everything else | · | not a lid / not a lidded vessel |

(`canFitOnto` is strongly instance/pairing-specific: a skillet lid does **not**
fit a pitcher even though both are (lid, vessel) class pairs — P2 in its
sharpest form.)

### 3.6 `canSupport(actor, recipient)` — actor.face → recipient.face
| Object | Role | Holds iff |
|---|---|---|
| cutting board, tray, plate, skillet, sugar box (closed top), pot lid (inverted) | A | depicted face is flat, rigid, and larger than the recipient's resting footprint |
| almost any object | R | has a stable resting face (a sphere-like fruit: marginal; a flat-bottomed mug: strong) |
| open bowl, pitcher (as actor) | · | concave / no flat bearing face on top |

### 3.7 `canSlideOn(actor, recipient)` — actor.`under` → recipient.`on`
| Object | Role | Holds iff |
|---|---|---|
| salt shaker, mug, sugar box, plate, bowl | A | has a flat low base and a low centre of mass (does not tip when pushed) |
| tall mustard bottle, narrow pitcher | A? | borderline — high CoM; instance-dependent, often **negative** |
| cutting board, tray, plate, skillet | R | flat, extended, rigid surface |
| fruits (round) | · | roll rather than slide — **negative** as actor |

---

## 4. The `isP2PSupportive` → `canSupport` Rename

`isP2PSupportive` is renamed to **`canSupport`** throughout the taxonomy and
thesis.

- **Why.** Under P4 every affordance is part-parameterized, so "P2P"
  (part-to-part) is no longer a distinguishing label — it is true of all of
  them. `canSupport` is verb-consistent with `canCut` / `canContain` /
  `canPourInto` / `canScoop` / `canFitOnto` / `canSlideOn`.
- **No collision.** The old *individual* `isSupportive` / `isSupportable`
  predicates are dropped entirely (see §1), so `canSupport` does not clash with
  any surviving name. PDDL arity further disambiguates.
- **Integration consequence.** This is a pure rename of the existing
  `isP2PSupportive` relation (`afford.py:833`). The 3-point integration
  (detector → PDDL emitter in `_parse_any_affordances()` → consuming domain
  action) still applies for the 6 *new* Tier 1 predicates, which U-TAMP-2 does
  not yet emit.

---

## 5. Summary

- **Tier 1 (learned, novel):** `canCut`, `canContain`, `canPourInto`,
  `canScoop`, `canFitOnto`, `canSupport`, `canSlideOn` — 7 directed
  inter-object relations.
- **Tier 2 (learned, rich, parity):** `canGraspFrom` (4-arity),
  `canLift` (2-arity) — the existing U-TAMP-2 predicates, now learned (distilled
  from the rule emitter) rather than rule-computed; explicitly not the novel
  claim, but full predicates not booleans.
- **22 object types** (18 YCB + 4 supplements), surfaces are first-class
  objects, the tabletop is background.
- **Binary, part-resolved** outputs: per Tier-1 pair the model predicts
  `(actor, actor_part, recipient)` → {0,1}, decided at the **instance** level by
  visual discriminators. The **actor part is model-predicted** (the load-bearing
  novel claim); the **recipient is whole-object in the baseline** —
  recipient-region / graded suitability is the explicit ambitious extension, out
  of baseline scope. Model confidence is optional non-contractual evidence.
- **Part model** reuses only existing `afford.py` vocabulary; actor → member
  (predicted), recipient → face (physical/adapter, not predicted in baseline),
  whole-object affordances un-parameterized.
- **Two-layer design:** the detector produces the rich part-resolved output; a
  per-predicate **PDDL adapter** (possibly lossy — supplies recipient face /
  defaults) is the boundary to Fast Downward. The detector is not constrained by
  PDDL's current arity; the adapter is the downstream concern.
