# affdet — inter-object affordance detection

Predicts whether one object in a photograph can act on another: whether *this*
knife can cut *this* loaf, whether *this* bowl can hold *this* apple. The output
is a set of symbolic predicates a task planner can consume.

Affordance detection usually asks what a single object affords — a knife's blade
cuts, a bowl's interior contains. Many manipulation actions are afforded by no
single object but by a **pair** of them, and the answer depends on the specific
objects in view rather than on their categories. Two knives of the same class can
differ in what they can sever.

A vision–language model can make these judgements but is too slow to query inside
a planning loop. This method distils that judgement into a small head over frozen
DINOv2 features, which runs in ~100 ms per scene.

Bachelor's thesis, University of Innsbruck.

## Acknowledgements

This work builds on two lines of research at the University of Innsbruck:

- **U-TAMP** — *Unified Task and Motion Planning using Object-centric
  Abstractions of Motion Constraints*, Alejandro Agostini and Justus Piater
  ([arXiv:2312.17605](https://arxiv.org/abs/2312.17605)), which encodes
  inter-object affordances as object-centric abstractions of motion constraints.
- **Its perception front end** — *Leveraging Inter-object Affordances for
  Efficient Planning in Contact-rich Tasks*, Pouya P. Niaz, Justus Piater and
  Alejandro Agostini ([arXiv:2608.25641](https://arxiv.org/abs/2608.25641)),
  which obtains affordance predicates by prompting a VLM for object attributes
  and parsing them with a rule layer.

This thesis replaces that rule layer with a learned, instance-sensitive model.
The repository began as a fork of the UTAMP-2 codebase by Pouya Niaz, whose
affordance-detection pipeline (`pipeline/`) is not redistributed here.
Supervised by Pouya P. Niaz and Alejandro Agostini.

## Install

```bash
pip install -e .
```

Python ≥ 3.11. Pulls torch, ultralytics and scikit-learn. DINOv2 is fetched from
`torch.hub` on first use.

## Predict

```bash
affdet predict --image photo.jpg
```

```
actor                 predicate     recipient               score
-----------------------------------------------------------------
knife_0               canCut        apple_0                 0.985 *
knife_0               canPierce     apple_0                 0.948 *
dish_0                canContain    apple_0                 0.566 *
dish_0                canSupport    apple_0                 0.517 *

* = predicate holds at the calibrated threshold.
```

(Real output, on a photograph of a knife, a dish and an apple.)

Pretrained weights are bundled in [`weights/`](weights/) (23 MB) and used by
default. `--json` emits machine-readable output; `--all-pairs` skips the
class-based candidate gate, which helps when the detector mislabels an object.

## Predicates

Six, each over an **ordered** pair — `canCut(knife, apple)` and
`canCut(apple, knife)` are different questions.

| Predicate | Actor → recipient | Holds when |
|---|---|---|
| `canCut` | knife → food | the blade can sever the item |
| `canPierce` | knife, fork → food | the tip can penetrate it |
| `canPeel` | peeler → food | the skin can be removed |
| `canContain` | container → item | the item fits inside |
| `canSupport` | container → item | the item rests stably on it |
| `canPourInto` | vessel → vessel | contents can be poured between them |

## How it works

```
photo ──► YOLOv11-S ──► crops ──► DINOv2 (frozen) ──► class probe ──► names
                                       │
                                       └──► pair vectors ──► candidate gate
                                                                  │
                                                            pairwise head
                                                                  │
                                                          affordance predicates
```

The backbone is frozen; only the pairwise head is trained. Supervision comes from
a VLM teacher queried **per object pair** rather than per scene, so a verdict
obtained once applies to every scene containing that pair — which is what let the
corpus grow from 1,000 to 12,753 scenes without re-querying.

## Results

Calibrated macro F1, by how much is withheld from training. Each level withholds
strictly more than the one above.

| Held out | Macro F1 (6 predicates) | Excluding `canPourInto` |
|---|---|---|
| Unseen scenes | 0.944 ± 0.005 | 0.937 ± 0.005 |
| Unseen pairings | 0.931 ± 0.009 | 0.923 ± 0.014 |
| Unseen actor mesh | 0.884 ± 0.021 | 0.888 ± 0.019 |
| **Unseen both meshes** | **0.880 ± 0.028** | **0.887 ± 0.021** |

Figures are averaged over seeds — six for the instance levels, three for the
others. `canPourInto` is supervised by a hand-authored rule rather than the
teacher, hence the second column.

**Against human annotation**, on the same 394 pairs:

| | Teacher (VLM) | Student | Retained |
|---|---|---|---|
| Macro accuracy vs. human | 83.2 % | 79.9 % | 96 % |

The student reproduces its teacher closely. The teacher itself agrees with a
human on roughly four judgements in five, so what the evaluation establishes is
that distillation transferred the judgement faithfully — not that the judgement
is correct. See the thesis discussion.

**Latency**, measured from the raw image over 100 scenes on one RTX 2070S:

| Stage | Per scene |
|---|---|
| Object detection (YOLOv11-S) | 47.0 ms |
| DINOv2 over detector crops | 42.1 ms |
| Class probe | 0.5 ms |
| Candidate gate and pairwise head | 11.0 ms |
| **Student, complete path** | **≈100 ms** |
| Live VLM, same conditions | ≈18 s |

Both figures are measured from the raw image, so this compares two complete
paths rather than a whole system against part of another.

## Training

```bash
affdet extract --scenes scenes/ --out features/
affdet train --scenes scenes/ --features features/ --labels labels/ --out runs/mine
```

Features are extracted once and cached; the frozen backbone is not re-run per
epoch.

## Dataset

The 12,753-scene corpus and its 59,592 pairwise labels are documented in
[`docs/dataset/README.md`](docs/dataset/README.md). Scenes carry bounding boxes,
instance masks, camera pose and intrinsics.

Scenes are rendered from [OmniObject3D](https://omniobject3d.github.io/) scans
under CC BY 4.0.

## Layout

```
affdet/
├── render/      Blender scene generation, physics settling, coverage sampling
├── label/       VLM teacher: prompt assembly, scoring, manual annotation tool
├── extract/     frozen DINOv2 features and the class probe
├── model/       the pairwise head: dataset, training, inference
├── detect/      detector training, box matching, instance-disjoint splits
├── eval/        metrics, threshold calibration, VLM baseline
├── experiment/  run ledger, seed aggregation, result tables
├── adapter/     PDDL emission
└── tests/       780 tests
```

`scripts/` holds the experiment drivers behind the published numbers. They are
provenance, not a stable interface.

## Limitations

- Trained on synthetic renders; expect degradation on photographs.
- Labels are VLM-generated and carry model error.
- Predicates are asserted over whole objects — the method does not localise which
  part of the actor effects the action.
- Predicates are emitted but never handed to a planner end to end.

## Citation

```bibtex
@mastersthesis{bek2026affordances,
  title  = {Learning-based Detection of Inter-object Affordances in Robotic
            Manipulation Tasks},
  author = {Bek, Maksym},
  school = {University of Innsbruck},
  year   = {2026},
  type   = {Bachelor's thesis}
}
```
