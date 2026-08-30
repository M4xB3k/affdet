# Inter-object Affordance Scenes

A synthetic dataset of tabletop scenes annotated with **pairwise functional
affordance labels** — relations that hold between an ordered pair of objects
rather than properties of a single object.

Each scene is one rendered RGB image of a tabletop arrangement, with per-object
bounding boxes, instance masks, camera pose and intrinsics, and graded
affordance verdicts for every admissible ordered pair of objects in the scene.

| | |
|---|---|
| Scenes | 12,753 |
| Pairwise affordance labels | 59,592 |
| Object instances | 186, across 46 categories |
| Predicates | 6 |
| Image resolution | 960 × 720 |
| Total size | ~3.2 GB (renders JPEG q95; masks lossless PNG) |
| License | CC BY 4.0 |
| DOI | [10.5281/zenodo.22209205](https://doi.org/10.5281/zenodo.22209205) |

## Why this dataset exists

Public affordance datasets annotate what a *single* object affords: a knife's
blade cuts, a bowl's interior contains. Many manipulation actions are afforded
by no single object but by a pair of them — whether a loaf can be sliced
depends on the knife and the bread together. No public dataset carried pairwise
functional affordance labels, so this one was generated.

The labels were produced by a vision–language model acting as an annotator, not
by human annotation. See [Label provenance](#label-provenance) for what this
means for their reliability.

## Layout

```
scenes/
  scene_v1_0001/
    scene_v1_0001.jpg        RGB render, 960×720, JPEG q95
    scene_v1_0001.json       scene metadata (below)
    bowl_016_mask.png        binary instance mask, one per object
    knife_003_mask.png
    sausage_054_mask.png
  scene_v1_0002/
  ...
labels/
  scene_v1_0001.json         pairwise affordance labels for that scene
  ...
  label_coverage.json        corpus-level counts per predicate
```

Scene IDs carry a generation prefix (`v1`, `v2`, `v3`) reflecting three rounds
of rendering: 1,000 + 3,753 + 8,000 scenes. The generations differ in scene
composition strategy, not in format — all three share one schema and are used
together as a single corpus.

### Scene metadata

```json
{
  "scene_id": "scene_v1_0001",
  "image_path": "scene_v1_0001.jpg",
  "image_size": [960, 720],
  "source": "blender",
  "objects": [
    {
      "obj_id": "bowl_016",
      "obj_class": "bowl",
      "bbox": [619.0, 411.0, 791.0, 552.0],
      "mask_path": "bowl_016_mask.png",
      "not_upright": false,
      "visible": true,
      "visible_px": 18979
    }
  ],
  "camera": {
    "pose": [[...4×4 world-to-camera...]],
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },
  "template": "cov_canPierce_pos",
  "instance_ids": ["bowl_016", "knife_003", "sausage_054"],
  "realised_affordances": [["canPierce", "pos"]]
}
```

- `bbox` is `[x_min, y_min, x_max, y_max]` in pixels.
- `obj_id` is the stable instance identifier (`bowl_016`); `obj_class` is its
  category (`bowl`). Two instances of one category are different objects with
  potentially different affordances — that distinction is the point of the
  dataset.
- `visible_px` counts the object's pixels **after** occlusion by other objects.
  `visible` is false when that count falls below the usability threshold; such
  objects are still rendered but were excluded from label coverage.
- `not_upright` marks containers detected as flipped after the physics drop, so
  that an upside-down bowl is not treated as able to contain.
- `camera.pose` is a 4×4 world-to-camera matrix; `intrinsics` is the 3×3 pinhole
  matrix. Together with the masks these support depth-free 3D and pose work
  beyond the affordance task.
- `template` names the generator recipe that produced the scene, and
  `realised_affordances` records which predicate/polarity the scene was composed
  to exhibit. Both are provenance, not labels — do not train on them.

### Labels

```json
{
  "scene_id": "scene_v1_0001",
  "tier1": [
    {
      "actor": "bowl_016",
      "recipient": "sausage_054",
      "predicate": "canContain",
      "score": 0.222,
      "verdict": 0,
      "confidence": 0.222,
      "source": "vlm_distill_v1"
    }
  ]
}
```

Labels are **ordered**: `actor` acts on `recipient`, and swapping them is a
different question. `canCut(knife, apple)` and `canCut(apple, knife)` are not
the same label and do not have the same answer.

- `score` is the graded verdict in `[0, 1]`, rescaled from the annotator's 0–9
  integer scale. This is the training target.
- `verdict` is `score` thresholded at 0.5, provided for convenience. Prefer
  `score`: the graded value distinguishes a marginal pair from a clear one, and
  that distinction is lost by binarising.
- Not every pair is labelled for every predicate. A pair is labelled only where
  the relation is structurally possible — no arrangement of a lemon and a pear
  has one cutting the other, so that pair carries no `canCut` label. Absence of
  a label is not a negative label.

### Predicates

| Predicate | Actor → recipient | Holds when | Labels |
|---|---|---|---|
| `canContain` | container → item | the item fits inside | 15,888 |
| `canSupport` | container → item | the item rests stably on it | 15,888 |
| `canPierce` | knife, fork → food | the tip can penetrate | 13,584 |
| `canCut` | knife → food | the blade can sever | 8,405 |
| `canPourInto` | vessel → vessel | contents can be poured between | 3,054 |
| `canPeel` | peeler → food | the skin can be removed | 2,773 |

Positive/negative balance varies sharply by predicate — `canPierce` positives
outnumber negatives roughly thirteen to one. Threshold calibration on a
validation split handles this better than resampling; see the thesis appendix.

## Label provenance

Labels were generated by **GPT-5-mini** (via Azure OpenAI) shown multi-view
renders of each object pair, scored 0–9 per predicate, then rescaled. Labels
were assigned **per object pair**, not per scene: a verdict obtained for
`(bowl_016, sausage_054)` is attached to every scene containing that pair. This
is what made the corpus extensible from 1,000 to 12,753 scenes without
re-querying.

**These are model-generated labels and carry model error.** Measured against
human annotation on a held-out sample, the annotator agrees with a human on
roughly 83% of pairs, and disagrees with *itself* on about one pair in nine when
asked the same question repeatedly. Treat them as a strong but imperfect signal,
not as ground truth. Bounding boxes, masks, classes, and camera parameters are
exact — they come from the rendering geometry, not from a model.

## Attribution

Scenes are rendered from 3D object scans from
[OmniObject3D](https://omniobject3d.github.io/), used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Changes made to the source material:** objects were scale-calibrated to
consistent physical size, composed into novel tabletop arrangements via
rigid-body physics simulation, and rendered to new RGB images with generated
annotations. The original 3D scans are **not** redistributed here; obtain them
from the OmniObject3D project.

```bibtex
@inproceedings{wu2023omniobject3d,
  title={{OmniObject3D}: Large-Vocabulary 3D Object Dataset for Realistic
         Perception, Reconstruction and Generation},
  author={Wu, Tong and Zhang, Jiarui and Fu, Xiao and Wang, Yuxin and
          Ren, Jiawei and Pan, Liang and Wu, Wayne and Yang, Lei and
          Wang, Jiaqi and Qian, Chen and Lin, Dahua and Liu, Ziwei},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## License

This dataset is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), matching its source
assets. You may share and adapt it, including commercially, with attribution.

## Citation

```bibtex
@misc{bek2026affordancescenes,
  title  = {Inter-object Affordance Scenes},
  author = {Bek, Maksym},
  year   = {2026},
  note   = {Dataset accompanying the bachelor's thesis
            "Learning-based Detection of Inter-object Affordances in Robotic
            Manipulation Tasks", University of Innsbruck},
  doi    = {10.5281/zenodo.22209205}
}
```

## Known limitations

- **Synthetic only.** Every scene is rendered. Models trained on it face a
  sim-to-real gap on photographs.
- **Labels are model-generated**, with the error rates given above.
- **Objects come from one source dataset**, so results measured here may not
  transfer to a different object set.
- **Pairwise labels ignore arrangement.** A verdict depends on the two objects,
  not on where they sit in the scene, so predicates that would depend on
  reachability or occlusion are out of scope.
- **No part localisation.** Labels state that a relation holds, not which part
  of the actor effects it.
