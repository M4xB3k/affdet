# Objaverse kitchenware candidates for the UTAMP-2 object roles

`kitchen_candidates.csv` lists Objaverse 1.0 objects whose VLM-assigned type
matches one of the five UTAMP-2 object roles (plus plates/bowls as the
contain-vs-support contrast).

Source: DeepMind `objaverse_annotations` (Kabra et al., ICML 2024,
<https://github.com/google-deepmind/objaverse_annotations>, CC-BY 4.0) —
PaLI-X type and material labels for 763,839 Objaverse 1.0 objects. These are
not the Objaverse-LVIS labels (the LVIS file on Hugging Face was unreachable
from the build environment); they cover all of Objaverse, are VLM-generated and
noisy, and carry a probability (`type_prob`). Per-object licences are not
included — filter with the Objaverse metadata before redistribution.

| Role | Types matched | All | type_prob ≥ 0.5 | Top materials |
|---|---|---|---|---|
| salt shaker | saltshaker, shaker, pepper mill | 183 | 61 | plastic, ceramic, glass (14), brass, wood |
| mug | mug, cup, teacup, coffee cup, beer mug | 2,293 | — | ceramic, wood, plastic; glass: 18 |
| cutting board | cutting board | 138 | 59 | wood (121), plastic (17) |
| frying pan | frying pan, wok | 200 | 100 | iron (80), metal, aluminium, steel |
| tray | tray, platter | 1,026 | 165 | plastic (383), wood (157), cardboard, ceramic |
| plate / bowl | plate, dish, bowl, dinner plate | 1,250 | — | ceramic, clay, stoneware |

Role totals in the table come from the counting script; the CSV holds the exact
rows (5,090). The "type_prob ≥ 0.5" column was computed with a slightly wider
type list for mug and plate/bowl, so it is left blank there.

## ObjaTHOR size / mass annotations

`objathor_kitchen.csv` joins the candidates with the ObjaTHOR 2024_08_16
annotations (allenai/objathor, GPT-4o; 625,252 objects), built with
`scripts/join_objathor.py`. 4,539 of 5,090 candidates are covered. Columns:
height / max dimension (cm), mass (kg), materials, quality (0–10), licence and
the metric THOR bounding box (m, y = up). Values are coarse (5 cm / 0.5 kg
steps) and the bbox is derived from the same scale, so it is not an independent
check. Salt-shaker heights and some upright cutting boards are off — clamp to
category ranges before use. Check the annotations' licence before redistribution.

## Curated salt-shaker set

`salt_shaker_set.csv` is a hand-picked set for the salt-shaker role, chosen by
looking at Sketchfab thumbnails of 259 candidates: the 183 DeepMind-labelled
candidates above, Objaverse-LVIS `saltshaker` / `shaker` / `pepper_mill` /
`condiment`, and every ObjaTHOR object whose category or description names a
salt/pepper shaker, pepper mill or spice jar. Name-based filtering alone was not
enough: ObjaTHOR often calls real shakers "container" or "bottle", and calls
cocktail shakers and protein bottles "shaker".

| Tier | Count | Why it is in the set |
|---|---|---|
| shaker | 21 | the benchmark object's own kind |
| pepper_mill | 11 | same affordances (wrap grasp on the body, top supports nothing); taller |
| spice_jar | 7 | same affordances and size; adds label and lid appearance |
| shaker_from_pair | 5 models → 7 shakers | two shakers in one model, split with `scripts/split_pairs.py`; the `note` column says which halves to keep (identical twins count once) |

`target_height_cm` / `target_mass_kg` are the sizes to scale to: ObjaTHOR's value
when it is plausible for the tier (shaker 6–14 cm, mill 12–25 cm, jar 8–14 cm),
otherwise the tier default (9 / 18 / 10 cm). Picks are from thumbnails only —
open the meshes before rendering. In total: 46 objects (28 shakers, 11 mills, 7 jars). Five objects have no licence in ObjaTHOR;
check them in the Objaverse metadata.

## Per-face labels for the salt-shaker set

`salt_shaker_face_labels.csv` holds geometry-derived UTAMP-2 labels for the 46
salt-shaker objects. It was built with `scripts/label_faces.py`, which uses
`affdet/geometry/` (install with `pip install -e .[geometry]`).
`salt_shaker_face_labels_overview.jpg` shows every object as it was labelled.
Per object, the script:

1. **Loads the GLB** from Hugging Face Objaverse. For two-object models it
   loads only the half that the `note` column names.
2. **Uprights it.** The glTF +Y axis becomes up. The `up_axis` column
   overrides this; `-x` is set for the two crank grinders modelled lying down.
3. **Levels it** onto its largest near-horizontal hull facet, which fixes
   scans that are tilted a few degrees.
4. **Aligns the yaw** of the footprint's minimum-area rectangle with x/y.
5. **Scales it** to `target_height_cm`.
6. **Labels the six box faces.** `on`/`under` are ±z, `front` is −y (the glTF
   front), `back` is +y, `left` is −x and `right` is +x.

| Column | Meaning |
|---|---|
| `supportable_<face>` | The object rests stably with that face down. The contact polygon is made of hull facets within 5° of flat. The COM must lie inside it by at least 5 mm, and by at least half the face width × sin 10°. The second bound is the strip a finely tessellated round body of that size would give. Tipping it over must need ≥ 5°. Resting directions up to 20° from the face count, because a ridged side settles onto a facet. |
| `tip_deg_<face>` | The tilt needed to tip the object over in that rest. |
| `supportive_on` | The flat region on top is ≥ 3 cm across and the object stands on `under`. |
| `top_flat_width_cm` | The narrowest width of that flat region. |
| `grasps` | `palm:finger1-finger2` for a Franka Hand (80 mm opening). The fingers reach 4 cm past the palm face, and all material in that reach must fit between the open fingers. Grasps from every face are listed, including `under`. |
| `review` | Faces that are supportable on geometry but rest on a few smooth-shaded facets. The low-poly model is probably meant to be round; the VLM or a person decides. |

The COM is the centroid of the solid convex hull, which suits a filled shaker.
Mugs and pans will need a better estimate.

Results: all 46 stand on `under`. 26 can stand upside down (flat cap). 19 can
lie on at least one side (square, octagonal or faceted bodies). 8 can support
something on top. 5 are flagged for review. Every object except the widest
mills (9–12 cm) can be grasped from the top and the sides.

`tip_deg_<face>` is kept so that borderline cases can become soft labels; a
handful sit within a degree or a millimetre of the thresholds.

Several rows share one mesh with a different texture: d8e2236b/270b9b45,
05308bf4/76fcdf15, 1f4e8a33/a991699c and 3437607e/f71db294. Keep each pair in
the same split. d8e2236b and 270b9b45 differ only in target height (12 vs
10 cm), and their side labels differ too, because the margin sits at the 5 mm
threshold.

## VLM pilot (GPT-6-sol)

`vlm_pilot_gpt6sol.md` / `.csv` hold the results; `scripts/vlm_face_pilot.py`
runs them. Twenty of these objects were each sent 3× to GPT-6-sol (Azure) as
one face-labelled 4-view render. The questions cover handle, materials,
fragility, body cross-section and top shape. Three more questions overlap with
the geometry labels: stands upside down, lies on which sides, top can hold
something. The pilot cost $0.40 (97k tokens).

Findings:

- **Handles are reliable.** Both crank grinders got `has_handle` with the
  right face, 3/3; no false handles. Every repeat agreed on every object.
- **Materials are the least consistent** (body 15/20, cap 14/20 identical
  across repeats). Use the majority of 3 and have a person check the splits.
- **Near-metric questions should stay with geometry.** The VLM agrees with
  the labeller on 75–85 % of objects, and the misses go the way size
  blindness predicts:
  - It says 1.5–2 cm tops and a knob can hold a coaster.
  - It says a square jar with a wider round lid (b8a406f8) lies on all four
    sides, but the lid makes it roll on two of them.
  - It calls one flat-sided body square yet says it lies on no side
    (d00ef026, 2/3).
- **It does help with the review flags.** It calls two of the five
  smooth-shaded low-poly bodies round (d8e2236b, e0b380d2, 3/3), so their
  side labels should be dropped. It calls two faceted (e07a892b, 5ba5959e).
  The novelty rocket (9ef6c018) gets "faceted" but "lies on no side".
  Resolving flags with `body_cross_section` and a person checking the
  leftovers is the planned workflow.

This matches the decision in `docs/utamp2_integration_notes.md`: numbers from
geometry, and the VLM only for non-metric attributes.
