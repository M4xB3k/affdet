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
