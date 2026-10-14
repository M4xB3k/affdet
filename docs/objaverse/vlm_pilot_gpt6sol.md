# VLM pilot: gpt-6-sol on 20 salt-shaker objects × 3 repeats

Tokens: 71,100 in / 25,860 out (≈ $0.40 at $2 / $10 per M).

## Consistency (all repeats give the same answer)

| Question | Objects |
|---|---|
| has_handle | 20/20 |
| handle_face | 20/20 |
| body_material | 15/20 |
| cap_material | 14/20 |
| fragile | 18/20 |
| body_cross_section | 17/20 |
| top_shape | 17/20 |
| stands_upside_down | 16/20 |
| rests_on_sides | 17/20 |
| top_can_hold | 14/20 |

## Agreement of the majority answer with the geometry labeller

| Question | Agree |
|---|---|
| stands_upside_down | 17/20 (85 %) |
| top_can_hold | 15/20 (75 %) |
| rests_on_side (per face) | 59/80 (74 %) |
| lies on any side | 15/20 (75 %) |

Per-object answers: `vlm_pilot_gpt6sol.csv`.
