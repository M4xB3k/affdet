# Pretrained weights

Three checkpoints, enough to run inference on a photograph end to end.
DINOv2 itself is not stored here; it is fetched from `torch.hub` on first use.

| File | Size | What it is |
|---|---|---|
| `yolo11s_detector.pt` | 19 MB | YOLOv11-S fine-tuned on the rendered corpus. Produces boxes only; its class head is discarded in favour of the probe below. |
| `dinov2_class_probe.pkl` | 160 KB | Multinomial logistic regression over frozen DINOv2 features, naming 52 object classes from a crop. |
|  `student/best.pt` | 3.8 MB | The pairwise affordance head. Trained on the v5 label set (59,592 labels), seed 0, unseen-pairings split. |

## Provenance

The student is run `e1b-pair-s0-v5` from the experiment ledger, trained at
git sha `a53d8402`, best epoch 35. Its calibrated macro F1 on that split is
**0.938**.

Note that the thesis reports **seed-averaged** figures (0.931 ± 0.009 for the
unseen-pairings rung, averaged over three seeds). This checkpoint is one seed
of that rung and scores slightly above the mean; it is not itself "the"
published number. Reproducing the reported figure means retraining across
seeds, not loading this file.

## Verified

`affdet predict` was run end to end against a real photograph
(`live_photos/DSC01933.JPG`) with exactly these three files, on CPU:

```
knife_0   canCut       apple_0   0.985 *
knife_0   canPierce    apple_0   0.948 *
dish_0    canContain   apple_0   0.566 *
dish_0    canSupport   apple_0   0.517 *
```

## Caveats

- `dinov2_class_probe.pkl` is a pickled scikit-learn estimator, saved with
  scikit-learn 1.9.0. Loading it under an older version warns and is not
  guaranteed to behave correctly.
- The probe names 52 classes; `peeler` is absent from it, so `canPeel` pairs
  depend on the object being named by other means.
- All three were trained on synthetic renders. Expect degradation on
  photographs — the thesis measures this gap directly.
