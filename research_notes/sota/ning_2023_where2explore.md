# Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects
**Authors**: Chuanruo Ning, Ruihai Wu, Haoran Lu, Kaichun Mo, Hao Dong
**Year / Venue**: 2023, NeurIPS 2023 (as listed on the arXiv page; the PDF header I read says "Preprint. Under review", v2 of 15 Dec 2023)
**Cite key**: ning_2023_where2explore
**Source read**: https://arxiv.org/abs/2309.07473 (abstract page); https://arxiv.org/pdf/2309.07473 (full PDF with appendix, text extracted locally and read in full)

## In one paragraph
Affordance models trained on a few object categories fail on new categories. Where2Explore tackles this by letting the robot try a small number of interactions on a few instances (10 per new category), but choosing those interactions cleverly. It learns a "similarity" score: how well the current affordance model already predicts a local geometry. Interactions go where the similarity is lowest, meaning the model is least sure and the geometry is new. After fine-tuning on these few trials, the model works on other unseen objects of that category without further interaction. It keeps the Where2Act style per-point affordance map and adds a second per-point map that says which regions are worth exploring.

## Input and output
- Input: an N-point partial point cloud O in R^(N x 3) of an articulated object (N not given in the main text) and, per point, a set of action directions and gripper orientations.
- Output of the affordance module: per point and action, a score in [0, 1] saying whether the action will move a part.
- Output of the similarity module: a per-point similarity Sim in R^N (also conditioned on the action), where high means "geometry the affordance model already handles".
- Two action types: pushing and pulling, with Where2Act's scripted short trajectories.

## Architecture
- One PointNet++ segmentation encoder, shared by both modules, giving a 128-d feature per point.
- Two decoders, each an MLP with one hidden layer of size 128: the affordance decoder and the similarity decoder. (Same head design as Where2Act.)
- Affordance loss: binary cross-entropy against "part motion reached 0.01".
- Similarity loss: L1 between the predicted similarity and a ground-truth "accuracy" of the affordance prediction on that interaction. Concretely, for objects in the similarity categories, Sim is trained to equal how accurate the affordance module was on that action (the paper writes this as the ratio of correct predictions to attempts during training). Only the affordance category updates the affordance module; the other similarity categories only train the similarity module.
- Few-shot loop on a novel category: predict similarity, sample actions in diverse directions, pick the one with the lowest similarity, run it, then update both modules with the result. Stop when similarity passes a bar or the budget runs out.
- Parameter count, learning rate and batch size: not verified in the paper text (the paper points to the supplementary but the appendix I read only has error-control tables and the real-world description).

## Simulation and labels
- Simulator: SAPIEN with PhysX, Panda flying gripper, RGB-D camera 5 units from the object, clean segmentation assumed (as in Where2Act and AdaAfford).
- Data: 942 articulated objects in 14 PartNet-Mobility categories. Only 3 training categories and 11 novel ones. Four different training-category combinations are used ({cabinet, faucet, window}, {cabinet, switch, refrigerator}, {table, faucet, refrigerator}, {table, switch, window}); the first category in each is the "affordance category". Few-shot learning uses 10 instances per novel category.
- Interaction and success: as in Where2Act, a scripted push or pull; success if part motion reaches 0.01.
- Interaction budgets of 1, 2 and 5. The paper says the method uses only 0.3 % of the data of the full-data upper bound.
- Number of training interactions used in the first stage: not verified in the paper text.

## Evaluation and headline results
- Metrics: F-score of the affordance prediction, and sample-success-rate (choose one action predicted as successful at random, run it, count the fraction of successes). Results are averaged over the four training-category combinations.
- Baselines: Where2Act, AdaAfford, PointEncoder (a pre-trained Point-BERT-style transformer encoder), plus ablations No-explore, Explore-random, Explore-noSim and a Full-data upper bound.
- Numbers (budgets 1 / 2 / 5): pushing F-score 35.4 / 38.5 / 41.6 versus Where2Act 25.6 / 28.0 / 30.4 and AdaAfford 27.5 / 29.7 / 32.0. Pulling F-score 12.1 / 12.5 / 24.2 versus Where2Act 6.4 / 7.5 / 8.5. Sample-success pushing 31.3 / 37.3 / 39.5.
- Ablation: No-explore F-score is 20.6 (pushing) and 3.9 (pulling); Full-data is 47.9 and 27.1. So five exploration trials recover a large share of the gap for pushing and most of it for pulling at budget 5.
- Real world: a Franka Panda arm and an Azure Kinect depth camera. The model, trained only on cabinets, windows and faucets, explores four mugs and then pulls the lid of a fifth mug. Qualitative demonstration, no success rate.

## How it differs from our setup
- Problem: cross-category adaptation with a few interactions at test time. Our problem is held-out generalisation without any test-time interaction.
- Simulator, label definition, backbone and single-object scope: as for Where2Act (SAPIEN, part-motion success, PointNet++, one object).
- Dense versus sampled: they spend a tiny interaction budget (1 to 5 trials per object) and must decide where to spend it. We have complete labels for every object, so the "where to explore" problem does not exist in our setting.
- Evaluation: their gap between seen and novel categories (for pulling, F-score drops from the full-data 27.1 to 3.9 without exploration) shows how hard category transfer is. Our 5-fold split by object is easier than a split by category.
- Worth copying: hold out whole categories (not just random objects) in an extra test. Our objects come from categories such as boards, plates and mugs, so a "leave one category out" fold would show whether the heatmap transfers to a new shape family. Also, the idea of using the model's own error on held-out geometry as a map of "where the model is weak" could be turned into a diagnostic: plot our per-point residuals on the object to see which local geometries fail.

## Likely supervisor questions
1. How would your model do on a new category? Our 5-fold split holds out objects, not categories, so we cannot claim category transfer yet. This paper shows that transfer can be much worse than object transfer (pulling F-score 3.9 without exploration versus 27.1 for a model trained on all categories). A leave-one-category-out fold is an easy test to add.
2. Could you use their similarity idea? As a diagnostic yes: train a small head to predict our own per-point error and use it to flag uncertain regions in a new object. It is not needed to generate labels because our grid labels everything.
3. Why not do few-shot adaptation on the real robot? Our labels are cheap in simulation. Adaptation helps when real physics differs from the simulation, which is outside the scope of this thesis but is a natural extension.

## Not verified
- Number of points N, parameter count, learning rate and batch size.
- The exact definition of the "accuracy" target of the similarity module beyond "ratio of correct predictions to total attempts".
- Number of first-stage training interactions.
- Whether the real-world demonstration has a quantitative success rate (only qualitative in the text I read).
- The NeurIPS venue is taken from the arXiv listing; the PDF text I read says "Preprint. Under review".
- Choice of paper: the brief allowed a swap for a clearly more relevant 2023 to 2025 paper on simulation-labelled object-object, placement or stacking affordance on point clouds. I ran one web search and a forward-citation listing of O2O-Afford (Semantic Scholar, titles only). Candidates by title were SPOTS (arXiv 2309.13937, stable placement, simulation-based stability check, abstract only, no peer-reviewed venue named), Xu et al. 2022/2023 stable placements (arXiv 2205.06970, abstract only), a 2025 placeability metric (arXiv 2510.14584, abstract only), PreAfford (arXiv 2404.03634, abstract only), Shelving, Stacking, Hanging / RPDiff (arXiv 2307.04751, title only) and O3Afford (arXiv 2509.06233, title only). None was shown to be clearly more relevant (none is confirmed to use simulation-labelled per-point maps on point clouds), so Where2Explore stays. RPDiff and O3Afford are worth a separate look for the thesis.

## Verification (second reader, 2026-11-16, numbers only)
Every number in this note was searched in the extracted paper text and found, except the ones that are ours or come from other papers (arXiv ids of later papers). Which table row or column a number belongs to was not re-checked; quote those with the table open.

## BibTeX
```bibtex
@inproceedings{ning2023where2explore,
  title     = {{Where2Explore}: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects},
  author    = {Ning, Chuanruo and Wu, Ruihai and Lu, Haoran and Mo, Kaichun and Dong, Hao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2309.07473},
  archivePrefix = {arXiv}
}
```
