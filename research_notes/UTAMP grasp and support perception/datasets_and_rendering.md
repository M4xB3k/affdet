# 3D object/mesh datasets for kitchenware grasp and support affordances, and rendering practicalities (CoppeliaSim, BlenderProc, others)

Scope note: the target categories are the 5 UTAMP-2 objects (salt shaker, glass mug, cutting board, frying pan, wooden tray) and their neighbours (cup, teacup, pepper mill, saucepan, wok, pan, platter, plate, bowl, serving dish). The thesis previously used OmniObject3D: per the requester, its curated kitchen subset had only 4 pans and no trays, cutting boards or salt shakers. That claim comes from the requester and I did not re-verify it.

Environment limits: huggingface.co, arxiv.org, ar5iv, openaccess.thecvf.com, manual.coppeliarobotics.com, coppeliarobotics.com, ycbbenchmarks.com and raw.githubusercontent.com could not be fetched. I tried to download the Objaverse `lvis-annotations.json.gz` from Hugging Face and from two mirrors, and all three were refused by the proxy. GitHub code search is also blocked. Because of this, **no per-category Objaverse-LVIS counts could be computed**. That is the largest gap in these notes, and closing it takes about 5 lines of Python on a machine with HF access (see Gaps). Some facts below come from search-engine snippets of the primary pages, and those are marked.

---

## Q1. Objaverse / Objaverse-LVIS / Objaverse++: size, our categories, licenses, scale/orientation, quality subsets

### Takeaway
Objaverse 1.0 is the only candidate large enough to give many instances per category for our full list. Its LVIS subset has about 47K objects over 1,156 LVIS categories, and it includes LVIS names such as `saltshaker`, `pepper_mill`, `mug`, `cup`, `teacup`, `chopping_board`, `frying_pan`, `saucepan`, `wok`, `tray`, `platter`, `plate`, `bowl` and `serving_dish`. The meshes have **no reliable metric scale** and arbitrary orientation, and each object carries its own CC licence. Filtering to CC-BY/CC0 plus Objaverse++ quality tags is the practical way to build a publishable subset. Per-category counts were not obtainable here.

### Cited Findings
- Objaverse-LVIS is "a 47K LVIS categorized object subset … comprised of objects uniquely assigned to one of 1156 LVIS categories." The annotators were given "sets of 500 candidate objects for each LVIS category," chosen by CLIP-based prediction. The authors note that popular categories "have substantially more objects that could be included in future annotations." — [Objaverse paper, arXiv 2212.08051 (search snippet)](https://arxiv.org/pdf/2212.08051)
  - Implication: the 500-candidate cap bounds LVIS counts per category. Any category has at most 500 labelled objects, and probably far fewer for rare items like `saltshaker`. This is an inference from the procedure, not a measured count.
- Another source counts "44,199 [objects] with LVIS categories" (out of 763,844 rendered objects), and one later paper says the LVIS subset spans 1,230 categories. The figures differ by source and version. — [search summary citing PaliGemma / other papers](https://arxiv.org/pdf/2407.07726); [Objaverse CVPR'23 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Deitke_Objaverse_A_Universe_of_Annotated_3D_Objects_CVPR_2023_paper.pdf)
- Licensing: "the use of the dataset as a whole is licensed under the ODC-By v1.0 license. Individual objects … are all licensed as creative commons distributable objects," with per-object licence metadata. Breakdown for the full ~800K corpus: CC-BY 4.0 ≈ 721K; CC-BY-NC 4.0 ≈ 25K; CC-BY-NC-SA 4.0 ≈ 52K; CC-BY-SA 4.0 ≈ 16K; CC0 1.0 ≈ 3.5K. This breakdown covers the full corpus, not the LVIS subset. — [HF dataset card via PyPI mirror](https://pypi.org/project/objaverse/0.0.4); [HF card](https://huggingface.co/datasets/allenai/objaverse)
- Precedent for a CC-BY-filtered LVIS subset in grasping: GraspGen took "the subset of meshes that matches LVIS categories and carries a CC BY license, totaling 36,366 meshes." It then randomly selected 8,515 objects to match ACRONYM's size, and labelled grasps by simulating a shaking motion in Isaac Sim. The released data has "over 57 million grasps" for "8515 objects from the Objaverse XL (LVIS) dataset" and targets the Franka Panda gripper and the Robotiq-2F-140. — [GraspGen, arXiv 2507.13097 (search snippet)](https://arxiv.org/pdf/2507.13097); [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen); [HF PhysicalAI-Robotics-GraspGen card](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen/blob/9edc7cf265cf8080f502e7e79e3075d4535d96c0/README.md)
  - GraspGen's own justification is that Objaverse is "more permissive, larger, and more diverse" than ShapeNetSem. — same source
  - The GraspGen demo takes a `--mesh_scale` argument, which suggests scale is supplied per object. How GraspGen normalised Objaverse scale is **unverified**. — [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- Objaverse++ (Lin et al., ICCV 2025 Workshops) had annotators label 10,000 objects by hand with quality and aesthetic tags. A network trained on those labels then tagged the rest, about 790K objects (one snapshot says ~500K, but the official proceedings say ~790K). — [ICCVW 2025 page](https://openaccess.thecvf.com/content/ICCV2025W/CDEL/html/Lin_Objaverse_Curated_3D_Object_Dataset_with_Quality_Annotations_ICCVW_2025_paper.html); [arXiv 2504.07334](https://arxiv.org/pdf/2504.07334)
  - Tags: a 4-level quality score (Low, "No semantic meaning … cannot identify or are corrupted"; Medium; High; Superior), plus binary traits Transparency, Scene, Single Color, Not a Single Object and Figure. Experimental density and art-style tags (Scanned, Arcade, Sci-Fi, Cartoon, Anime, Realistic, Other) are "not included in the paper." Licence ODC-By v1.0, hosted at `huggingface.co/datasets/cindyxl/ObjaversePlusPlus`. — [ObjaversePlusPlus GitHub](https://github.com/TCXX/ObjaversePlusPlus)
  - The authors report that models pre-trained on the quality subset outperform full-Objaverse training for image-to-3D. — [arXiv 2504.07334](https://arxiv.org/pdf/2504.07334)
- Download method: the `objaverse` Python package (`objaverse.load_lvis_annotations()`, `objaverse.load_annotations(uids)` for licence/metadata, `objaverse.load_objects(uids)` for GLB files). The package is listed on PyPI. — [PyPI objaverse](https://pypi.org/project/objaverse/0.0.4). The function names are from my prior knowledge of the package API and could not be re-read from the HF card here.

### Inferences
- **The Scale and Orientation tags are useful for us.** Objaverse objects are user-uploaded Sketchfab GLBs. Their units are arbitrary (cm, m, or unit-normalised), and up-axis and canonical orientation are inconsistent. Nothing I found says Objaverse provides metric scale; GraspGen's per-mesh `--mesh_scale` is consistent with that. The pipeline should therefore rescale each mesh to a sampled real-world size, for example a frying pan of 20–30 cm diameter or a salt shaker of 7–12 cm height (sizes would come from product specs), and align the up-axis. That could mean picking the smallest-extent axis for flat objects like boards, trays and plates, or checking visually.
- **Transparency:** for the "glass mug", Objaverse++'s Transparency tag can either select or exclude see-through mugs. Transparent glass is hard to render faithfully in CoppeliaSim's OpenGL path (Q8).
- **Licence filter for publishing:** keep CC-BY 4.0 and CC0 objects only. Drop NC and SA objects if the derived dataset will carry a single permissive licence. Keep the per-object attribution list (uid, author, licence URL), because CC-BY requires attribution in any redistribution of renders or meshes.
- **Parts:** Objaverse has no part annotations. Handle/body masks would have to come from geometric heuristics, VLM-based or 2D-lifted part segmentation, or manual labelling.
- **Category confusion:** `pan_(for_cooking)` vs `frying_pan`, `saucepan` vs `pot`, and `tray` vs `platter` vs `serving_dish` are likely to overlap in LVIS assignments. Each object was assigned to a single category ("uniquely assigned"), so our category groups should merge neighbouring LVIS labels.

### Gaps
- **Per-category Objaverse-LVIS counts** for `saltshaker, shaker, pepper_mill, mug, cup, teacup, chopping_board, frying_pan, pan_(for_cooking), saucepan, wok, tray, platter, serving_dish, plate, bowl`. No published table, blog post or issue with these counts surfaced, and the annotation file was unreachable from this sandbox. To compute them on a machine with HF access:
  ```python
  import objaverse
  lvis = objaverse.load_lvis_annotations()          # {category: [uid, ...]}
  cats = ["saltshaker","shaker","pepper_mill","mug","cup","teacup","chopping_board","frying_pan",
          "pan_(for_cooking)","saucepan","wok","tray","platter","serving_dish","plate","bowl"]
  print({c: len(lvis.get(c, [])) for c in cats})
  ann = objaverse.load_annotations([u for c in cats for u in lvis.get(c, [])])  # has 'license'
  ```
  Then join the result with the Objaverse++ quality and transparency tags.
- Exact LVIS spellings (for example `saltshaker` vs `salt_shaker`, and `chopping_board`) are as given in the brief and were not verified against the annotation keys.
- I found no Objaverse-specific study quantifying how many meshes are mis-scaled or mis-oriented.

---

## Q2. ACRONYM (Eppner et al. 2021)

### Takeaway
ACRONYM is a simulated grasp dataset: 17.7M parallel-jaw grasps on 8,872 ShapeNetSem meshes in 262 categories, labelled in NVIDIA FleX. The data is **CC BY-NC 4.0** and the meshes must be fetched from ShapeNetSem separately, so a commercially redistributable derived dataset is ruled out. The gripper is reported elsewhere as the Franka Panda, but I could not confirm that from the primary repo.

### Cited Findings
- "ACRONYM is a dataset of 17.7M simulated parallel-jaw grasps of 8872 objects." The grasps span 262 categories, and each is labelled with the outcome from a physics simulator (FleX). — [NVlabs/acronym](https://github.com/NVlabs/acronym); [arXiv 2011.09584 (snippet)](https://arxiv.org/pdf/2011.09584); [NVIDIA publication page](https://research.nvidia.com/publication/2021-05_acronym-large-scale-grasp-dataset-based-simulation)
- Meshes: the user downloads ShapeNetSem meshes from shapenet.org, makes them watertight with Manifold, and simplifies them (`simplify -i temp.watertight.obj -o model.obj -m -r 0.02`). Grasp file names embed a number, for example `_0.0002682457830986903.h5`, which is probably the object scale but **is not documented** on the README. — [NVlabs/acronym](https://github.com/NVlabs/acronym)
- Licence: "The dataset is released under CC BY-NC 4.0"; the code is MIT. — [NVlabs/acronym](https://github.com/NVlabs/acronym)
- Gripper: the README says only "parallel-jaw." A third-party benchmark page lists the gripper as a Franka (Panda) parallel-jaw. GraspGen describes building "a similar dataset for the Franka Panda gripper," following ACRONYM's pipeline. — [codesota ACRONYM page (secondary)](https://www.codesota.com/robotics/grasping/acronym); [GraspGen arXiv 2507.13097](https://arxiv.org/pdf/2507.13097)

### Inferences
- ShapeNetSem has categories like Mug, Bowl, Pan and Plate, and the ACRONYM category list probably includes them, but the per-category list and counts were **not verified**. ShapeNetSem meshes are mostly untextured or poorly textured CAD and carry ShapeNet's non-commercial research terms. ACRONYM grasps are therefore a strong Franka-specific supervision source for mugs, bowls and pans, but a weak source of photoreal textured instances.
- For a publishable derived dataset, GraspGen's Objaverse CC-BY grasps are the more permissive analogue. The GraspGen HF data licence itself was not checked here.

### Gaps
- Per-category ACRONYM counts for mug, bowl, pan, plate, tray, cutting board and salt shaker were not retrieved; the paper table was unreachable.
- Primary confirmation of the Franka Panda gripper model, and of the scale semantics of the filenames.

---

## Q3. EGAD (Morrison, Corke & Leitner 2020)

### Takeaway
EGAD is **not** kitchenware. It is a set of over 2,000 abstract, evolutionarily generated shapes spanning grasp difficulty and geometric complexity, plus 49 3D-printable evaluation objects, licensed BSD-3. It suits generic grasp-robustness testing, not category-level kitchen affordances.

### Cited Findings
- "EGAD! an Evolved Grasping Analysis Dataset for diversity and reproducibility in robotic manipulation" (RA-L 2020) has over 2,000 generated objects for training and evaluating visual grasp detection. Shapes range from simple to complex and grasp difficulty from easy to hard, and there are 49 3D-printable evaluation objects. — [arXiv 2003.01314](https://arxiv.org/abs/2003.01314v3)
- The objects are created with evolutionary algorithms and designed for 1:1 sim-to-real transfer by 3D printing. Licence BSD-3. Project site: dougsm.github.io/egad. — [QUT Research Data Finder](https://researchdatafinder.qut.edu.au/individual/n18178); [Research Data Australia](https://researchdata.edu.au/egad-evolved-grasping-analysis-dataset/1459883)

### Inferences
- EGAD is useful as distractor clutter or as a negative or "unknown object" set, not as a source of mugs, pans or trays.

### Gaps
- None material to our use.

---

## Q4. YCB object and model set

### Takeaway
YCB has 77 objects in 5 groups, and its kitchen group overlaps our list well: mug, bowl, plate, a skillet with glass lid, a pitcher, a spatula and cutlery. There is **one instance per type**, so it is good for real-world evaluation (physical sets exist) but not for intra-category variation. It has no salt shaker, cutting board or tray.

### Cited Findings
- 77 objects in 5 categories: "Food items, kitchen items, tool items, shape items, task items." — [YCB object set page (snippet)](https://www.ycbbenchmarks.com/object-set/)
- Kitchen items: "pitcher, bleach cleanser, glass cleaner; … plastic wine glass, enamel-coated metal bowl, metal mug, abrasive sponge; … cooking skillet with glass lid, metal plate, eating utensils (knife, spoon, fork), spatula, white table cloth." — [YCB object set page (snippet)](https://www.ycbbenchmarks.com/object-set/)
- "Some of the physical YCB objects have been replaced due to lack of availability of the original objects"; there is a purchase-links page for sourcing the items. — [YCB object set page](https://www.ycbbenchmarks.com/object-set/); [Object Set Purchase Links](http://www.ycbbenchmarks.com/object-set-purchase-links/)
- The set includes "objects that are commonly used in standard assessment procedures, such as a mug, a bowl, a pitcher …"; the database provides high-resolution RGB-D scans, physical properties and geometric models. — [Calli et al. 2015, RAM version](https://people.eecs.berkeley.edu/~pabbeel/papers/2015-RAM-YCB-object-set.pdf); [arXiv 1502.03143](https://arxiv.org/pdf/1502.03143); [models S3 site](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/)
- 2015 paper versions differ in captions (one says "coffee mug and plate", another "metal mug"). — same sources

### Inferences
- YCB models are real scans with textures in metric units. This is from common knowledge of the dataset (they are used directly in BOP/YCB-Video); the YCB pages did not state it explicitly. They can serve as metric "anchors" for sanity-checking Objaverse rescaling.
- Licence of the YCB models: commonly cited as CC BY 4.0, but **unverified here**.

### Gaps
- Exact physical dimensions of the YCB mug, skillet and plate (the YCB site was not fetchable).

---

## Q5. Google Scanned Objects (GSO), HANDAL, OmniObject3D, ABO, RoboCasa, Toys4K, NVIDIA kitchen assets

### Takeaway
Scanned datasets (GSO, OmniObject3D, HANDAL) give metric, photoreal textures but few kitchenware instances per category. HANDAL is the only one with **handle affordance segmentation**, but it is CC-BY-NC-SA and covers handled tools, not trays or boards. RoboCasa (2,509 assets in 153 kitchen-oriented categories) is the most kitchen-specific simulation library. ABO is furniture- and product-heavy.

### Cited Findings
**Google Scanned Objects (Downs et al., ICRA 2022)**
- "A collection of more than one thousand 3D-scanned household items," "preprocessed for use in Ignition Gazebo and the Bullet simulation platforms, but … easily adaptable to other simulators." — [Google Research blog](https://www.research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/); [arXiv 2204.11918](https://arxiv.org/abs/2204.11918v1)
- The count of 1,030 objects comes from a TIB dataset record. — [TIB record](https://service.tib.eu/ldmservice/dataset/google-scanned-objects-dataset)
- Licence: the paper says "Creative Commons." A MuJoCo-format HF mirror states the "3D assets (obj and png) are licensed under the CC-BY 4.0 License." This comes from a mirror, so check it against the source. — search snippets of [arXiv 2204.11918](https://www.arxiv.org/pdf/2204.11918)
- Kitchen content (from memory, **unverified**): GSO is dominated by toys, shoes and boxed consumer products, with some mugs, bowls and plates. Per-category counts are not published because the dataset has no category taxonomy.

**HANDAL (Guo et al., IROS 2023, NVIDIA)**
- 308K annotated frames from 2.2K videos of 212 real-world objects in 17 categories (the NVIDIA page says 306K, 210 objects). It is "robotics-ready," with "6-DoF category-level pose and scale, handle affordance segmentation, and 3D reconstructed meshes," and focuses on hardware and kitchen tools. — [arXiv 2308.01477](https://arxiv.org/pdf/2308.01477); [NVIDIA publication](https://research.nvidia.com/publication/2023-10_handal-dataset-real-world-manipulable-object-categories-pose-annotations)
- Licence: dataset **CC-BY-NC-SA-4.0**; code under the NVIDIA Source Code License. Download from Google Drive. The pipeline notes "The initial annotations lack ground truth scale because we use RGB-only inputs." Scale is entered from measured millimetre dimensions into `canonical_scale.json`. The default mesh output is vertex-coloured (`colored.ply`). — [NVlabs/HANDAL](https://github.com/NVlabs/HANDAL)
- Category list (from memory, **unverified**; the README did not list them): includes mugs, pots/pans, measuring cups, ladles, spatulas, whisks, strainers, utensils, hammers, screwdrivers, pliers (several kinds), wrenches, ratchets and power drills.

**OmniObject3D (Wu et al., CVPR 2023)**
- "6,000 scanned objects in 190 daily categories," sharing classes with ImageNet and LVIS. Each comes with textured meshes, point clouds, multiview renders and real captured videos. — [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_OmniObject3D_Large-Vocabulary_3D_Object_Dataset_for_Realistic_Perception_Reconstruction_and_CVPR_2023_paper.pdf); [PapersWithCode](https://PapersWithCode.com/dataset/omniobject3d)
- "The OmniObject3D dataset is released under the CC BY 4.0." Download via the OpenXLab CLI (about 1.2 TB compressed) or Google Drive. The README does not state that meshes are metric; the only scale mention is "poses_bounds_rescaled.npy # rescaled to world-scale" for video poses. — [omniobject3d/OmniObject3D GitHub](https://github.com/omniobject3d/OmniObject3D)
- The category list was not retrievable. The requester's prior curation found only 4 pans and no tray, cutting board or salt shaker categories (requester-supplied).

**ABO (Amazon Berkeley Objects, Collins et al. CVPR 2022)**
- 147,702 product listings, 398,212 catalog images, and "7,953 products with corresponding artist-designed 3D meshes," distributed as glTF 2.0. — [ABO paper (ar5iv)](https://ar5iv.arxiv.org/html/2110.06199); [AWS Open Data listing](https://aws.amazon.com/marketplace/pp/prodview-ygocdgcstr7v2)
- Licence: CC BY 4.0 according to a third-party add-on seller. **Not verified from Amazon's page**, and the brief's suggested CC BY-NC is unconfirmed. — [Superhive ABO Connect FAQ (secondary)](https://superhivemarket.com/products/abo-connect-access-8000-free-high-quality-3d-models/faq)
- A secondary source says 3D models cover 63 categories. ABO meshes are mostly furniture and home goods (**unverified** for kitchenware counts). — [alphaXiv benchmark page (secondary)](https://www.alphaxiv.org/benchmarks/uc-berkeley/amazon-berkeley-objects)

**RoboCasa (Nasiriany et al., RSS 2024)**
- "2,509 high-quality assets spanning 153 unique object categories" for kitchen tasks ("fruits and vegetables, dairy, poultry, drinks, receptacles, tools, and more"). Most of these (1,592) were generated with Luma.ai text-to-3D, and the rest come from Objaverse. The later RoboCasa365 release reportedly has over 3,200 assets (secondary source). — [RoboCasa arXiv 2406.02523 (snippet)](https://arxiv.org/pdf/2406.02523); [emergentmind summary](https://www.emergentmind.com/topics/robocasa)
- Licence of the RoboCasa asset library and per-category counts (pans, trays, boards) were **not verified**. AI-generated (Luma) meshes may carry generator terms that are separate from the code licence.

**NVIDIA PhysicalAI kitchen assets**
- A search hit for `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` lists kitchen items such as kettles, pots, pitchers and measuring cups. Related NVIDIA PhysicalAI manipulation datasets use non-commercial or changing licences. Contents and licence of this specific asset set are **unverified**. — [HF card (snippet)](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/blob/main/README.md); [related Objects dataset card](https://www.huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects/blob/main/README.md)

**Toys4K**
- Not researched successfully. From memory (**unverified**): about 4,000 objects in 105 toy-like categories, so it is unlikely to help with kitchenware.

### Inferences
- No single dataset covers all five UTAMP-2 objects with many instances. A realistic composite:
  1. **Objaverse-LVIS (CC-BY/CC0, quality-filtered)** for breadth across all 15+ categories, rescaled to metric.
  2. **GSO / OmniObject3D / YCB scans** for photoreal, metric mugs, bowls, plates and pans as a higher-fidelity subset.
  3. **HANDAL** only for internal handle-affordance validation, not redistribution, because of NC-SA.
- Cutting boards, trays, platters and salt shakers are geometrically simple. Procedural generation (parametric boxes or rounded slabs with wood PBR textures, cylinders with caps) is a credible, licence-free supplement. This is a design suggestion, not a sourced finding.

### Gaps
- OmniObject3D 190-category list; GSO per-category kitchen counts; RoboCasa per-category counts and licence; HANDAL category list from a primary source; ABO licence from Amazon.

---

## Q6. Part annotations: PartNet, PartNet-Mobility, ShapeNet

### Takeaway
PartNet has kitchen-relevant categories **Mug (212 shapes), Bowl (208) and Knife (384)** with fine-grained hierarchical parts, so mug handle/body labels are available. These are ShapeNet CAD meshes under ShapeNet's non-commercial terms. PartNet-Mobility (SAPIEN) has 2,346 articulated models in 46 categories, including a pot category (25 models in the paper table). It has no frying pans, trays or cutting boards.

### Cited Findings
- PartNet: "573,585 part instances over 26,671 3D models covering 24 object categories." Categories include bowl, knife, mug, bottle, chair, faucet and scissors, but no kettle. The paper table gives shape counts of bowl 208, knife 384 and mug 212; these were read from a snippet of the table, so double-check them. — [PartNet arXiv 1812.02713](https://arxiv.org/pdf/1812.02713); [ar5iv](https://ar5iv.arxiv.org/html/1812.02713)
- PartNet-Mobility: "over 14K articulated parts over 2,346 object models from 46 common indoor object categories." The SAPIEN paper's table includes a "Pot" row with 25 models (from a garbled snippet). — [SAPIEN arXiv 2003.08515](https://arxiv.org/pdf/2003.08515)
- ShapeNetCore: 55 categories, about 51,300 unique models (51,162 in the SHREC'16 release). — [shapenet.org](https://shapenet.org/); [SHREC16](https://people.cs.umass.edu/~elm/papers/shrec16shapenet.pdf)

### Inferences
- PartNet's mug hierarchy (handle, body, containing part) gives direct "graspable handle" vs "support/contain body" labels for mugs. It could be used to train or validate a part-segmentation model before transferring to Objaverse mugs and pans.
- ShapeNetCore's mug and bowl categories are commonly cited as about 214 mugs and 186 bowls, but **these counts are unverified**: no source confirmed them.
- PartNet and ShapeNet terms are generally non-commercial research use (**licence text not retrieved here**). Renders derived from them may be publishable for research, but redistributing the meshes is not allowed.

### Gaps
- Exact PartNet mug part names (for example `handle`, `body`, `containing_things`) were not confirmed from primary docs.
- The full PartNet-Mobility category list, including whether it is spelled "KitchenPot", was not confirmed.
- ShapeNet licence wording.

---

## Q7. CoppeliaSim vision sensors: RGB, depth, per-object segmentation, API, renderers, export to Blender

### Takeaway
CoppeliaSim can output RGB (`sim.getVisionSensorImg`), depth (`sim.getVisionSensorDepth`, normalised 0–1 between the near and far planes, or metric with an option bit) and **per-object instance masks** via the "color coded handles" render mode, where each pixel encodes an object handle. All three are callable over the ZMQ remote API. Scenes can be exported as glTF 2.0, with animation, for rendering in Blender, and selected shapes can be exported as meshes via the Assimp plugin.

### Cited Findings
- `sim.getVisionSensorImg` retrieves the pixel buffer; an options bitfield selects greyscale or RGB and an optional alpha. The data "doesn't make sense if sim.handleVisionSensor wasn't called previously." The older `getVisionSensorImage` and `getVisionSensorCharImage` are deprecated. — [manual: simGetVisionSensorImg](https://manual.coppeliarobotics.com/en/regularApi/simGetVisionSensorImg.htm) (search snippet); [simGetVisionSensorCharImage](https://manual.coppeliarobotics.com/en/sim/simGetVisionSensorCharImage.htm)
- `sim.getVisionSensorDepth`: if options "bit0 set (1): the returned values indicate distances in meters, otherwise return values are in the range of 0.0 - 1.0" (0 = near plane, 1 = far plane). It supports a sub-region (`pos`, `size`). In Python/Lua the return is a packed buffer: "Use sim.unpackFloatTable." `sim.getVisionSensorDepthBuffer` is deprecated. — [manual: simGetVisionSensorDepth](https://manual.coppeliarobotics.com/en/regularApi/simGetVisionSensorDepth.htm) (snippet)
- Near and far clipping planes are set via the vision sensor float parameters (`floatParams[i+0]` near, `floatParams[i+1]` far) at creation, or in the properties dialog. — [manual: simCreateVisionSensor](https://manual.coppeliarobotics.com/en/sim/simCreateVisionSensor.htm) (snippet)
- Render modes in the vision sensor dialog: "Legacy OpenGL" (the default, visible colour channels), and "Legacy OpenGL, color coded handles," which renders objects by coding their handles into the colours, with the detected handles returned in the first auxiliary data packet. A POV-Ray option, via a plugin, "allows for shadows (also soft shadows)." — [manual: Vision sensor dialog](https://manual.coppeliarobotics.com/en/visionSensorPropertiesDialog.htm) (snippet)
- The forum recommends the color-coded handles mode for per-object masks: each object is drawn in a unique colour corresponding to its handle, and occlusions are handled properly, unlike toggling visibility layers. — [CoppeliaSim forum p=40673](https://forum.coppeliarobotics.com/viewtopic.php?p=40673)
- The OpenGL3 external renderer plugin (`simOpenGL3`) "uses OpenGL 3+ and supports shadows." Light parameters "will have visibly different effects than the default renderer," and shadows increase rendering time. — [CoppeliaRobotics/simOpenGL3](https://github.com/CoppeliaRobotics/simOpenGL3)
- Export: "[Modules > Exporters > GLTF exporter...]" and "[Modules > Exporters > GLTF animation exporter...]" export a model, scene or whole simulation to glTF 2.0. Mesh import/export goes through the Assimp plugin ("[File > Export > Selected shapes...]"). CoppeliaSim's blog tutorial (Blender 4.0) records a glTF animation and imports it into Blender via File > Import > glTF 2.0, and advises running in real-time because "simulated time will be the only source of time information." — [manual: import/export](https://manual.coppeliarobotics.com/en/importExport.htm); [CoppeliaSim blog: Realistic video renders using Blender](https://coppeliarobotics.com/blog/article/realistic-video-renderings-using-blender) (snippets)
- An older community tool (V-REP / Blender 2.8 era) exported visible meshes as OBJ plus pose keyframes. It is likely outdated. — [BorisBogaerts/CoppeliaSim-Blender-Tools](https://github.com/BorisBogaerts/CoppeliaSim-Blender-Tools)

### Inferences
- A practical label-generation recipe in CoppeliaSim is two co-located vision sensors with identical intrinsics: one in normal mode (RGB plus depth) and one in "color coded handles" mode, decoding pixel colour to handle to object or part. Parts such as a handle could be made separate child shapes so they receive their own handle and become part masks for free. Pixel decoding should use nearest-neighbour rendering; anti-aliasing would corrupt the codes.
- Read depth with bit0 = 1 for metric depth. Otherwise linearise the 0–1 values using near and far.
- From Python, the ZMQ remote API exposes the same `sim.*` functions (`client.require('sim')`), so `sim.getVisionSensorImg` and `sim.getVisionSensorDepth` return bytes plus resolution, which numpy can reshape. Note the vertical flip: CoppeliaSim images are bottom-up. These remote-API usage details come from general knowledge of the API and were **not verified** in this session.
- The CoppeliaSim OpenGL renderer has limited material realism (no PBR, weak transparency and refraction). The "glass mug" and wood textures will look synthetic. Exporting to Blender (glTF) and rendering with Cycles via BlenderProc is the route to photorealism, but the glTF material conversion fidelity is **unverified**.

### Gaps
- Primary-manual confirmation of the exact current render-mode names in CoppeliaSim 4.6–4.9, and whether OpenGL3 supports color-coded handles. The manual's domain could not be fetched.
- Whether textured imported meshes such as Objaverse GLBs keep their textures when imported via Assimp and exported again via glTF.

---

## Q8. BlenderProc and other synthetic-data renderers (Kubric, Isaac Sim Replicator)

### Takeaway
BlenderProc2 (DLR, **GPL-3.0**) is the closest fit for photoreal, physically placed kitchen scenes. It produces RGB, depth, normals and segmentation, and writes COCO, BOP and HDF5. Its licence covers the tool, not the rendered images. Kubric (Apache-2.0) and Isaac Sim Replicator are the main alternatives.

### Cited Findings
- BlenderProc features: loaders for "`*.obj`, `*.ply`, `*.blend`, `*.fbx`, BOP, ShapeNet, Haven, 3D-FRONT, etc."; setting or sampling object poses with "simple simulated physical interactions between objects" and collision checking; "Set or sample physically-based materials and textures"; light sampling; camera sampling with constraints. It renders "RGB, stereo, depth, normal and segmentation images/sequences" and writes ".hdf5 containers, COCO & BOP annotations." It generated the BOP Challenge 2020 data. Licence **GPL-3.0**. Citation: Denninger et al., "BlenderProc2: A Procedural Pipeline for Photorealistic Rendering," JOSS 8(82), 2023, doi:10.21105/joss.04901. — [DLR-RM/BlenderProc](https://github.com/DLR-RM/BlenderProc)
- Kubric (Greff et al., CVPR 2022): an "open-source Python framework that interfaces with PyBullet and Blender to generate photo-realistic scenes, with rich annotations," producing "instance segmentation masks, depth maps, and optical flow." It uses Blender 2.93, and the repo is **Apache-2.0**. — [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.html); [google-research/kubric](https://github.com/google-research/kubric)
- Isaac Sim Replicator annotators: `semantic_segmentation` (uint32 H×W×1), `instance_segmentation` (only semantically labelled entities), `distance_to_camera` (float32 depth from the camera centre) vs `distance_to_image_plane`. BasicWriter flags these outputs, with colourised or raw uint32 PNG options. — [Isaac Sim 6.0 camera docs](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.sensors.experimental.camera/docs/index.html); [Replicator API docs](https://docs.omniverse.nvidia.com/py/replicator/1.4.7/source/extensions/omni.replicator.core/docs/API.html)

### Inferences
- BlenderProc does not list an Objaverse loader, but Objaverse GLBs can be imported through Blender's glTF importer inside a BlenderProc script, for example with `bpy.ops.import_scene.gltf` and then wrapping the objects as `MeshObject`. This is plausible but **not verified as a documented BlenderProc path**.
- GPL-3.0 governs the BlenderProc code. Images rendered with it are not GPL-encumbered (general GPL understanding, not a sourced legal opinion). What governs the derived-dataset licence is the asset licences (CC-BY attribution).
- To stay consistent with the CoppeliaSim/Franka scene, there are two options. One is to export the CoppeliaSim scene to glTF and re-render it in BlenderProc with identical camera intrinsics and extrinsics. The other is to rebuild the tabletop in BlenderProc and keep CoppeliaSim only for planning and physics. The first keeps poses identical; the second gives more control over materials.

### Gaps
- Isaac Sim licensing and hardware requirements (RTX GPU) were not researched.
- BlenderProc's handling of glTF/PBR transparency (glass) was not checked.
