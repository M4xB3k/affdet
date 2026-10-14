# Size review page

Builds the "Kitchen Object Sizes" page (https://claude.ai/artifact/TJXEXFtwZazn2sEKfAmhFJ)
from `docs/objaverse/objathor_kitchen.csv`. Run from a scratch directory:

    python scripts/size_review/outliers.py docs/objaverse/objathor_kitchen.csv outliers.json
    python scripts/size_review/classify.py outliers.json classified.json
    python scripts/size_review/thumbs.py classified.json thumbs/      # Sketchfab thumbnails
    python scripts/size_review/build_data.py classified.json thumbs/ data.json
    # then replace __DATA__ in template.html with data.json (escape "</" as "<\/")

Ranges per role live in `outliers.py` (R), wrong-object words in `classify.py`
(WRONG). The salt-shaker tab's extra sections (curated set, the 122 name
mismatches) were added to data.json by hand from `docs/objaverse/salt_shaker_set.csv`.
