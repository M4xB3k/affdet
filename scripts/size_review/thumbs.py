import json, sys, os, io, urllib.request, concurrent.futures as cf
from PIL import Image
d = json.load(open(sys.argv[1])); outdir = sys.argv[2]
uids = sorted({r["uid"] for v in d.values() for r in v["outliers"] + v["typical"]})
def get(uid):
    p = os.path.join(outdir, uid + ".jpg")
    if os.path.exists(p): return uid, "cached"
    try:
        meta = json.load(urllib.request.urlopen(f"https://api.sketchfab.com/v3/models/{uid}", timeout=20))
        ims = meta.get("thumbnails", {}).get("images", [])
        if not ims: return uid, "no thumb"
        im = min(ims, key=lambda i: abs(i["width"] - 256))
        img = Image.open(io.BytesIO(urllib.request.urlopen(im["url"], timeout=20).read())).convert("RGB")
        img.thumbnail((176, 176)); img.save(p, "JPEG", quality=72)
        return uid, "ok"
    except Exception as e:
        return uid, f"err {type(e).__name__} {getattr(e,'code','')}"
from collections import Counter
c = Counter()
with cf.ThreadPoolExecutor(8) as ex:
    for uid, s in ex.map(get, uids): c[s.split(' ')[0] if s.startswith('err') else s] += 1; 
print(len(uids), c)
