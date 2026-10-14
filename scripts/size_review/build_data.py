import json, sys, os, base64
d = json.load(open(sys.argv[1])); th = sys.argv[2]
def img(uid):
    p = os.path.join(th, uid + ".jpg")
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else None
slim = lambda r: dict(uid=r["uid"], h=r["h"], d=r["d"], m=r["m"], cat=r["cat"], mat=r["mat"].replace("|air", "").replace("air|", ""),
                      q=r["q"], lic=r["lic"], desc=r["desc"], why=r["why"], width=r.get("width"), kind=r.get("kind"), img=img(r["uid"]))
out = {k: dict(range=v["range"], n_usable=v["n_usable"], n_kept=v["n_kept"], n_ok=v["n_ok"], median=v["median"],
               values=v["values"], typical=[slim(r) for r in v["typical"]], outliers=[slim(r) for r in v["outliers"]])
       for k, v in d.items()}
open(sys.argv[3], "w").write(json.dumps(out, separators=(",", ":")))
