"""Where does one stack_test spend its time? Usage: python scripts/sim_profile_one.py PICKS GLB SHAKER_CSV CACHE TOP BASE [FACE]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cProfile, pstats
from affdet.sim.mujoco_support import stack_test
from affdet.sim.registry import Registry

picks, glb, csv_, cache, top_s, base_s = sys.argv[1:7]
face = sys.argv[7] if len(sys.argv) > 7 else "under"
reg = Registry(picks, glb, csv_, cache)
top, _ = reg.get(top_s); base, bm = reg.get(base_s)
off = (0.0, 0.0)
stack_test(base, top, "under", face, offset=off)  # warm up
t = time.time()
for _ in range(5):
    stack_test(base, top, "under", face, offset=off)
print("mean s per stack_test:", (time.time() - t) / 5)
pr = cProfile.Profile(); pr.enable(); stack_test(base, top, "under", face, offset=off); pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
