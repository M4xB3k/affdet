# Simulation validity: is a simulated grasp label worth anything?

Written 2026-11-16 from the extracted texts of the papers below (second reader not yet; each claim names its source).
Purpose: answer the supervisors' question "why trust MuJoCo labels?" without overclaiming.

## Sources read
- Eppner, Mousavian, Fox (2019), "A Billion Ways to Grasp", arXiv 1912.05604 (venue not in the text).
- Eppner, Mousavian, Fox (2021), ACRONYM, arXiv 2011.09584 (see `eppner_2021_acronym.md`, second-read).
- Erez, Tassa, Todorov (2015), "Simulation tools for model-based robotics: comparison of Bullet, Havok, MuJoCo, ODE and PhysX", ICRA 2015.
- Todorov, Erez, Tassa (2012), "MuJoCo: a physics engine for model-based control", IROS 2012.
- Collins, McVicar, Wedlock, Brown, Howard, Leitner (2019), "Benchmarking simulated robotic manipulation through a real world dataset", RA-L (preprint accepted Nov 2019; arXiv 1911.01557).
- Collins, Chand, Vanderkop, Howard (2021), "A review of physics simulators for robotic applications", IEEE Access 9.

## What each says that matters for us
1. **A dense, exhaustive grid of simulated grasps is an accepted reference.** Eppner 2019 simulated every parallel-jaw grasp on a
   grid in SE(3) (5 mm, 7.5 deg) for 21 YCB objects in FleX: about 317 billion candidates, about 1 billion passed a collision and
   "object between the fingers" test and were simulated, 15.57 % succeeded. 100 GPUs for one and a half months. Our grid is the
   same idea in a smaller space (top-down only: x, y, 6 yaws), and our "blocked" and "off object" cells are their two filters.
2. **Their physics is simpler than ours.** Eppner 2019 and ACRONYM simulate "in free space, without any gravity", close with a
   force-based controller, then shake. Friction 1.0 between pads and all objects, uniform density. ACRONYM reports that shaking
   without gravity agrees with gravity labels at 91 % precision and 93 % recall. We lift 50 cm under gravity and measure slip and
   tilt, which is closer to a real pick; our pad friction is also 1.0.
3. **Real-robot checks exist, but they are small.** Eppner 2019: 25 grasps from the simulated set on 5 objects with a real Franka
   Panda; 3 failed, attributed to uniform density, rigid objects and friction differences. ACRONYM cites 86 % average precision
   for FleX against 2,625 real grasps (Danielczuk et al.; that paper not read here).
4. **Simulation beats analytic grasp metrics.** Eppner 2019 states that classical metrics "are very sensitive w.r.t. contact
   point locations and do not capture stability", and that simulation covers the whole grasp process including object
   dynamics. This matches our own result that the analytic rule is a strong but different baseline.
5. **Why MuJoCo.** Erez 2015 compared five engines on robotics models, including a grasp test (a hand grasping a capsule while
   the arm swings); its conclusion: each engine does best on what it was built for, and "MuJoCo wins the robotics-related
   tests". The Collins 2021 review notes MuJoCo's contact stability as a reason for its popularity. Neither paper tests our
   gripper or objects.
6. **Every simulator has a reality gap; nobody has measured MuJoCo for our setup.** Collins 2019 replayed a real recorded
   manipulation dataset in PyBullet and V-Rep (not MuJoCo) and found clear differences in joint torque, force readings and object
   pose. They present the numbers to establish the metrics, "not to rank the performance of the simulators".

## What we can and cannot claim
- Can: our labels follow the accepted recipe for simulation-labelled grasps (dense candidates, a geometric pre-filter, a
  physics outcome, fixed friction and uniform density), with a stricter test than ACRONYM's (gravity, lift, slip, tilt).
- Can: we checked our own simulator for artefacts: recording the raw quantities does not change any outcome (bit-identical);
  the clamp failures on the mug are real loss of contact, not contact flicker (both pads at 0 N for the whole 0.1 s window).
- Cannot: that MuJoCo labels match a real Panda. We have no real-robot check. The supervisors accepted simulation as ground
  truth for this thesis (meeting 2026-11); say so plainly.
- Known weakness of our own setup (found 2026-11-16): with the old 0.1 s wait the pads still press with about 300 N instead of
  30 N when the lift starts. With 0.3 s they settle to about 29-31 N on the bar and the mug; a bowl wall needs about 0.5 s. Outcomes on the test bar barely changed; on 5 real objects
  see `research_notes/grasp_stages_2026-11-16.md`.

## Likely supervisor questions
1. Why should we trust MuJoCo grasp labels? Answer with points 1, 2, 5 and the "cannot" line: same recipe as ACRONYM and the
   Billion-grasp set, a stricter outcome test, an engine that did best on robotics tests, and no real-robot validation of our own.
2. Why not gravity-free shaking like ACRONYM? Because our label must say whether the object can be lifted and stays level; a
   shake in free space cannot test tilt. ACRONYM's own 91 / 93 % check is only about retention.
3. What would a real check cost? Eppner 2019 used 25 grasps on 5 objects. A similar small set on the lab robot (best and worst
   predicted cells of 5 objects) would be the cheapest credible check.

## Not verified
- Venue of Eppner 2019 (not in the extracted text). The Erez 2015 author order and the ICRA venue of Erez 2015 and IROS of
  Todorov 2012 are from the texts' mentions, not from a title page.
- Collins et al. 2018 "Quantifying the reality gap" is cited by Eppner 2019 but was not read.
- Erez 2015 figure values (speed-accuracy curves) are garbled in the text extraction and are not quoted.
