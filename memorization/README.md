# Memorisation check

`attack_results.json` is the hack-resistance card computed on an earlier internal environment with the same game
and accounting as `hedge_gym/` (2,000 practice + 500 held-out synthetic worlds; that environment is not part of
this release). `attacks.py` is the script that produced it; it imports that environment's package and is included
for the record. Key rows (score, higher is better): honest BS-delta −0.0076 on the practice worlds / −0.0078 on
held-out worlds (no gap); a lookup cheat that memorises the practice bank −0.0004 on it (the oracle ceiling) /
−0.0184 on held-out worlds (53× worse). Memorising synthetic practice paths does not transfer to fresh worlds, let
alone to real markets.
