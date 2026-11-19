import random


def pick_misses(rows, per_class=5, seed=42):
    """Pick failure cases at random, one per picture, per_class of each class."""
    rng = random.Random(seed)
    pools = {"Pedestrian": [], "Cyclist": []}
    for row in sorted(rows, key=lambda r: (r["id"], r["line"])):
        if row["kind"] != "hit" and row["type"] in pools:
            pools[row["type"]].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    chosen, used = [], set()
    for pool in pools.values():
        taken = 0
        for row in pool:
            if taken < per_class and row["id"] not in used:
                chosen.append(row)
                used.add(row["id"])
                taken += 1
    for pool in pools.values():
        for row in pool:
            if len(chosen) < 2 * per_class and row["id"] not in used:
                chosen.append(row)
                used.add(row["id"])
    return chosen
