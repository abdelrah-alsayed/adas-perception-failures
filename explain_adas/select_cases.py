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


def pick_matched_hits(misses, rows, seed=42):
    """For each miss, pick one hit: same class, same occlusion, similar height."""
    rng = random.Random(seed)
    hits = sorted((r for r in rows if r["kind"] == "hit"),
                  key=lambda r: (r["id"], r["line"]))
    used = {m["id"] for m in misses}
    pairs = []
    for miss in sorted(misses, key=lambda r: (r["id"], r["line"])):
        free = [h for h in hits if h["id"] not in used and h["type"] == miss["type"]]
        same_occ = [h for h in free if h["occluded"] == miss["occluded"]]
        hit, how = None, None
        for tol, name in ((0.25, "height 25%"), (0.40, "height 40%")):
            close = [h for h in same_occ if abs(h["height"] / miss["height"] - 1) <= tol]
            if close and hit is None:
                hit, how = rng.choice(close), name
        for pool, name in ((same_occ, "closest"), (free, "closest any occlusion")):
            if pool and hit is None:
                hit = min(pool, key=lambda h: abs(h["height"] - miss["height"]))
                how = name
        if hit is not None:
            used.add(hit["id"])
            pairs.append({"miss": miss, "hit": hit, "match": how})
    return pairs
