import random

def split_ids(ids, strata, val_fraction, seed):
    """Split picture ids into (train, val). Each group in strata is split on its own."""
    rng = random.Random(seed)
    groups = {}
    for pid in sorted(ids):
        groups.setdefault(strata[pid], []).append(pid)
    train, val = [], []
    for name in sorted(groups):
        members = groups[name]
        rng.shuffle(members)
        n_val = round(len(members) * val_fraction)
        val.extend(members[:n_val])
        train.extend(members[n_val:])
    return sorted(train), sorted(val)