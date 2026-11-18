from explain_adas.splitting import split_ids


def make_data():
    ids = [f"{i:06d}" for i in range(100)]
    strata = {pid: ("rare" if int(pid) % 4 == 0 else "common") for pid in ids}
    return ids, strata


def test_split_keeps_everything_and_has_no_overlap():
    ids, strata = make_data()
    train, val = split_ids(ids, strata, 0.2, seed=42)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == sorted(ids)


def test_same_seed_gives_the_same_split():
    ids, strata = make_data()
    assert split_ids(ids, strata, 0.2, 42) == split_ids(ids, strata, 0.2, 42)


def test_different_seed_gives_a_different_split():
    ids, strata = make_data()
    assert split_ids(ids, strata, 0.2, 42) != split_ids(ids, strata, 0.2, 7)


def test_each_group_gets_its_share():
    ids, strata = make_data()
    train, val = split_ids(ids, strata, 0.2, seed=42)
    rare_in_val = [p for p in val if strata[p] == "rare"]
    assert len(rare_in_val) == 5
    assert len(val) == 20