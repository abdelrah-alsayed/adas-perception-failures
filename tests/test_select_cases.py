from explain_adas.select_cases import pick_matched_hits, pick_misses


def rows_for(type_, n, kind="not_found", start=0):
    return [{"id": f"{start + i:06d}", "line": 0, "type": type_, "kind": kind}
            for i in range(n)]


def test_picks_five_of_each_class():
    rows = rows_for("Pedestrian", 20) + rows_for("Cyclist", 20, start=100)
    types = [r["type"] for r in pick_misses(rows)]
    assert types.count("Pedestrian") == 5
    assert types.count("Cyclist") == 5


def test_hits_are_never_picked():
    rows = rows_for("Pedestrian", 20, kind="hit") + rows_for("Cyclist", 20, start=100)
    assert all(r["kind"] != "hit" for r in pick_misses(rows))


def test_one_case_per_picture():
    rows = [{"id": "000001", "line": i, "type": "Pedestrian", "kind": "low_score"}
            for i in range(10)]
    rows += rows_for("Cyclist", 10, start=100)
    chosen = pick_misses(rows)
    assert len({r["id"] for r in chosen}) == len(chosen)


def test_same_seed_same_cases_other_seed_other_cases():
    rows = rows_for("Pedestrian", 50) + rows_for("Cyclist", 50, start=100)
    assert pick_misses(rows, seed=42) == pick_misses(rows, seed=42)
    assert pick_misses(rows, seed=42) != pick_misses(rows, seed=7)


def test_fallback_fills_up_from_the_other_class():
    rows = rows_for("Pedestrian", 20) + rows_for("Cyclist", 2, start=100)
    chosen = pick_misses(rows)
    assert len(chosen) == 10
    assert [r["type"] for r in chosen].count("Cyclist") == 2


def test_input_order_does_not_change_the_result():
    rows = rows_for("Pedestrian", 30) + rows_for("Cyclist", 30, start=100)
    assert pick_misses(rows) == pick_misses(list(reversed(rows)))


def obj(pid, type_, kind, height, occluded=0):
    return {"id": pid, "line": 0, "type": type_, "kind": kind,
            "height": height, "occluded": occluded}


def test_matches_same_class_occlusion_and_similar_height():
    miss = obj("000001", "Pedestrian", "not_found", 100)
    rows = [obj("000002", "Pedestrian", "hit", 110),
            obj("000003", "Pedestrian", "hit", 300),
            obj("000004", "Cyclist", "hit", 100),
            obj("000005", "Pedestrian", "hit", 100, occluded=2)]
    pairs = pick_matched_hits([miss], rows)
    assert pairs[0]["hit"]["id"] == "000002"
    assert pairs[0]["match"] == "height 25%"


def test_hit_comes_from_another_picture():
    miss = obj("000001", "Pedestrian", "not_found", 100)
    rows = [obj("000001", "Pedestrian", "hit", 100), obj("000002", "Pedestrian", "hit", 100)]
    assert pick_matched_hits([miss], rows)[0]["hit"]["id"] == "000002"


def test_a_hit_is_used_only_once():
    misses = [obj("000001", "Pedestrian", "not_found", 100),
              obj("000002", "Pedestrian", "not_found", 100)]
    rows = [obj("000010", "Pedestrian", "hit", 100), obj("000011", "Pedestrian", "hit", 100)]
    ids = [p["hit"]["id"] for p in pick_matched_hits(misses, rows)]
    assert sorted(ids) == ["000010", "000011"]


def test_fallback_widens_the_height_then_takes_the_closest():
    miss = obj("000001", "Pedestrian", "not_found", 100)
    wide = [obj("000002", "Pedestrian", "hit", 135)]
    assert pick_matched_hits([miss], wide)[0]["match"] == "height 40%"
    far = [obj("000003", "Pedestrian", "hit", 400), obj("000004", "Pedestrian", "hit", 250)]
    pair = pick_matched_hits([miss], far)[0]
    assert pair["hit"]["id"] == "000004" and pair["match"] == "closest"


def test_any_occlusion_is_the_last_resort():
    miss = obj("000001", "Pedestrian", "not_found", 100, occluded=2)
    rows = [obj("000002", "Pedestrian", "hit", 100, occluded=0)]
    assert pick_matched_hits([miss], rows)[0]["match"] == "closest any occlusion"


def test_same_seed_gives_the_same_pairs():
    miss = obj("000001", "Pedestrian", "not_found", 100)
    rows = [obj(f"{i:06d}", "Pedestrian", "hit", 100) for i in range(10, 40)]
    assert pick_matched_hits([miss], rows, seed=42) == pick_matched_hits([miss], rows, seed=42)
