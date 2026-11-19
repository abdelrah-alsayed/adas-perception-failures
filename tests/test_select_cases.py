from explain_adas.select_cases import pick_misses


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
