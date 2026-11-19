from explain_adas.case_table import object_row
from explain_adas.make_cases import case_rows, read_table

OBJ = {"type": "Pedestrian", "box": [100, 100, 140, 200], "occluded": 1, "truncated": 0.1}


def det(name, score, box=(100, 100, 140, 200)):
    return {"name": name, "score": score, "box": list(box)}


def test_row_of_a_hit_has_kind_and_target_score():
    row = object_row("000007", 2, OBJ, [det("Pedestrian", 0.8)])
    assert row["kind"] == "hit"
    assert row["target_score"] == 0.8
    assert row["height"] == 100
    assert row["eligible"] is True
    assert (row["id"], row["line"], row["occluded"]) == ("000007", 2, 1)


def test_target_score_is_zero_when_nothing_of_the_class_overlaps():
    row = object_row("000007", 2, OBJ, [det("Cyclist", 0.9)])
    assert row["kind"] == "wrong_class"
    assert row["target_score"] == 0.0


def test_target_score_of_a_low_score_miss_is_kept():
    row = object_row("000007", 2, OBJ, [det("Pedestrian", 0.12)])
    assert row["kind"] == "low_score"
    assert row["target_score"] == 0.12


def test_read_table_turns_text_into_numbers(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "id,line,type,height,occluded,truncated,kind,target_score,eligible\n"
        "000007,2,Pedestrian,100.0,1,0.1,hit,0.8,True\n"
    )
    row = read_table(path)[0]
    assert row["line"] == 2 and row["occluded"] == 1
    assert row["height"] == 100.0 and row["target_score"] == 0.8
    assert row["eligible"] is True


def test_case_rows_put_each_hit_next_to_its_miss():
    miss = {"id": "000001", "line": 0, "type": "Pedestrian", "kind": "not_found"}
    hit = {"id": "000002", "line": 1, "type": "Pedestrian", "kind": "hit"}
    pairs = [{"miss": miss, "hit": hit, "match": "height 25%"}]
    rows = case_rows([miss], pairs)
    assert [r["role"] for r in rows] == ["miss", "hit"]
    assert [r["pair"] for r in rows] == [1, 1]
    assert rows[1]["match"] == "height 25%"


def test_a_miss_without_a_hit_is_still_listed():
    miss = {"id": "000001", "line": 0, "type": "Cyclist", "kind": "low_score"}
    rows = case_rows([miss], [])
    assert [r["role"] for r in rows] == ["miss"]
