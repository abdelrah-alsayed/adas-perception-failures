from explain_adas.train import make_train_args


def test_train_args_are_explicit():
    args = make_train_args(seed=7, epochs=50, name="seed_7")
    assert args["seed"] == 7
    assert args["epochs"] == 50
    assert args["name"] == "seed_7"
    assert args["deterministic"] is True
    assert args["cache"] == "disk"
    assert args["optimizer"] == "AdamW"


def test_only_seed_and_name_change_between_runs():
    a = make_train_args(seed=0, epochs=50, name="a")
    b = make_train_args(seed=1, epochs=50, name="b")
    different = {key for key in a if a[key] != b[key]}
    assert different == {"seed", "name"}