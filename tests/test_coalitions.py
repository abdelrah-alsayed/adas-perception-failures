from explain_adas.coalitions import n_coalitions

def test_three_regions_give_eight_coalitions():
    assert n_coalitions(3) == 8


def test_sixteen_regions_give_65536_coalitions():
    assert n_coalitions(16) == 65536