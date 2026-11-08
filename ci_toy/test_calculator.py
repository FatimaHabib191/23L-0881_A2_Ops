from calculator import add


def test_add():
    # For the red run, change 5 to 6 and push. Change it back for the green run.
    assert add(2, 3) == 5
