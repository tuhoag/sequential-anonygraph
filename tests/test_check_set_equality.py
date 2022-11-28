def _test_check_set_equality(set1, set2):
    return set1 == set2

def main():
    set1 = {(0, 0), (1, 2)}
    set2 = {(1, 2), (0, 0)}

    # assert set1==set2, "{} should be equal to {}".format(set1, set2)
    assert _test_check_set_equality(set1, set2), "{} should be equal to {}".format(set1, set2)

    set1 = {frozenset({(0, 0), (1, 2)}),frozenset({(2, 1), (3, 2)})}
    set2 = {frozenset({(1, 2), (0, 0)}),frozenset({(3, 2), (2, 1)})}
    assert _test_check_set_equality(set1, set2), "{} should be equal to {}".format(set1, set2)

    Ia1 = frozenset({frozenset({(0, 0), (1, 2)}),frozenset({(2, 1), (3, 2)})})
    print(hash(Ia1))
main()