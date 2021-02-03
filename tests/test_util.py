from sciutils import util as suu


def test_transpose_dict_roundtrip():
    list_of_dicts = [
        {'a': 2, 'b': 3},
        {'a': 4, 'b': 7},
    ]
    dict_of_lists = suu.dict_list_transpose(list_of_dicts)
    assert dict_of_lists == {
        'a': [2, 4],
        'b': [3, 7],
    }
    assert list_of_dicts == suu.dict_list_transpose(dict_of_lists)
