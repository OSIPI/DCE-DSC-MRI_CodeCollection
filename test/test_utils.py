from src.utils import add, remove_spaces
import pytest

@pytest.mark.parametrize('x, y, result', [
    (10,10,20),
    (12,0,12),
    (-1,-6,-7)
    ])

def test_add(x, y, result):
    assert add(x,y) == result


@pytest.mark.parametrize('data, result', [
    ('a b', 'ab'),
    ('sudarshan ragunathan', 'sudarshanragunathan')
    ])
def test_remove_spaces(data,result):
    assert remove_spaces(data) == result

