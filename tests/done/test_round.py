import math

import pytest

from src.preprocessing import Round
from tests.test_utilities import fit_transform_and_compare, check_attributes, expected

column = 'float'
wrong_column = 'categorical'
digits = 1


def test_sklearn_compatibility():
    check_attributes(Round)


def test_raise_error_not_numerical_colum(expected):
    with pytest.raises(TypeError):
        Round(wrong_column).fit_transform(expected)


def test_raise_error_not_correct_mode(expected):
    with pytest.raises(Exception):
        Round(column, 'wrong_mode').fit_transform(expected)


@pytest.mark.parametrize('mode', ['round', 'ceil', 'floor'])
def test_round(mode, expected):
    if mode == 'round':
        expected[column] = expected[column].apply(lambda x: round(x, digits))
    elif mode == 'ceil':
        expected[column] = expected[column].apply(lambda x: math.ceil(x))
    elif mode == 'floor':
        expected[column] = expected[column].apply(lambda x: math.floor(x))
    fit_transform_and_compare(expected, Round(column, mode, digits))


@pytest.mark.parametrize('mode', ['round', 'ceil', 'floor'])
def test_equals(mode):
    for other_mode in ['round', 'ceil', 'floor']:
        if mode != other_mode:
            assert Round(column, mode) != Round(column, other_mode)
    assert Round(column, mode) == Round(column, mode)
