import numpy as np
import pytest

from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes, full_dataframe
from prepydf import Cycle

col_name = 'repeated_integers'
not_existing_col = 'I_dont_exist'


@pytest.mark.parametrize('generic_cycle_value', [lambda x: x.nunique(), None, full_dataframe()[col_name].nunique()])
@pytest.mark.parametrize('cos_col_name', [None, 'tempo_1'])
@pytest.mark.parametrize('sin_col_name', [None, 'tempo_2'])
def test_generic_cycle_cos_col_name_sin_col_name_values(generic_cycle_value, cos_col_name, sin_col_name, expected):
    cycle_value = expected[col_name].nunique()
    cos_col_name = cos_col_name if cos_col_name is not None else f'{col_name}_cos'
    sin_col_name = sin_col_name if sin_col_name is not None else f'{col_name}_sin'
    expected[cos_col_name] = np.cos(np.pi * 2 * expected[col_name] / cycle_value)
    expected[sin_col_name] = np.sin(np.pi * 2 * expected[col_name] / cycle_value)
    expected = expected.drop(columns=col_name)
    fit_transform_and_compare(expected, Cycle(col_name, generic_cycle_value, cos_col_name, sin_col_name))


def test_sklearn_compatibility():
    check_attributes(Cycle)


@pytest.mark.parametrize('cycle_value', [None, 7])
@pytest.mark.parametrize('cycle_value2', [None, 7])
@pytest.mark.parametrize('cos_col_name', [None, 'tempo_1'])
@pytest.mark.parametrize('cos_col_name2', [None, 'tempo_1'])
@pytest.mark.parametrize('sin_col_name', [None, 'tempo_2'])
@pytest.mark.parametrize('sin_col_name2', [None, 'tempo_2'])
@pytest.mark.parametrize('col_name', [col_name, not_existing_col])
@pytest.mark.parametrize('col_name2', [col_name, not_existing_col])
def test_equals(cycle_value, cos_col_name, sin_col_name, cycle_value2, cos_col_name2, sin_col_name2, col_name, col_name2):
    if cycle_value == cycle_value2 and sin_col_name == sin_col_name2 and cos_col_name == cos_col_name2 and col_name == col_name2:
        assert Cycle(col_name, cycle_value, cos_col_name, sin_col_name) == Cycle(col_name2, cycle_value2, cos_col_name2, sin_col_name2)
    else:
        assert Cycle(col_name, cycle_value, cos_col_name, sin_col_name) != Cycle(col_name2, cycle_value2, cos_col_name2, sin_col_name2)

