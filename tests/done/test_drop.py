import pytest

from tests.test_utilities import fit_transform_and_compare, check_attributes, expected
from src.preprocessing import Drop

single_feature = 'integer'
multiple_feature = ['integer', 'float']
column_not_present = 'column not present'


@pytest.mark.parametrize('feature', [single_feature, multiple_feature])
def test_drop_one_colum(feature, expected):
    fit_transform_and_compare(expected.drop(columns=feature), Drop(feature))


def test_select_one_column(expected):
    drop_features = set(expected.columns) - {single_feature}
    fit_transform_and_compare(expected.drop(columns=drop_features), Drop(drop_features))


def test_invariant_unknown_column(expected):
    fit_transform_and_compare(expected, Drop(column_not_present))


def test_sklearn_compatibility():
    check_attributes(Drop)


def test_equals():
    assert Drop(single_feature) != Drop(multiple_feature)
    assert Drop(multiple_feature) == Drop(multiple_feature)
