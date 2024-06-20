import pytest
from pandas import DataFrame

from prepydf.preprocessing import Select
from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes, expected

one_feature = 'integer'
multiple_features = ['integer', 'float']
non_present_column = 'non present column'


@pytest.mark.parametrize('feature', [one_feature, multiple_features])
def test_select_one_column_is_dataframe(expected, feature):
    fit_transform_and_compare(DataFrame(expected[feature]), Select(feature))


def test_select_unknown_column(expected):
    with pytest.raises(KeyError):
        Select(non_present_column).fit_transform(expected)


def test_equals():
    assert Select(one_feature) != Select(multiple_features)
    assert Select(multiple_features) == Select(multiple_features)


def test_sklearn_compatibility():
    return check_attributes(Select)
