import pytest

from prepydf.commons import at_least_list
from prepydf.tests.test_utilities import check_attributes, fit_transform_and_compare, expected
from prepydf.preprocessing import DropNa

single_feature = 'nan_numeric'
multiple_feature = ['nan_numeric', 'nan_categorical']
column_not_present = 'column not present'


@pytest.mark.parametrize('subset', [single_feature, multiple_feature, None])
def test_subsets(subset, expected):
    fit_transform_and_compare(expected.dropna(subset=at_least_list(subset) if subset is not None else None),
                              DropNa(subset_features=subset))


def test_select_unknown_column(expected):
    with pytest.raises(KeyError):
        DropNa(column_not_present).fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(DropNa)


def test_equals():
    assert DropNa(single_feature) != DropNa(multiple_feature)
    assert DropNa(single_feature) == DropNa(single_feature)
