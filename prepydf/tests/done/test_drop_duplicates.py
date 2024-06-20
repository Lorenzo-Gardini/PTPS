import pytest

from prepydf import DropDuplicates
from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes

single_feature = 'categorical_repeated'
multiple_feature = ['numerical_repeated', 'categorical_repeated']
column_not_present = 'column not present'


def test_select_unknown_column(expected):
    with pytest.raises(KeyError):
        DropDuplicates(subset_features=column_not_present).fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(DropDuplicates)


@pytest.mark.parametrize('features', [single_feature, multiple_feature, None])
@pytest.mark.parametrize('mode', ['first', 'last', False])
@pytest.mark.parametrize('features2', [single_feature, multiple_feature, None])
@pytest.mark.parametrize('mode2', ['first', 'last', False])
def test_equals(features, mode, features2, mode2):
    if features == features2 and mode == mode2:
        assert DropDuplicates(features, mode) == DropDuplicates(features2, mode2)
    else:
        assert DropDuplicates(features, mode) != DropDuplicates(features2, mode2)


@pytest.mark.parametrize('features', [single_feature, multiple_feature, None])
@pytest.mark.parametrize('mode', ['first', 'last', False])
def test_all_keeps(features, mode, expected):
    fit_transform_and_compare(expected.drop_duplicates(subset=features, keep=mode),
                              DropDuplicates(subset_features=features, keep=mode))
