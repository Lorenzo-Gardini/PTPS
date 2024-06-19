import pytest

from src.preprocessing import Clip
from tests.test_utilities import check_attributes, fit_transform_and_compare, expected

single_feature = 'integer'
negative_feature = 'negative_integer'
categorical_feature = 'categorical'
nan_feature = 'nan_numeric'
lower = 2
upper = 7
multiple_feature = ['integer', 'float']
column_not_present = 'column not present'


@pytest.mark.parametrize(('feature', 'error'), [(categorical_feature, TypeError), (column_not_present, KeyError)])
def test_clip_error_on_categorical_or_unknown_column(feature, error, expected):
    with pytest.raises(error):
        Clip(feature, lower, upper).fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(Clip)


def test_equals():
    assert Clip(single_feature, lower, upper) != Clip(multiple_feature, lower, upper)
    assert Clip(single_feature, lower, upper) == Clip(single_feature, lower, upper)


@pytest.mark.parametrize('feature', [single_feature, multiple_feature, negative_feature, nan_feature])
def test_with_feature(feature, expected):
    expected[feature] = expected[feature].clip(lower, upper)
    fit_transform_and_compare(expected, Clip(feature, lower, upper))
