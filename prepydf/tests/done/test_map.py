import pytest

from prepydf.preprocessing import Map
from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes

dict_transform = {'most': 3}
dict_feature = 'categorical_repeated'
function_feature = 'integer'


@pytest.mark.parametrize(('feature', 'fn'), [(dict_feature, lambda x: dict_transform[x] if x in dict_transform else x),
                                             (function_feature, lambda x: 'under 5' if x < 5 else x)])
def test_features_with_function(feature, fn, expected):
    expected[feature] = expected[feature].map(fn)
    fit_transform_and_compare(expected, Map(feature, fn))


def test_sklearn_compatibility():
    check_attributes(Map)


def test_equals():
    assert Map(dict_feature, _nonsense_function) != Map(function_feature, _nonsense_function)
    assert Map(dict_feature, lambda x: x) != Map(dict_feature, _nonsense_function)
    assert Map(dict_feature, _nonsense_function) == Map(dict_feature, _nonsense_function)


def _nonsense_function(x):
    return x
