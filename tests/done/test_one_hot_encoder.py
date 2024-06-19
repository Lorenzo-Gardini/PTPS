import pandas as pd
import pytest

from src.preprocessing import OneHotEncode
from tests.test_utilities import expected, check_attributes, fit_transform_and_compare

multi_categorical_bar = 'multi_categorical_|'
multi_categorical_hashtag = 'multi_categorical_#'
categorical = 'categorical'
divider_bar = '|'
divider_hashtag = '#'


@pytest.mark.parametrize('feature', [categorical, multi_categorical_bar])
@pytest.mark.parametrize('divider', [None, divider_hashtag, divider_bar])
def test_features_with_dividers(feature, divider, expected):
    one_hot_encoding = expected[feature].str.get_dummies(divider) if divider \
        else pd.get_dummies(expected[feature])
    expected = pd.concat([expected.drop(feature, axis=1), one_hot_encoding], axis=1)
    fit_transform_and_compare(expected, OneHotEncode(feature, divider))


def test_sklearn_compatibility():
    check_attributes(OneHotEncode)


@pytest.mark.parametrize('feature', [categorical, multi_categorical_bar])
@pytest.mark.parametrize('feature2', [categorical, multi_categorical_bar])
@pytest.mark.parametrize('divider', [None, divider_hashtag, divider_bar])
@pytest.mark.parametrize('divider2', [None, divider_hashtag, divider_bar])
def test_equals(feature, divider, feature2, divider2):
    if feature == feature2 and divider == divider2:
        assert OneHotEncode(feature, divider) == OneHotEncode(feature2, divider2)
    else:
        assert OneHotEncode(feature, divider) != OneHotEncode(feature2, divider2)
