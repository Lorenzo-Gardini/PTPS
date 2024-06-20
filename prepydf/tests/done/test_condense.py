import pytest

from prepydf import Condense
from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes

categorical_condense = 'categorical_repeated'
numerical_condense = 'numerical_repeated'
not_present_column = 'I_don_exist'
default_separator = '|'


@pytest.mark.parametrize('feature', [categorical_condense, numerical_condense])
def test_condense_and_compare(feature, expected):
    expected = expected.groupby(feature).agg(lambda g: default_separator.join(g.astype(str).unique())).reset_index()
    fit_transform_and_compare(expected, Condense(feature, default_separator))


def test_not_present_column(expected):
    with pytest.raises(Exception):
        Condense(not_present_column, default_separator).fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(Condense)


@pytest.mark.parametrize('feature', [categorical_condense, numerical_condense])
@pytest.mark.parametrize('feature2', [categorical_condense, numerical_condense])
@pytest.mark.parametrize('separator', [default_separator, '#'])
@pytest.mark.parametrize('separator2', [default_separator, '#'])
def test_equals(feature, feature2, separator, separator2):
    if feature == feature2 and separator == separator2:
        assert Condense(feature, separator) == Condense(feature2, separator2)
    else:
        assert Condense(feature, separator) != Condense(feature2, separator2)
