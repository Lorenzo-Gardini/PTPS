import numpy as np
import pytest
from pandas import DataFrame

from prepydf.preprocessing import Filter
from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes


def condition(dataframe: DataFrame):
    return dataframe['integer'] > 2


@pytest.mark.parametrize('filter_fn', [lambda x: condition,
                                       lambda x: condition(x).to_numpy(),
                                       lambda x: list(condition(x))])
def test_filter(expected, filter_fn):
    fit_transform_and_compare(expected[filter_fn(expected)], Filter(lambda x: filter_fn(x)))


@pytest.mark.parametrize('filter_fn', [lambda x: None, lambda: None, lambda: np.Nan])
def test_filter_invalid_function(expected, filter_fn):
    with pytest.raises(Exception):
        Filter(filter_fn).fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(Filter)


@pytest.mark.parametrize('filter_fn', [lambda x: x, lambda x: x.to_numpy(), lambda x: list(x)])
def test_equals(filter_fn):
    assert Filter(filter_fn) != Filter(lambda x: x > 5)
    assert Filter(filter_fn) == Filter(filter_fn)
