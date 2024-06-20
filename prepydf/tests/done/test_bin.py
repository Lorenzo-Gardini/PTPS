import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from prepydf.preprocessing import Bin
from prepydf.tests.test_utilities import check_attributes, fit_transform_and_compare

column = 'float'
n_bins = 3
multiple_bins = [2, 3, 5, 8]
labels_value = ['bad', 'medium', 'good']


def test_raise_error_no_fit(expected):
    with pytest.raises(NotFittedError):
        Bin(column, n_bins).transform(expected)


def test_sklearn_compatibility():
    check_attributes(Bin)


def test_equals():
    assert Bin(column, n_bins, labels_value, digitize=True) == Bin(column, n_bins, labels_value, digitize=True)


@pytest.mark.parametrize('labels', [False, labels_value])
@pytest.mark.parametrize('bins', [n_bins, multiple_bins])
@pytest.mark.parametrize('digitize', [True, False])
def test_bin_and_compare(bins, labels, digitize, expected):
    if digitize:
        _, bin_edges = pd.cut(expected[column], bins=bins, retbins=True)
        expected[column] = np.digitize(expected[column], bins=bin_edges, right=True)
    else:
        expected[column] = pd.cut(expected[column], bins=bins, labels=labels)
    fit_transform_and_compare(expected, Bin(column, bins=bins, labels=labels, digitize=digitize))
