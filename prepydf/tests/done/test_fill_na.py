from typing import List, Literal

import pandas as pd
import pytest

from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes
from prepydf.preprocessing import FillNa

fill_values = [0, "text", True, pd.Timestamp(2017, 1, 1, 12)]
fill_methods: List[Literal['backfill', 'bfill', 'pad', 'ffill']] = ['backfill', 'bfill', 'pad', 'ffill']
single_feature_no_nan = 'integer'
multiple_features_no_nan = ['integer', 'float']
multiple_features = ['nan_numeric', 'nan_categorical']
single_feature_numeric = 'nan_numeric'
single_feature_categorical = 'nan_categorical'
column_not_present = 'column not present'


def test_select_unknown_column(expected):
    with pytest.raises(KeyError):
        FillNa(column_not_present).fit_transform(expected)


def test_raise_error_if_both_method_and_value_are_provided(expected):
    with pytest.raises(Exception):
        FillNa(single_feature_numeric, value=0, method='pad').fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(FillNa)


@pytest.mark.parametrize('value', fill_values)
@pytest.mark.parametrize('features', [single_feature_numeric,
                                      single_feature_categorical,
                                      single_feature_no_nan,
                                      multiple_features,
                                      multiple_features_no_nan])
def test_fill_na_all_features(features, value, expected):
    expected[features] = expected[features].fillna(value=value)
    fit_transform_and_compare(expected, FillNa(features, value=value))


@pytest.mark.parametrize('features', [single_feature_numeric,
                                      single_feature_categorical,
                                      single_feature_no_nan,
                                      multiple_features,
                                      multiple_features_no_nan])
@pytest.mark.parametrize('method', fill_methods)
def test_fill_na_with_method_and_compare(features, method, expected):
    expected[features] = expected[features].fillna(method=method)
    fit_transform_and_compare(expected, FillNa(features, method=method))
