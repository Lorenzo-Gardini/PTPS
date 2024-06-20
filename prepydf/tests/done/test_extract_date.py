from typing import Dict

import pytest
from pandas import Series

from prepydf.preprocessing import ExtractDate
from prepydf.tests.test_utilities import check_attributes, fit_transform_and_compare, full_dataframe, expected

feature = 'date_time'
feature_2 = 'date_time_2'
extractions = ['year', 'month', 'day', 'weekday', 'week', 'dayofyear']
mapping_value = {'year': 'year_changed', 'month': 'month_changed', 'day': 'day_changed'}


def test_extract_date_using_list_literals():
    extractions_mapping = _extract_map_values(full_dataframe()[feature])
    for cut in range(1, len(extractions)):
        expected = full_dataframe()
        for extraction in extractions[:cut]:
            expected[extraction] = extractions_mapping[extraction]
        expected.drop(columns=feature, inplace=True)
        fit_transform_and_compare(expected, ExtractDate(feature, extractions[:cut]))


@pytest.mark.parametrize('extraction', extractions)
def test_extract_date_using_single_value(extraction, expected):
    expected[extraction] = _extract_map_values(expected[feature])[extraction]
    expected.drop(columns=feature, inplace=True)
    fit_transform_and_compare(expected, ExtractDate(feature, extraction))


def test_extract_date_using_map(expected):
    extractions_mapping = _extract_map_values(expected[feature])
    for extraction, new_name in mapping_value.items():
        expected[new_name] = extractions_mapping[extraction]
    expected.drop(columns=feature, inplace=True)
    fit_transform_and_compare(expected, ExtractDate(feature, mapping_value))


def test_extract_date_using_none(expected):
    extractions_mapping = _extract_map_values(expected[feature])
    for extraction in extractions:
        expected[extraction] = extractions_mapping[extraction]
    expected.drop(columns=feature, inplace=True)
    fit_transform_and_compare(expected, ExtractDate(feature))


def test_raise_value_error_invalid_extraction(expected):
    with pytest.raises(ValueError):
        ExtractDate(feature, 'wrong').fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(ExtractDate)


def test_equals():
    assert ExtractDate(feature) != ExtractDate(feature_2)
    assert ExtractDate(feature, extractions) != ExtractDate(feature, mapping_value)
    assert ExtractDate(feature, extractions) != ExtractDate(feature, 'day')
    assert ExtractDate(feature, mapping_value) != ExtractDate(feature, 'day')
    assert ExtractDate(feature, extractions) == ExtractDate(feature, extractions)


def _extract_map_values(date_time_series: Series) -> Dict[str, Series]:
    return {
        'year': date_time_series.dt.year,
        'month': date_time_series.dt.month,
        'day': date_time_series.dt.day,
        'weekday': date_time_series.dt.weekday,
        'week': date_time_series.dt.isocalendar().week,
        'dayofyear': date_time_series.dt.dayofyear
    }
