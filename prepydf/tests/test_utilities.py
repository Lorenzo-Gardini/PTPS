from datetime import datetime, timedelta
from typing import Type, List, Optional

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from typeguard import typechecked

from prepydf import DataFramePreprocessFunction

__all__ = ['fit_transform_and_compare', 'check_attributes', 'full_dataframe', 'expected']

_BASE_SKLEARN_ATTRIBUTES = ['fit', 'transform', 'fit_transform']


@typechecked
def fit_transform_and_compare(expected: DataFrame, process_function: DataFramePreprocessFunction):
    new_df = full_dataframe()
    assert new_df is not process_function.fit_transform(X=new_df, y=None)
    assert_frame_equal(expected, process_function.fit_transform(X=full_dataframe(), y=None))


@typechecked
def check_attributes(preprocess_function: Type[DataFramePreprocessFunction],
                     extra_attributes: Optional[List[str]] = None):
    attributes = _BASE_SKLEARN_ATTRIBUTES + extra_attributes if extra_attributes is not None else []
    assert all(hasattr(preprocess_function, attribute) for attribute in attributes)


def full_dataframe() -> DataFrame:
    return pd.DataFrame({
        'integer': np.arange(1, 10),
        'negative_integer': np.arange(-2, 7),
        'float': np.linspace(2, 8, 9),
        'categorical': ['one', 'two', 'tree', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
        'categorical_repeated': ['most', 'most', 'most', 'few', 'few', 'few', 'medium', 'medium', 'medium'],
        'multi_categorical_|': ['a|b', 'a|b|c', 'a', 'c', 'b|c', 'a|c', 'b', 'b|a', 'c|a'],
        'multi_categorical_#': ['a|b', 'a|b|c', 'a', 'c', 'b|c', 'a|c', 'b', 'b|a', 'c|a'],
        'numerical_repeated': [1, 2, 2, 3, 3, 3, 4, 4, 4],
        'nan_numeric': [1, np.nan, 3, 4, np.nan, 5, np.nan, np.nan, 8],
        'nan_categorical': ['one', np.nan, 'tree', 'four', np.nan, 'five', np.nan, np.nan, np.nan],
        'repeated_integers': [1, 2, 3] * 3,
        'date_time': [datetime(year=2024, month=6, day=8) - timedelta(days=i * 20, seconds=i * 1000)
                      for i in range(9)]
    })


@pytest.fixture
def expected() -> DataFrame:
    return full_dataframe()
