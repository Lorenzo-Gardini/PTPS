from __future__ import annotations

from typing import Iterable, Union, Any, List, Optional
import sklearn
from numpy import ndarray
from pandas.api.types import (is_integer_dtype, is_float_dtype)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame, Series
from collections import defaultdict

from typeguard import typechecked

# --------------- CONSTANTS --------------------

# columns indices
USER_ID_COLUMN = 0
ITEM_ID_COLUMN = 1
RATINGS_COLUMN = 2
FEATURE_ID_COLUMN = 0
# dataframe data
DEFAULT_VALUE_WEIGHTS = 1
CATEGORICAL_FEATURE_DEFAULT_WEIGHT = 1
DEFAULT_WEIGHT_NAME = 'weight'
# sizes
MIN_COLUMNS_INTERACTIONS_DATAFRAME = 2
MAX_COLUMNS_INTERACTIONS_DATAFRAME = 3
MIN_COLUMNS_FEATURE_DATAFRAME = 2

# combination methods
PAIR = 0
CARTESIAN_PRODUCT = 1

# --------------- TYPE ALIAS --------------------

Id = Union[str, int]
Feature = Union[str, int, float]


# --------------- UTILITY --------------------


@typechecked
def to_ndarray(data: Optional[DataFrame | ndarray]):
    if data is not None:
        if isinstance(data, DataFrame):
            return data.values
        else:
            return data
    else:
        return None


def groupby(iterable, key_function, map_function=None, keep_order=True):
    groups = defaultdict(list if keep_order else set)
    for item in iterable:
        mapped_item = item if map_function is None else map_function(item)
        if keep_order:
            groups[key_function(item)].append(mapped_item)
        else:
            groups[key_function(item)].add(mapped_item)
    return dict(groups)


def at_least_list(data: Any) -> List[Any]:
    if isinstance(data, (str, bytes)) or not isinstance(data, Iterable):
        return [data]
    else:
        return list(data)


def opt_at_least_list(data: Any) -> Optional[List[Any]]:
    return at_least_list(data) if data is not None else None


def unique(values: Iterable[int | float | str] | ndarray) -> List[int | float | str]:
    return list(np.unique(values))


def add_weight(dataframe: DataFrame, weight: float | int) -> DataFrame:
    new_dataframe = dataframe.copy(deep=True)
    new_dataframe[RATINGS_COLUMN] = weight

    return new_dataframe


# ------------- DATAFRAME LOADING -------------

def to_dataframe(data: Any, column_names: List[str] = None) -> DataFrame:
    return DataFrame(data, columns=column_names)


def to_dataframe_utility_matrix(weights, user_ids=None, items_ids=None):
    return DataFrame(weights, index=user_ids, columns=items_ids)


# ---------------- CHECKS ---------------

@typechecked
def check_interactions_dataframe(dataset: DataFrame) -> None:
    if not (MIN_COLUMNS_INTERACTIONS_DATAFRAME <= len(dataset.columns) <= MAX_COLUMNS_INTERACTIONS_DATAFRAME):
        raise ValueError("DataFrame must have 2 or 3 columns")
    if len(dataset.columns) == MAX_COLUMNS_INTERACTIONS_DATAFRAME and not is_numerical(dataset.iloc[:, RATINGS_COLUMN]):
        raise ValueError("Ratings column, if present, must contain either int or float values")
    if has_na(dataset):
        raise ValueError("DataFrame can't contain NaN values")
    if has_duplicates(dataset, dataset.columns[[USER_ID_COLUMN, ITEM_ID_COLUMN]].tolist()):
        raise ValueError("DataFrame can't contain duplicates")


@typechecked
def check_features(dataframe: DataFrame) -> None:
    if len(dataframe.columns) < MIN_COLUMNS_FEATURE_DATAFRAME:
        raise ValueError("Features DataFrame must have at least 2 columns")
    if has_na(dataframe):
        raise ValueError("Features DataFrame can't contain NaN values")
    if has_duplicates(dataframe, dataframe.columns[FEATURE_ID_COLUMN]):
        raise ValueError("Features DataFrame can't contain duplicates")


def is_multi_categorical(series: Series, divider: str) -> bool:
    return is_categorical(series) and series.str.contains(divider, regex=False).sum() > 0


def has_na(dataframe: DataFrame) -> bool:
    return any(dataframe[column].isna().sum() > 0 for column in dataframe.columns)


def is_categorical(series: Series) -> bool:
    return not is_float_dtype(series.dtype) and not is_integer_dtype(series.dtype)


def is_numerical(series: Series) -> bool:
    return is_float_dtype(series.dtype) or is_integer_dtype(series.dtype)


def not_between_0_1(series: Series) -> bool:
    return is_numerical(series) and (series.max() > 1 or series.min() < 0)


def is_not_normalized(series: Series) -> bool:
    if is_numerical(series):
        normalized_value = sklearn.preprocessing.Normalizer().fit_transform(np.atleast_2d(series.values)).flatten()
        return np.array_equal(normalized_value, series.values)
    return False


def has_duplicates(dataframe: DataFrame, subset: Iterable[str] = None) -> bool:
    return dataframe.duplicated(subset=subset).sum() > 0


# TODO refactors
def dataframe_advisor(dataframe: DataFrame,
                      id_columns: str | [str],
                      is_feature_matrix: bool = False,
                      verbose: bool = True) -> None:
    if is_feature_matrix:
        if dataframe.columns.size < 2 and verbose:
            print('DataFrame has less than the minimum of 2 columns')
    else:
        if not (2 <= dataframe.columns.size <= 3) and verbose:
            print(f"DataFrame doesn't a minimum of 2 columns and a maximum of 3. "
                  f"It has {dataframe.columns.size} columns")

    has_na(dataframe, verbose)
    has_duplicates(dataframe, id_columns, verbose)

    for column in dataframe.drop(id_columns, axis=1):
        # TODO prints
        not_between_0_1(dataframe[column])
        is_not_normalized(dataframe[column], )
        is_multi_categorical(dataframe[column])


# --------------DESCRIBE--------------

def plot_hist(series, bins='auto', ax=None):
    plt.figure(series.name)
    sns.histplot(series, bins=bins, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_continuous_distribution(series, ax=None):
    plt.figure(series.name)
    sns.kdeplot(series, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(series, split=None, ax=None):
    series = series.str.split(split).explode() if split is not None else series
    grouped_values = series.value_counts()
    grouped_values_percent = series.value_counts(normalize=True) * 100
    groups_info = zip(grouped_values.index, grouped_values.values, grouped_values_percent.values)
    label_string = "{}  ({} - {:.2f}%)"
    labels = [label_string.format(index, value, percentage) for index, value, percentage in groups_info]
    plt.figure(series.name)
    sns.barplot(y=labels, x=grouped_values.values, orient='h', ax=ax).set(
        title=f"Number of features: {grouped_values.index.size} Total values: {grouped_values.sum()}")
    plt.tight_layout()
    plt.show()


'''
def describe(dataframe: DataFrame,
             display_dataframe=True,
             categorical=None,
             continuous=None,
             hist=None):
    categorical = np.atleast_1d(categorical) if categorical is not None else []
    continuous = np.atleast_1d(continuous) if continuous is not None else []
    hist = np.atleast_1d(hist) if hist is not None else []

    if isinstance(dataframe, preprocessing.PreprocessedDataFrame):
        dataframe = dataframe.dataframe

    if display_dataframe:
        print(dataframe)

    for categorical_value in np.atleast_1d(categorical):
        if isinstance(categorical_value, dict):
            for feature, split in categorical_value.items():
                plot_categorical_distribution(dataframe[feature], split=split)
        elif isinstance(categorical_value, str):
            plot_categorical_distribution(dataframe[categorical_value])

    for feature in continuous:
        plot_continuous_distribution(dataframe[feature])

    for hist_value in hist:
        if isinstance(hist_value, dict):
            for feature, bins in hist_value.items():
                plot_hist(dataframe[feature], bins)
        elif isinstance(hist_value, str):
            plot_hist(dataframe[hist_value])
'''
