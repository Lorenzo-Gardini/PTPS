from __future__ import annotations

import math
from abc import abstractmethod, ABC  # for PreprocessFunction
from itertools import chain
from typing import Any, Dict, Set, Callable, Literal, Optional, Union, List, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.groupby import SeriesGroupBy
from scipy import sparse
from sklearn import preprocessing  # for Normalizer, MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, has_fit_parameter
from typeguard import typechecked

from src.commons import unique, at_least_list, opt_at_least_list


class DataFramePreprocessFunction(TransformerMixin, BaseEstimator, ABC, object):

    @typechecked
    def fit(self, dataframe: DataFrame, y=None, **fit_params) -> DataFramePreprocessFunction:
        return self

    @abstractmethod
    def transform(self, dataframe: DataFrame) -> DataFrame:
        pass

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)


class InverseTransformer(ABC):
    @abstractmethod
    def inverse_transform(self, dataframe: DataFrame) -> DataFrame:
        pass


class _GroupByFunction(DataFramePreprocessFunction):
    @typechecked
    def __init__(self,
                 features: Optional[str | Iterable[str]],
                 group_by_features: Optional[str | Iterable[str]] = None):
        self.features = features
        self.group_by_features = group_by_features

    @typechecked
    def fit(self, dataframe: DataFrame, y=None, **fit_params) -> _GroupByFunction:
        group_by_opt_list = opt_at_least_list(self.group_by_features)
        features_list = at_least_list(self.features) if self.features is not None else dataframe.columns
        if group_by_opt_list is None:
            self.transformers_ = {feature: self._fit_transformer(dataframe[[feature]])
                                  for feature in features_list}
        else:
            groups = dataframe.groupby(group_by_opt_list)
            # fit every subgroup in a feature
            self.transformers_ = {key: {feature: self._fit_transformer(group[[feature]])
                                        for feature in features_list}
                                  for key, group in groups}
        return self

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        check_is_fitted(self)
        group_by_opt_list = opt_at_least_list(self.group_by_features)
        features_list = at_least_list(self.features) if self.features is not None else dataframe.columns
        new_df = dataframe.copy()
        if group_by_opt_list is None:
            for feature in features_list:
                new_df[feature] = self._transform_transformer(self.transformers_[feature], dataframe[[feature]])
        else:
            groups = new_df.groupby(group_by_opt_list)
            for feature in features_list:
                new_df[feature] = groups[[feature]].apply(
                    lambda x: self._transform_transformer(self.transformers_[x.name][feature], x))
        return new_df

    @abstractmethod
    def _fit_transformer(self, feature: DataFrame) -> Any:
        pass

    @abstractmethod
    def _transform_transformer(self, transformer: Any, feature: DataFrame) -> Iterable:
        pass

    def __eq__(self, other: Any) -> bool:
        return super(_GroupByFunction, self).__eq__(other) and \
            at_least_list(self.features) == at_least_list(other.features) and \
            opt_at_least_list(self.group_by_features) == opt_at_least_list(other.group_by_features)


class _InvertibleGroupByFunction(_GroupByFunction, InverseTransformer):
    @typechecked
    def __init__(self, features: str | Iterable[str], group_by_features: Optional[str | Iterable[str]] = None):
        super(_InvertibleGroupByFunction, self).__init__(features, group_by_features)

    @typechecked()
    def inverse_transform(self, dataframe: DataFrame) -> DataFrame:
        opt_groups = opt_at_least_list(self.group_by_features)
        features_list = at_least_list(self.features) if self.features is not None else dataframe.columns
        if opt_groups is None:
            for feature in features_list:
                dataframe[feature] = self._inverse_transformer(self.transformers_[feature], dataframe[[feature]])
        else:
            groups = dataframe.groupby(opt_groups)
            for feature in features_list:
                dataframe[feature] = groups[[feature]] \
                    .apply(lambda x: self._inverse_transformer(self.transformers_[x.name][feature], x))
        return dataframe

    @abstractmethod
    def _inverse_transformer(self, transformer: Any, feature: DataFrame) -> Iterable:
        pass


class _Scaler(_InvertibleGroupByFunction):
    def __init__(self, features: Optional[str | Iterable[str]] = None,
                 group_by_features: Optional[str | Iterable[str]] = None):
        super(_InvertibleGroupByFunction, self).__init__(group_by_features=group_by_features, features=features)

    def _transformer_transformers(self, transformer: Any, one_column_dataframe: DataFrame) -> DataFrame:
        return _replace_dataframe_data(one_column_dataframe, lambda d: transformer.transform(d).flatten())

    def _inverse_transformer(self, transformer: Any, one_column_dataframe: DataFrame) -> DataFrame:
        return _replace_dataframe_data(one_column_dataframe, lambda d: transformer.inverse_transform(d).flatten())

    def _fit_transformer(self, feature: DataFrame) -> Any:
        return transformer.fit(feature)

    @abstractmethod
    def _transformer_supplier(self) -> Any:
        pass


class _BinDensity(DataFramePreprocessFunction, ABC):
    def __init__(self, feature: str, binning_value: str | float | int, divider: str = None):
        super(_BinDensity, self).__init__(transform_on_copy=True)
        self.feature = feature
        self.binning_value = binning_value
        self.divider = divider

    def _fit(self, dataframe: DataFrame, y=None):
        if self.divider is None:
            column = dataframe[self.feature]
        else:
            items_features = dataframe[self.feature].str.split(self.divider).values  # list of lists
            all_features = chain(*items_features)  # flat map
            column = Series(all_features)

        self.feature_values_ = set(column.values)
        self.excluded_features_, self.grouped_features_ = self._split(column)

    # TODO fix
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        check_is_fitted(self)
        column = dataframe[self.feature]
        values = column.values if self.divider is None else list(chain(*column.str.split(self.divider).values))
        extra_features = set(values) - self.feature_values_  # new feature non-present during fit

        if extra_features:
            raise ValueError(f'{extra_features} no present during fit call')
        # transformation
        if self.divider is None:
            dataframe[self.feature] = column.map(lambda x: _keep_or_replace(x,
                                                                            self.excluded_features_,
                                                                            self.binning_value))
        else:
            dataframe[self.feature] = [
                # for every list of features, eventually replace the value.
                # Then unique again to remove repeated binning_value
                self.divider.join(unique([_keep_or_replace(feature, self.excluded_features_, self.binning_value)
                                          for feature in multiple_feature]))
                for multiple_feature in values]

        return dataframe

    @abstractmethod
    def _split(self, column: pd.Series) -> (Set[Any], Set[Any]):
        pass

    def __eq__(self, other: Any) -> bool:
        return super(_BinDensity, self).__eq__(other) and \
            self.feature == other.feature and \
            self.binning_value == other.binning_value and \
            self.divider == other.divider


# --------------UTILITY FUNCTION--------------

def _replace_dataframe_data(dataframe: DataFrame, mapping_function: Callable[[DataFrame], DataFrame]) -> DataFrame:
    return DataFrame(mapping_function(dataframe), columns=dataframe.columns, index=dataframe.index)


def _keep_or_replace(value: Any, to_keep: Iterable[Any], replace_value: Any) -> Any:
    return value if value in to_keep else replace_value


# -------------- PREPROCESS FUNCTIONS --------------

class Select(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, features: str | Iterable[str]):
        self.features = features

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        selected_features = dataframe[self.features].copy()
        if isinstance(selected_features, pd.Series):
            return selected_features.to_frame()
        return selected_features

    def __eq__(self, other: Any) -> bool:
        return super(Select, self).__eq__(other) and np.array_equal(self.features, other.features)


class FillNa(DataFramePreprocessFunction):
    @typechecked
    def __init__(self,
                 features: str | Iterable[str],
                 value: Optional[Any] = None,
                 method: Optional[Literal['backfill', 'bfill', 'ffill', 'pad']] = None):
        self.features = features
        self.value = value
        self.method = method

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        new_df = dataframe.copy()
        new_df[self.features] = new_df[self.features].fillna(value=self.value, method=self.method)
        return new_df

    def __eq__(self, other: Any) -> bool:
        return super(FillNa, self).__eq__(other) and \
            self.features == other.features and \
            self.value == other.value and \
            self.method == other.method


class Filter(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, filter_function: Callable[[DataFrame], Iterable[bool]]):
        self.filter_function = filter_function

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        return dataframe[self.filter_function(dataframe)]

    def __eq__(self, other: Any) -> bool:
        return super(Filter, self).__eq__(other) and self.filter_function == other.filter_function


class Drop(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, features: str | Iterable[str]):
        self.features = features

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        selected_columns = [column for column in dataframe.columns if column not in at_least_list(self.features)]
        return Select(selected_columns).transform(dataframe)

    def __eq__(self, other: Any) -> bool:
        return super(Drop, self).__eq__(other) and np.array_equal(self.features, other.features)


class Rename(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, columns_dict: Dict[str, str]):
        self.columns_dict = columns_dict

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        return dataframe.rename(columns=self.columns_dict).copy()

    def __eq__(self, other: Any) -> bool:
        return super(Rename, self).__eq__(other) and self.columns_dict == other.columns_dict


class Update(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, update: DataFrame | Series | Dict[str, Any]):
        self.update = update

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        new_df = dataframe.copy()
        if isinstance(self.update, Series):
            # add named column
            if self.update.name is not None:
                new_df[self.update.name] = self.update
            # append column
            else:
                new_df = pd.concat([dataframe, self.update], axis=1)
        elif isinstance(self.update, DataFrame):  # not concatenate (more optimized) due to column rewrite case
            for column in self.update.columns:
                new_df[column] = self.update[column]
        elif isinstance(self.update, Dict):
            for column, values in self.update.items():
                new_df[column] = values
        return new_df

    def __eq__(self, other: Any) -> bool:
        if not super(Update, self).__eq__(other):
            return False
        if not isinstance(self.update, type(other.update)):
            return False
        if isinstance(self.update, DataFrame):
            return self.update.equals(other.update)
        elif isinstance(self.update, Series):
            return self.update.equals(other.update)
        elif isinstance(self.update, Dict):
            return self.update == other.update
        else:
            return False


class Round(DataFramePreprocessFunction):
    @typechecked
    def __init__(self,
                 features: str | Iterable[str],
                 mode: Literal['ceil', 'floor', 'round'] = 'round',
                 digits: Optional[int] = None):
        self.features = features
        self.mode = mode
        self.digits = digits

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        if self.mode == 'ceil':
            function = math.ceil
        elif self.mode == 'floor':
            function = math.floor
        else:
            def round_with_digits(x):
                return round(x, self.digits)

            function = round_with_digits
        new_df = dataframe.copy()
        for feature in at_least_list(self.features):
            new_df[feature] = new_df[feature].map(function)
        return new_df

    def __eq__(self, other: Any) -> bool:
        return super(Round, self).__eq__(other) and \
            self.features == self.features and \
            self.mode == other.mode and \
            self.digits == other.digits


class Bin(DataFramePreprocessFunction):
    @typechecked
    def __init__(self,
                 feature: str,
                 bins: int | Iterable[int] | Iterable[float],
                 labels: Optional[Iterable[str] | Literal[False]] = None,
                 digitize: bool = False):
        self.feature = feature
        self.bins = bins
        self.labels = labels
        self.digitize = digitize

    @typechecked
    def fit(self, dataframe: DataFrame, y=None, **fit_params) -> Bin:
        _, self.bin_edges_ = pd.cut(dataframe[self.feature], bins=self.bins, retbins=True)
        return self

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        check_is_fitted(self)
        new_df = dataframe.copy()
        if self.digitize:
            new_df[self.feature] = np.digitize(dataframe[self.feature], bins=self.bin_edges_, right=True)
        else:
            new_df[self.feature] = pd.cut(dataframe[self.feature], bins=self.bin_edges_, labels=self.labels)
        return new_df

    def __eq__(self, other: Any) -> bool:
        return super(Bin, self).__eq__(other) and \
            self.feature == other.feature and \
            self.bins == other.bins and \
            self.labels == other.labels


# TODO test
class BinThreshold(_BinDensity):
    def __init__(self, feature, binning_value, threshold=0.2, divider=None):
        super(BinThreshold, self).__init__(feature, binning_value, divider)
        self.threshold = threshold

    def fit(self, dataframe: DataFrame, y=None, **fit_params) -> BinThreshold:
        super(BinThreshold, self).fit(dataframe, y, **fit_params)
        return self

    def _split(self, column: pd.Series) -> (Set[Any], Set[Any]):
        value_counts_percent = column.value_counts(normalize=True)
        select_over_threshold = value_counts_percent.values >= self.threshold
        return set(value_counts_percent[select_over_threshold].index), set(
            value_counts_percent[~select_over_threshold].index)

    def __eq__(self, other: Any) -> bool:
        return super(BinThreshold, self).__eq__(other) and self.threshold == other.threshold


# TODO test
class BinCumulative(_BinDensity):
    def __init__(self,
                 feature: str,
                 binning_value: Any,
                 cumulative_threshold: float = 0.7,
                 divider: str = None):
        super(BinCumulative, self).__init__(feature, binning_value, divider)
        self.cumulative_threshold = cumulative_threshold

    def fit(self, dataframe: DataFrame, y=None, **fit_params) -> BinCumulative:
        super(BinCumulative, self).fit(dataframe, y, **fit_params)
        return self

    def _split(self, column: pd.Series) -> (Set[Any], Set[Any]):
        value_counts_percent = column.value_counts(normalize=True)
        remaining_items = list(value_counts_percent.items())  # get items as tuple
        selected_items = []
        selected_items_weight = 0

        while remaining_items and selected_items_weight < self.cumulative_threshold:
            head, *remaining_items = remaining_items
            selected_items.append(head)
            selected_items_weight += head[1]  # add weight to total
        # extract only class
        return {item[0] for item in selected_items}, {item[0] for item in remaining_items}

    def __eq__(self, other: Any) -> bool:
        return super(BinCumulative, self).__eq__(other) and self.cumulative_threshold == other.cumulative_threshold


class ExtractDate(DataFramePreprocessFunction):
    _DEFAULT_VALUES: List[Literal] = ['year', 'month', 'day', 'weekday', 'week', 'dayofyear']

    @typechecked
    def __init__(self,
                 feature: str,
                 extractions: Optional[
                     Literal['year', 'month', 'day', 'weekday', 'week', 'dayofyear'] |
                     Iterable[str] |
                     Dict[str, str]
                     ] = None):
        self.feature = feature
        self.extractions = extractions

    def transform(self, dataframe: DataFrame) -> DataFrame:
        new_df = dataframe.drop(columns=self.feature).copy()
        components_dict = self._extract_date_component(dataframe[self.feature])
        if self.extractions is None:
            components = self._DEFAULT_VALUES
            names = self._DEFAULT_VALUES
        elif isinstance(self.extractions, Dict):
            components = self.extractions.keys()
            names = self.extractions.values()
        else:
            components = at_least_list(self.extractions)
            names = at_least_list(self.extractions)

        if not all(component in self._DEFAULT_VALUES for component in components):
            raise ValueError(f'extractions must be one or a Iterable of {self._DEFAULT_VALUES} '
                             f'or a Dict[{self._DEFAULT_VALUES}, str]')
        for component, name in zip(components, names):
            new_df[name] = components_dict[component]
        return new_df

    @staticmethod
    def _extract_date_component(date_time_series: Series) -> Dict[str, Series]:
        return {
            'year': date_time_series.dt.year,
            'month': date_time_series.dt.month,
            'day': date_time_series.dt.day,
            'weekday': date_time_series.dt.weekday,
            'week': date_time_series.dt.isocalendar().week,
            'dayofyear': date_time_series.dt.dayofyear
        }

    def __eq__(self, other: Any) -> bool:
        return super(ExtractDate, self).__eq__(other) and \
            self.feature == other.feature and \
            self.extractions == other.extractions


class DropNa(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, subset_features: Optional[str | Iterable[str]] = None):
        self.subset_features = subset_features

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        opt_list_features = opt_at_least_list(self.subset_features)
        return dataframe.dropna(subset=opt_list_features).copy()

    def __eq__(self, other: Any) -> bool:
        return super(DropNa, self).__eq__(other) and np.array_equal(self.subset_features, other.subset_features)


class DropDuplicates(DataFramePreprocessFunction):
    @typechecked
    def __init__(self,
                 subset_features: Optional[str | Iterable[str]] = None,
                 keep: Literal['first', 'last', False] = "last"):
        self.subset_features = subset_features
        self.keep = keep

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame | Series:
        opt_list_features = opt_at_least_list(self.subset_features)
        return dataframe.drop_duplicates(subset=opt_list_features, keep=self.keep).copy()

    def __eq__(self, other: Any) -> bool:
        return super(DropDuplicates, self).__eq__(other) and \
            np.array_equal(self.subset_features, other.subset_features) and \
            self.keep == other.keep


class Cycle(DataFramePreprocessFunction):
    @typechecked
    def __init__(self,
                 feature: str,
                 cycle_value: Optional[float | int | Callable[[Series], float | int]] = None,
                 cos_column_name: Optional[str] = None,
                 sin_column_name: Optional[str] = None):
        self.feature = feature
        self.cycle_value = cycle_value
        self.cos_column_name = cos_column_name
        self.sin_column_name = sin_column_name

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        computed_cycle_value = self._compute_cycle_value(dataframe)
        cos_name = self.cos_column_name if self.cos_column_name else f'{self.feature}_cos'
        sin_name = self.sin_column_name if self.sin_column_name else f'{self.feature}_sin'
        new_df = dataframe.drop(columns=self.feature).copy()
        new_df[cos_name] = np.cos(np.pi * 2 * dataframe[self.feature] / computed_cycle_value)
        new_df[sin_name] = np.sin(np.pi * 2 * dataframe[self.feature] / computed_cycle_value)
        return new_df

    @typechecked
    def _compute_cycle_value(self, dataframe: DataFrame):
        if isinstance(self.cycle_value, Callable):
            return self.cycle_value(dataframe[self.feature])
        if isinstance(self.cycle_value, (int, float)):
            return self.cycle_value
        else:
            return dataframe[self.feature].nunique()

    def __eq__(self, other: Any) -> bool:
        return super(Cycle, self).__eq__(other) and \
            self.cycle_value == other.cycle_value and \
            self.feature == other.feature and \
            self.cos_column_name == other.cos_column_name and \
            self.sin_column_name == other.sin_column_name


class Clip(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, features: str | Iterable[str], lower: int | float, upper: int | float):
        self.features = features
        self.lower = lower
        self.upper = upper

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        new_df = dataframe.copy(deep=True)
        for feature in at_least_list(self.features):
            new_df[feature] = new_df[feature].clip(lower=self.lower, upper=self.upper)
        return new_df

    def __eq__(self, other: Any) -> bool:
        return super(Clip, self).__eq__(other) and \
            np.array_equal(self.features, other.features) and \
            self.lower == other.lower and \
            self.upper == other.upper


class LabelEncoder(DataFramePreprocessFunction, InverseTransformer):
    @typechecked
    def __init__(self, feature: str):
        self.feature = feature
        self._encoder = preprocessing.LabelEncoder()

    @typechecked
    def fit(self, dataframe: DataFrame, y=None, **fit_params) -> LabelEncoder:
        self._encoder.fit(dataframe[self.feature])
        self.classes_ = self._encoder.classes_
        return self

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        check_is_fitted(self)
        new_df = dataframe.copy(deep=True)
        new_df[self.feature] = self._encoder.transform(dataframe[self.feature])
        return new_df

    @typechecked
    def inverse_transform(self, dataframe: DataFrame) -> DataFrame | Series:
        check_is_fitted(self)
        new_df = dataframe.copy(deep=True)
        new_df[self.feature] = self._encoder.inverse_transform(dataframe[self.feature])
        return new_df

    def __eq__(self, other: Any) -> bool:
        base_eq_condition = super(LabelEncoder, self).__eq__(other) and self.feature == other.feature
        if has_fit_parameter(self, 'classes_') and has_fit_parameter(other, 'classes_'):
            return base_eq_condition and np.array_equal(self.classes_, other.classes_)
        return base_eq_condition


class Map(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, feature: str, mapping: Optional[Dict[str, Any] | Callable] = None):
        self.feature = feature
        self.mapping = mapping

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        new_df = dataframe.copy()
        if isinstance(self.mapping, Dict):
            new_df[self.feature] = dataframe[self.feature].map(lambda x: self.mapping[x] if x in self.mapping else x)
        elif isinstance(self.mapping, Callable):
            new_df[self.feature] = dataframe[self.feature].map(self.mapping)
        return new_df

    def __eq__(self, other: Any) -> bool:
        return super(Map, self).__eq__(other) and self.feature == other.feature and self.mapping == other.mapping


class OneHotEncode(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, feature: str, divider: Optional[str] = None):
        self.feature = feature
        self.divider = divider

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        one_hot_feature_matrix = self._one_hot_series(dataframe[self.feature], self.divider)
        dataframe = pd.concat([dataframe.drop(columns=self.feature), one_hot_feature_matrix], axis=1)
        return dataframe

    @staticmethod
    def _one_hot_series(series: Series, divider: Optional[str] = None):
        return series.str.get_dummies(divider) if divider else pd.get_dummies(series)

    def __eq__(self, other: Any) -> bool:
        return super(OneHotEncode, self).__eq__(other) and \
            self.feature == other.feature and \
            self.divider == other.divider


class StandardScaler(_InvertibleGroupByFunction):
    @typechecked
    def __init__(self,
                 features: Optional[str | Iterable[str]] = None,
                 group_by_features: Optional[str | Iterable[str]] = None):
        super(StandardScaler, self).__init__(features=features, group_by_features=group_by_features)

    @typechecked
    def _fit_transformer(self, feature: DataFrame) -> object:
        return preprocessing.StandardScaler().fit(feature)

    @typechecked
    def _transform_transformer(self, transformer: preprocessing.StandardScaler, feature: DataFrame) -> Iterable:
        return transformer.transform(feature).flatten()

    @typechecked
    def _inverse_transformer(self, transformer: preprocessing.StandardScaler, feature: DataFrame) -> Iterable:
        return transformer.inverse_transform(feature).flatten()


class MinMaxScaler(_InvertibleGroupByFunction):
    @typechecked
    def __init__(self,
                 features: Optional[str | Iterable[str]],
                 group_by_features: Optional[str | Iterable[str]] = None):
        super(MinMaxScaler, self).__init__(features=features, group_by_features=group_by_features)

    @typechecked
    def _inverse_transformer(self, transformer: preprocessing.MinMaxScaler, feature: DataFrame) -> Iterable:
        return transformer.inverse_transform(feature).flatten()

    @typechecked
    def _fit_transformer(self, feature: DataFrame) -> object:
        return preprocessing.MinMaxScaler().fit(feature)

    @typechecked
    def _transform_transformer(self, transformer: preprocessing.MinMaxScaler, feature: DataFrame) -> Iterable:
        return transformer.transform(feature).flatten()


class Normalizer(_GroupByFunction):
    @typechecked
    def __init__(self,
                 features: Optional[str | Iterable[str]] = None,
                 group_by_features: Optional[str | Iterable[str]] = None,
                 norm: Literal['l1', 'l2', 'max'] = 'l2'):
        super(Normalizer, self).__init__(group_by_features=group_by_features, features=features)
        self.norm = norm

    @typechecked
    def _fit_transformer(self, feature: DataFrame) -> object:
        return preprocessing.Normalizer(norm=self.norm).fit(feature.T)

    @typechecked
    def _transform_transformer(self, transformer: preprocessing.Normalizer, feature: DataFrame) -> Iterable:
        return DataFrame(transformer.transform(feature.T).flatten(), columns=feature.columns, index=feature.index)

    def __eq__(self, other: Any) -> bool:
        return super(Normalizer, self).__eq__(other) and self.norm == other.norm


class Condense(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, feature: str, join_separator: str):
        self.feature = feature
        self.join_separator = join_separator

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        def aggregate(group):
            return self.join_separator.join(group.astype(str).unique())

        new_df = dataframe.copy()
        return new_df.groupby(self.feature).agg(aggregate).reset_index()

    def __eq__(self, other: Any) -> bool:
        return super(Condense, self).__eq__(other) and \
            self.feature == other.feature and \
            self.join_separator == other.join_separator


class ToCOOMatrix(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, first_column_name: str, second_column_name: str, data_column_name: str):
        self.first_column_name = first_column_name
        self.second_column_name = second_column_name
        self.data_column_name = data_column_name

    @typechecked
    def transform(self, utility_matrix: DataFrame) -> DataFrame:
        sparse_matrix = sparse.csr_matrix(utility_matrix.values)
        non_zero_row, non_zero_col = sparse_matrix.nonzero()
        coo_matrix = [(utility_matrix.index[r], utility_matrix.columns[c], utility_matrix.iloc[r][c])
                      for r, c in zip(non_zero_row, non_zero_col)]

        return DataFrame(coo_matrix, columns=[self.first_column_name, self.second_column_name, self.data_column_name])

    def __eq__(self, other: Any) -> bool:
        return super(ToCOOMatrix, self).__eq__(other) and \
            self.first_column_name == other.user_column_name and \
            self.data_column_name == other.item_column_name and \
            self.data_column_name == other.weights_column_name


# -----------PIPELINE---------------

class PreprocessPipeline(DataFramePreprocessFunction):
    @typechecked
    def __init__(self, preprocess_functions: List[DataFramePreprocessFunction]):
        self.preprocess_functions = preprocess_functions

    @typechecked
    def fit(self, dataframe: DataFrame, y=None, **fit_params) -> PreprocessPipeline:
        for function in self.preprocess_functions:
            function.fit(dataframe)
        return self

    @typechecked
    def __getitem__(self, index: int) -> DataFramePreprocessFunction:
        return self.preprocess_functions[index]

    @typechecked
    def transform(self, dataframe: DataFrame) -> DataFrame:
        new_df = dataframe.copy()
        for current_function in self.preprocess_functions:
            new_df = current_function.transform(new_df)
        return new_df

    def __eq__(self, other: Any) -> bool:
        return super(PreprocessPipeline, self).__eq__(other) and \
            self.preprocess_functions == other.preprocess_functions

# # ------------ AUTO PREPROCESS -----------
#
#
# def _apply_and_append(dataframe: DataFrame,
#                       preprocess_function: DataFramePreprocessFunction,
#                       prev_preprocess_function: List[DataFramePreprocessFunction]) \
#         -> Tuple[DataFrame, List[DataFramePreprocessFunction]]:
#     dataframe = preprocess_function.fit_transform(dataframe)
#     return dataframe, prev_preprocess_function + [preprocess_function]
#
#
# # TODO test
# def auto_preprocess_weights_dataframe(dataframe: DataFrame,
#                                       dense_input: bool = False,
#                                       verbose: bool = True,
#                                       **kwargs) -> Tuple[DataFrame, List[DataFramePreprocessFunction]]:
#     # preprocess functions
#     preprocess_functions = []
#
#     check_is_dataframe(dataframe)
#
#     if dataframe.columns.size < MIN_COLUMNS_INTERACTIONS_DATAFRAME:
#         raise ValueError('DataFrame must have at least 2 ids columns')
#
#     if dataframe.columns.size == 2:
#         print_if_verbose('DataFrame has two columns, no preprocessing required', verbose)
#         return dataframe, []
#
#     # sparse input
#     if dense_input:
#         print_if_verbose('Transform matrix to COO form', verbose)
#         function = ToCOOMatrix(kwargs.get('coo_first_column_name', 'user_id'),
#                                kwargs.get('coo_second_column_name', 'item_id'),
#                                kwargs.get('data_column_name', 'rating'))
#         dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#     # reduce columns number
#     if dataframe.columns.size > MAX_COLUMNS_INTERACTIONS_DATAFRAME:
#         print_if_verbose('DataFrame has more than 3 columns. Select only first 3 ones', verbose)
#         function = Select(dataframe.columns[:MAX_COLUMNS_INTERACTIONS_DATAFRAME])
#         dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#     # has na
#     if has_na(dataframe):
#         print_if_verbose('Drop all Nans from DataFrame', verbose)
#         function = DropNa()
#         dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#     # has duplicates
#     if has_duplicates(dataframe, dataframe.columns[:RATINGS_COLUMN]):
#         print_if_verbose('Drop duplicates from first 2 columns', verbose)
#         function = DropDuplicates(subset_features=dataframe.columns[:RATINGS_COLUMN])
#         dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#     # weight col is categorical
#     if is_categorical(dataframe.iloc[:, RATINGS_COLUMN]):
#         print_if_verbose('Label encoding on weights', verbose)
#         function = LabelEncoder(dataframe.columns[RATINGS_COLUMN])
#         dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#
#     return dataframe, preprocess_functions
#
#
# _MAX_CATEGORICAL_VALUES = 10
# _BINNING_THRESHOLD = 0.15
# _BINNING_VALUE = 'other'
# _DEFAULT_DIVIDER = '|'
#
#
# # TODO test
# def auto_preprocess_features_dataframe(dataframe: DataFrame,
#                                        verbose: bool = True,
#                                        **kwargs) -> Tuple[DataFrame, List[DataFramePreprocessFunction]]:
#     binning_value = kwargs.get('binning_value', _BINNING_VALUE)
#     binning_threshold = kwargs.get('binning_threshold', _BINNING_THRESHOLD)
#     divider = kwargs.get('divider', _DEFAULT_DIVIDER)
#     preprocess_functions = []
#
#     check_is_dataframe(dataframe)
#
#     if dataframe.columns.size < 2:
#         raise ValueError("Features DataFrame must have id column and at least one feature column")
#
#     # has na
#     if has_na(dataframe):
#         print_if_verbose('Drop all Nans from DataFrame', verbose)
#         function = DropNa()
#         dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#     # has duplicate
#     if has_duplicates(dataframe, subset=dataframe.columns[FEATURE_ID_COLUMN]):
#         print_if_verbose('Drop duplicates from id colum', verbose)
#         function = DropDuplicates(subset_features=dataframe.columns[FEATURE_ID_COLUMN])
#         dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#     # scale float values
#     for feature in dataframe.columns[FEATURE_ID_COLUMN + 1:]:
#         feature_series = dataframe[feature]
#         # check 0 <= values <= 1
#         if not_between_0_1(feature_series):
#             print_if_verbose(f'Scale values of {feature} between 0 and 1', verbose)
#             function = MinMaxScaler(features=feature)
#             dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#
#         if is_categorical(feature_series):
#             # Binning
#             feature_binning_value = feature + '_' + binning_value
#             if is_multi_categorical(feature_series, divider):
#                 print_if_verbose(
#                     f"Apply binning with threshold: '{binning_threshold}' with divider '{divider}'",
#                     verbose)
#                 function = BinThreshold(feature,
#                                         binning_value=feature_binning_value,
#                                         threshold=binning_threshold,
#                                         divider=divider)
#             else:
#                 print_if_verbose(f'Apply binning with threshold: {feature_binning_value}', verbose)
#                 function = BinThreshold(feature, binning_value=feature_binning_value, threshold=binning_threshold)
#
#             dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#
#             # Encoding
#             if is_multi_categorical(feature_series, divider):
#                 print_if_verbose(f"Apply OneHotEncode to feature '{feature}' with divider '{divider}'", verbose)
#                 function = OneHotEncode(feature, dividers=divider)
#
#             dataframe, preprocess_functions = _apply_and_append(dataframe, function, preprocess_functions)
#
#     return dataframe, preprocess_functions
