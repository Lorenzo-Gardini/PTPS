import numpy as np
import pytest
from pandas import DataFrame
from sklearn import preprocessing

from src.rex.commons import at_least_list
from src.rex.preprocessing import MinMaxScaler
from tests.preprocessing_tests.test_utilities import expected, check_attributes, fit_transform_and_compare
group_by_feature = 'categorical_repeated'
group_by_features = [group_by_feature, 'numerical_repeated']
feature = 'integer'
features = [feature, 'float']

def test_min_max_scaler_one_group_one_feature(self):
    self._transform(self.group_by_feature, self.feature)

def test_min_max_scaler_groups_feature(self):
    self._transform(self.group_by_features, self.feature)

def test_min_max_scaler_one_group_features(self):
    self._transform(self.group_by_feature, self.features)

def test_min_max_scaler_groups_features(self):
    self._transform(self.group_by_features, self.features)

def test_inverse_transform(self):
    scaler = MinMaxScaler(self.group_by_features, self.features).fit(self.dataframe)
    computed = scaler.inverse_transform(scaler.transform(self.dataframe))
    self.assertEqual(self.dataframe, computed)


@pytest.mark.parametrize('features', [feature, features])
@pytest.mark.parametrize('group_by', [group_by_feature, group_by_features])
def _transform(group_by, features, expected):
    computed = self.transform_default_dataframe(MinMaxScaler(group_by, features))
    features = at_least_list(features)
    group_by = at_least_list(group_by)
    groups = self.dataframe.groupby(group_by)
    for feature in features:
        self.dataframe[feature] = groups[[feature]].apply(lambda x: DataFrame(preprocessing.MinMaxScaler().fit_transform(x), index=x.index))
    fit_transform_and_compare(expected, MinMaxScaler(group_by, features))

def _preprocess_function_class(self):
    return MinMaxScaler
